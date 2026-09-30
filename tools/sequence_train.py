"""Train the parity-memory recipe on pre-encoded game sequences
(docs/parity_memory_design_20260929.md "Training"; the records come from
tools/preencode_sequences.py).

`--streams` game-sides run side by side (`wesnoth_ai/sequence_streams.py`);
each optimizer step unrolls `--window` decisions of each, back-propagates
through the memory across them, and carries the memory into the next window
without gradient (truncated back-propagation through time; Williams and
Peng, 1990). Each time step's forward is recomputed in the backward pass
(activation checkpointing), its inputs staged once. The losses are `obs8`'s
policy and value losses and the belief loss (`wesnoth_ai/sequence_loss.py`),
summed over the window's positions and divided by streams x window, so a
short last window steps at the same rate per position. AdamW, a linear
warm-up, then a constant rate, one pass.

Always-on telemetry (user ruling 2026-09-01): every `--signal-every`
positions a row in <out>.signal.jsonl splits the last window's gradient on
its first SIGNAL_STREAMS slots by loss term (the four policy heads, the
value, the belief) over the encoder, the trunk, the heads and the memory, in
gradient and in update space (`signal_telemetry.GradientProbe`); the log
carries each step's gradient norm and the memory's.

The holdout probe (`tools/sequence_probe.py`) runs every `--probe-every`
positions and at the end; the first probe at or past `--barrier-positions`
is the memory's crash barrier: unless the belief loss at 64 slots is below
the one at 0 slots by more than two standard errors, paired over holdout
games, the run stops (exit 3), and the checkpoint keeps the verdict, so
no resume trains past it. A pass that ends short of every pre-encoded
position exits 4. A checkpoint every
`--checkpoint-every` positions holds the schedule, each slot's carried
memory and the optimizer, so `--resume` continues the pass exactly.

    python tools/sequence_train.py --sequences replays_dataset_sequences \\
        --dataset replays_dataset_imitation --out training/checkpoints/parity_memory.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from tools.preencode_sequences import (ENCODING, load_manifest, read_record,  # noqa: E402
                                       record_path)
from tools.sequence_probe import fit_last_seen_rates, memory_barrier_passes, probe  # noqa: E402
from tools.signal_telemetry import (POLICY_SOURCES, GradientProbe,  # noqa: E402
                                    named_model_parameters, signal_group, summarize_gram)
from wesnoth_ai.checkpoint_structure import checkpoint_structure  # noqa: E402
from wesnoth_ai.constants import OBSERVATION_EPOCH  # noqa: E402
from wesnoth_ai.encoder import GameStateEncoder  # noqa: E402
from wesnoth_ai.imitation_loss import build_imitation_targets, imitation_loss_parts  # noqa: E402
from wesnoth_ai.model import WesnothModel  # noqa: E402
from wesnoth_ai.sequence_loss import GameFacts, belief_loss, step_labels  # noqa: E402
from wesnoth_ai.sequence_streams import GameSide, StreamSchedule, epoch_order  # noqa: E402

log = logging.getLogger("sequence_train")

# The recipe's architecture; the flags of the same names change it (a smoke run).
ARCH = {"d_model": 384, "num_layers": 8, "num_heads": 12, "d_ff": 1536}
MEMORY_SLOTS = 64
# The recipe (docs/parity_memory_design_20260929.md "Training").
DEFAULTS = {"streams": 32, "window": 16, "lr": 2.8e-4, "warmup_steps": 300, "weight_decay": 1e-4,
            "grad_clip": 1.0, "value_states_per_game": 16, "belief_weight": 1.0,
            "seed": 20260929, "probe_every": 500_000, "barrier_positions": 500_000,
            "checkpoint_every": 250_000, "probe_ks": (0, 16, 64), "signal_every": 25_000}
# Game-sides the loader fetches ahead of the slots, and its threads.
PREFETCH_SIDES = 64
LOADER_THREADS = 4
# Training game-sides the last-seen baseline's rates are fitted on.
BASELINE_FIT_SIDES = 400
EXIT_MEMORY_BARRIER = 3
EXIT_PASS_INCOMPLETE = 4
EXIT_NONFINITE = 5
# Non-finite steps in a row that stop the pass (a single one is skipped and counted).
NONFINITE_LIMIT = 3
# The telemetry's slots, terms and parameter groups.
SIGNAL_STREAMS = 4
SIGNAL_TERMS = POLICY_SOURCES + ("value", "belief")
SIGNAL_GROUPS = ("encoder", "trunk", "heads", "memory")


def _grad_norm(params) -> torch.Tensor:
    """The L2 norm of the parameters' gradients, zero when none has one."""
    grads = [p.grad for p in params if p.grad is not None]
    return torch.linalg.vector_norm(torch.stack([g.norm() for g in grads])) if grads else torch.zeros(())


def sequence_signal_group(name: str) -> str:
    """The telemetry group of a namespaced parameter: the learned memory
    (its initial state, slot embedding and write), else the encoder, the
    trunk or the heads."""
    return "memory" if name.startswith("model.slot_memory.") else signal_group(name)


class SequenceLoader:
    """The game records the slots read, fetched ahead on threads (the zlib
    half releases the GIL) and dropped once no slot or upcoming game-side
    needs them."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self._pool = ThreadPoolExecutor(LOADER_THREADS)
        self._records: Dict[str, object] = {}

    def prefetch(self, files) -> None:
        for f in files:
            if f not in self._records:
                self._records[f] = self._pool.submit(read_record, record_path(self.directory, f))

    def sides(self, game_side: GameSide):
        rec = self._records.get(game_side.file)
        if rec is None:
            self.prefetch([game_side.file])
            rec = self._records[game_side.file]
        if not hasattr(rec, "sides"):
            rec = self._records[game_side.file] = rec.result()
        return rec.sides[game_side.side]

    def keep_only(self, files) -> None:
        keep = set(files)
        for f in [f for f in self._records if f not in keep]:
            del self._records[f]


def game_facts(dataset: Path, games: Dict[str, Dict[str, int]]) -> Tuple[Dict[str, GameFacts], List[str], List[str]]:
    """Each pre-encoded game's facts, and the train and holdout files, from
    the dataset's manifest. The policy weight is `obs8`'s per-game weight:
    the median winner's action count over the game's, clipped to [0.25, 4]."""
    rows = [json.loads(line) for line in (Path(dataset) / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    acts = sorted(r["winner_actions"] for r in rows if r["winner_actions"] > 0)
    median = acts[len(acts) // 2] if acts else 1
    facts, train, holdout = {}, [], []
    for r in rows:
        f = r["file"]
        if f not in games:
            continue
        w = max(0.25, min(4.0, median / r["winner_actions"])) if r["winner_actions"] > 0 else 1.0
        facts[f] = GameFacts(winner=int(r["winner_side"]), n_commands=int(games[f]["n_commands"]),
                             policy_weight=float(w))
        (holdout if r.get("holdout") else train).append(f)
    return facts, sorted(train), sorted(holdout)


def build_modules(unit_type_to_id: Dict[str, int], faction_to_id: Dict[str, int],
                  device: torch.device, arch: Dict[str, int]) -> Tuple[GameStateEncoder, WesnothModel]:
    encoder = GameStateEncoder(d_model=arch["d_model"], relevant_set_hexes=True,
                               fog_hides_enemy_villages=True, terrain_multi_hot=True,
                               observation_parity=True, relevant_set_version=2,
                               unit_type_to_id=unit_type_to_id, faction_to_id=faction_to_id).to(device)
    encoder.freeze_vocab()
    model = WesnothModel(observation_parity=True, memory_slots=MEMORY_SLOTS, **arch).to(device)
    return encoder, model


def save_checkpoint(path: Path, encoder, model, opt, state: Dict, schedule: StreamSchedule,
                    memories: Dict[int, torch.Tensor], meta: Dict, arch: Dict[str, int]) -> None:
    """Atomic: a temporary file, then a rename. Loadable by the policy
    loader (`eval_players.peek_checkpoint_arch`, `TransformerPolicy`)."""
    payload = {
        "observation_epoch": int(OBSERVATION_EPOCH), "arch": dict(arch),
        "relevant_set_hexes": True, "fog_hides_enemy_villages": True, "terrain_multi_hot": True,
        **checkpoint_structure(model, encoder),
        "model_state": model.state_dict(), "encoder_state": encoder.state_dict(),
        "unit_type_to_id": dict(encoder.unit_type_to_id), "faction_to_id": dict(encoder.faction_to_id),
        "optimizer_state": opt.state_dict(), "training_meta": dict(meta),
        "sequence_resume": {"state": dict(state), "schedule": schedule.state_dict(),
                            "memories": {int(s): m.detach().cpu() for s, m in memories.items()},
                            "rng": {"cpu": torch.get_rng_state(),
                                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


class Trainer:
    def __init__(self, args, device: torch.device):
        self.args, self.device = args, device
        man = load_manifest(args.sequences)
        if man is None:
            raise SystemExit(f"{args.sequences}: no sequence manifest (tools/preencode_sequences.py)")
        if man["encoding"] != ENCODING or int(man["observation_epoch"]) != int(OBSERVATION_EPOCH):
            raise SystemExit(f"{args.sequences} was encoded as {man['encoding']} at observation epoch "
                             f"{man['observation_epoch']}; this trainer reads {ENCODING} at {OBSERVATION_EPOCH}")
        self.manifest = man
        self.games, train, holdout = game_facts(args.dataset, man["games"])
        lengths = {GameSide(f, s): int(man["games"][f].get(f"positions_side{s}", 0))
                   for f in train for s in (1, 2)}
        self.holdout = [GameSide(f, s) for f in holdout for s in (1, 2)
                        if int(man["games"][f].get(f"positions_side{s}", 0)) > 0]
        self.schedule = StreamSchedule(epoch_order(list(lengths), args.seed), lengths, args.streams, args.seed)
        self.arch = {k: int(getattr(args, k)) for k in ARCH}
        self.encoder, self.model = build_modules(man["unit_type_to_id"], man["faction_to_id"], device,
                                                 self.arch)
        self.params = [p for p in list(self.encoder.parameters()) + list(self.model.parameters())
                       if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=args.lr, weight_decay=args.weight_decay)
        self.loader = SequenceLoader(args.sequences)
        self.memories: Dict[int, torch.Tensor] = {}
        self.state = {"positions": 0, "steps": 0, "windows": 0, "next_probe": args.probe_every,
                      "barrier_done": False, "next_checkpoint": args.checkpoint_every,
                      "next_signal": args.signal_every}
        self.signal = GradientProbe(named_model_parameters(self.model, self.encoder),
                                    sequence_signal_group, SIGNAL_GROUPS, self.opt)
        self.rates: Optional[Tuple[float, float]] = None
        self.start_positions = 0                 # where this process's run began (the logged rate)
        self.autocast = torch.bfloat16 if (device.type == "cuda" and not args.fp32) else None
        from tools.supervised_train import _DEFAULT_ACTION_TYPE_LOSS_WEIGHT
        self.type_loss_weights = dict(_DEFAULT_ACTION_TYPE_LOSS_WEIGHT)
        self.value_weight = float(json.loads(Path(args.imitation_config).read_text(encoding="utf-8"))
                                  .get("value_from_outcome_weight", 1.0))
        self.meta = {"seed": args.seed, "streams": args.streams, "window": args.window, "lr": args.lr,
                     "warmup_steps": args.warmup_steps, "sequences": str(args.sequences),
                     "fingerprint": man["fingerprint"], "value_weight": self.value_weight,
                     "arch": dict(self.arch)}
        self.head_shapes: Optional[Tuple[int, int, int]] = None
        log.info("%d training game-sides (%d positions), %d holdout game-sides, %d streams x %d decisions",
                 len(self.schedule.order), self.schedule.total_positions, len(self.holdout),
                 args.streams, args.window)

    # ---- resume ---------------------------------------------------------
    def resume(self, path: Path) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        meta = ck.get("training_meta", {})
        for key in ("seed", "streams", "window", "fingerprint", "arch"):
            if meta.get(key) != self.meta[key]:
                raise SystemExit(f"{path} was trained with {key}={meta.get(key)!r}; this run has "
                                 f"{self.meta[key]!r}")
        self.model.load_state_dict(ck["model_state"])
        self.encoder.load_state_dict(ck["encoder_state"])
        self.opt.load_state_dict(ck["optimizer_state"])
        saved = ck["sequence_resume"]
        self.state.update(saved["state"])
        self.schedule.load_state_dict(saved["schedule"])
        self.memories = {int(s): m.to(self.device) for s, m in saved["memories"].items()}
        torch.set_rng_state(saved["rng"]["cpu"])
        if saved["rng"]["cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
        log.info("resumed at %d positions, %d steps", self.state["positions"], self.state["steps"])

    # ---- one window -----------------------------------------------------
    def _lr_now(self) -> float:
        return self.args.lr * min(1.0, (self.state["steps"] + 1) / max(1, self.args.warmup_steps))

    def _head_shapes(self, staged, mems) -> Tuple[int, int, int]:
        if self.head_shapes is None:
            # A forward outside the training's random stream (dropout), so a
            # resumed run draws what the uncut run draws.
            devices = [self.device.index or 0] if self.device.type == "cuda" else []
            with torch.no_grad(), torch.random.fork_rng(devices=devices):
                out = self.model.forward_embedded(self.encoder.embed_staged(staged), memory=mems)
            self.head_shapes = (out.type_logits.shape[2], out.weapon_logits.shape[2], out.value_logits.shape[1])
        return self.head_shapes

    def _step_forward(self, staged, mems, targets, b_target, b_mask):
        """One time step's forward and losses; recomputed in the backward pass."""
        with torch.autocast(self.device.type, dtype=self.autocast, enabled=self.autocast is not None):
            streams = self.encoder.embed_staged(staged)
            out = self.model.forward_embedded(streams, memory=mems)
        out = out.float32()
        parts = imitation_loss_parts(out, targets)
        bl = belief_loss(out.belief_logits, b_target, b_mask)
        return parts.total, bl.sum(), out.memory_padded, parts.log_tensor(), bl.detach()

    def _prepare(self, steps, memories: Dict[int, torch.Tensor]):
        """One time step's inputs: the staged batch, each slot's memory (the
        learned initial one at a game-side's first decision), the imitation
        targets and the belief targets."""
        positions = [self.loader.sides(s.game_side)[s.offset] for s in steps]
        staged = self.encoder.stage_raws([p.raw for p in positions], device=self.device)
        mems = [self.model.initial_memory(s.k).to(self.device) if s.starts else memories[s.slot]
                for s in steps]
        labels = step_labels(positions, steps, staged.sizes, self.games, seed=self.args.seed,
                             value_states_per_game=self.args.value_states_per_game,
                             value_weight=self.value_weight)
        n_types, n_weapons, n_atoms = self._head_shapes(staged, mems)
        targets = build_imitation_targets(labels.ais, labels.zw, staged.sizes, n_types=n_types,
                                          n_weapons=n_weapons, n_atoms=n_atoms,
                                          type_loss_weights=self.type_loss_weights, device=self.device)
        b_target = labels.belief_target.to(self.device, non_blocking=True)
        b_mask = labels.belief_mask.to(self.device, non_blocking=True)
        return staged, mems, targets, b_target, b_mask

    def train_window(self) -> Dict[str, float]:
        window = self.schedule.window(self.args.window)
        if not window:
            return {}
        self.loader.prefetch(g.file for g in self.schedule.upcoming(PREFETCH_SIDES))
        start_memories = {s: m for s, m in self.memories.items() if s < SIGNAL_STREAMS}
        total = torch.zeros((), device=self.device)
        sums = {"policy": 0.0, "policy_n": 0, "value": 0.0, "value_n": 0, "belief": 0.0, "belief_n": 0}
        logs = []
        n_positions = 0
        for steps in window:
            staged, mems, targets, b_target, b_mask = self._prepare(steps, self.memories)
            pv, bsum, mem_padded, log_t, bl = checkpoint(self._step_forward, staged, mems, targets,
                                                         b_target, b_mask, use_reentrant=False)
            total = total + pv + self.args.belief_weight * bsum
            for b, s in enumerate(steps):
                self.memories[s.slot] = mem_padded[b, :s.k]
            logs.append((targets, log_t, bl))
            n_positions += len(steps)
        loss = total / float(self.args.streams * self.args.window)
        for g in self.opt.param_groups:
            g["lr"] = self._lr_now()
        loss.backward()
        write_norm = _grad_norm([*self.model.slot_memory.gate.parameters(),
                                 *self.model.slot_memory.candidate.parameters()])
        norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_clip)
        finite = bool(torch.isfinite(norm)) and bool(torch.isfinite(loss.detach()))
        if finite:
            self.opt.step()
            self.state["nonfinite_run"] = 0
        else:
            # The update is skipped and counted; a slot whose carried memory
            # is not finite starts again from the learned initial memory.
            self.state["nonfinite_steps"] = self.state.get("nonfinite_steps", 0) + 1
            self.state["nonfinite_run"] = self.state.get("nonfinite_run", 0) + 1
            log.warning("non-finite step %d (loss %s, gradient norm %s): update skipped",
                        self.state["steps"], float(loss.detach()), float(norm))
        self.opt.zero_grad(set_to_none=True)
        carried = {}
        for s, m in self.memories.items():
            if bool(torch.isfinite(m).all()):
                carried[s] = m.detach()
            else:                        # counted: the slot's game-side goes on from the initial memory
                self.state["memory_resets"] = self.state.get("memory_resets", 0) + 1
                log.warning("slot %d's carried memory is not finite at step %d: reset", s, self.state["steps"])
                carried[s] = self.model.initial_memory(m.shape[0]).detach().to(self.device)
        self.memories = carried
        for targets, log_t, bl in logs:
            vals = log_t.cpu()
            pw = torch.from_numpy(targets.policy_w)
            ok_policy = pw > 0
            sums["policy"] += float(vals[:4].sum(0)[ok_policy].sum())
            sums["policy_n"] += int(ok_policy.sum())
            ok_value = torch.from_numpy(targets.ok["value"])
            sums["value"] += float(vals[4][ok_value].sum())
            sums["value_n"] += int(ok_value.sum())
            sums["belief"] += float(bl.sum())
            sums["belief_n"] += int(bl.numel())
        self.state["positions"] += n_positions
        self.state["steps"] += 1
        self.state["windows"] += 1
        if self.state["positions"] >= self.state["next_signal"]:
            self.signal_row(window, start_memories)
            self.state["next_signal"] += self.args.signal_every
        held = {g.file for g in self.schedule.upcoming(PREFETCH_SIDES)}
        self.loader.keep_only(held)
        return {"loss": float(loss.detach()), "grad_norm": float(norm), "memory_write_grad_norm": float(write_norm),
                "positions": n_positions, **sums}

    def signal_row(self, window, start_memories: Dict[int, torch.Tensor]) -> None:
        """The telemetry row: the window just trained, on its first
        SIGNAL_STREAMS slots from the memories they started it with, its loss
        split by term in gradient and update space (the optimizer's state
        read, nothing written, the training's random stream untouched)."""
        t0 = time.time()
        memories = dict(start_memories)
        terms = {t: torch.zeros((), device=self.device) for t in SIGNAL_TERMS}
        n = 0
        with self.signal.fork_rng():
            for steps in window:
                steps = [s for s in steps if s.slot < SIGNAL_STREAMS]
                if not steps:
                    continue
                staged, mems, targets, b_target, b_mask = self._prepare(steps, memories)
                with torch.autocast(self.device.type, dtype=self.autocast, enabled=self.autocast is not None):
                    out = self.model.forward_embedded(self.encoder.embed_staged(staged), memory=mems)
                out = out.float32()
                for term, loss in imitation_loss_parts(out, targets).source_losses().items():
                    terms[term] = terms[term] + loss
                terms["belief"] = terms["belief"] + self.args.belief_weight * \
                    belief_loss(out.belief_logits, b_target, b_mask).sum()
                for b, s in enumerate(steps):
                    memories[s.slot] = out.memory_padded[b, :s.k]
                n += len(steps)
            if not n:
                return
            gradient, update, stateless = self.signal.grams(terms, 1.0 / n)
        kw = dict(policy_terms=POLICY_SOURCES, shared_groups=("encoder", "trunk", "memory"))
        row = {"positions": self.state["positions"], "steps": self.state["steps"], "probe_positions": n,
               "gradient": summarize_gram(gradient.tolist(), SIGNAL_TERMS, SIGNAL_GROUPS, **kw),
               "update": None if update is None else summarize_gram(update.tolist(), SIGNAL_TERMS,
                                                                    SIGNAL_GROUPS, **kw),
               "stateless": stateless, "seconds": round(time.time() - t0, 2)}
        with open(self.args.out.with_suffix(".signal.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    # ---- probe, checkpoints, the pass -----------------------------------
    def run_probe(self, final: bool = False) -> Dict:
        if self.rates is None:
            order = self.schedule.order[:BASELINE_FIT_SIDES]
            self.rates = fit_last_seen_rates(read_record(record_path(self.args.sequences, g.file)).sides[g.side]
                                             for g in order)
        holdout_loader = SequenceLoader(self.args.sequences)
        winners = {f: g.winner for f, g in self.games.items()}
        t0 = time.time()
        result = probe(self.model, self.encoder, self.holdout, holdout_loader.sides, winners,
                       self.args.probe_ks, self.device, self.type_loss_weights, rates=self.rates,
                       autocast_dtype=self.autocast)
        result.update(positions=self.state["positions"], steps=self.state["steps"],
                      total_positions=self.schedule.total_positions, final=final,
                      seconds=round(time.time() - t0, 1))
        with open(self.args.out.with_suffix(".probe.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        log.info("PROBE %s", json.dumps(result))
        return result

    def save(self) -> None:
        save_checkpoint(self.args.out, self.encoder, self.model, self.opt, self.state, self.schedule,
                        self.memories, self.meta, self.arch)

    def run(self) -> int:
        if self.state.get("barrier_failed"):
            log.error("MEMORY_BARRIER_FAILED earlier in this pass (the .probe.jsonl): no resume past it")
            return EXIT_MEMORY_BARRIER
        t0, last_log = time.time(), time.time()
        self.start_positions = self.state["positions"]
        window_logs: List[Dict] = []
        limit = self.args.max_positions
        while not self.schedule.exhausted() and (limit is None or self.state["positions"] < limit):
            window_logs.append(self.train_window())
            if self.state.get("nonfinite_run", 0) >= NONFINITE_LIMIT:
                self._log(window_logs, t0)
                log.error("NONFINITE_TRAINING: %d non-finite steps in a row at %d positions",
                          self.state["nonfinite_run"], self.state["positions"])
                return EXIT_NONFINITE
            if time.time() - last_log >= self.args.log_seconds:
                self._log(window_logs, t0)
                window_logs, last_log = [], time.time()
            if self.state["positions"] >= self.state["next_checkpoint"]:
                self.state["next_checkpoint"] += self.args.checkpoint_every
                self.save()
            if self.state["positions"] >= self.state["next_probe"]:
                result = self.run_probe()
                self.state["next_probe"] += self.args.probe_every
                barrier_now = (not self.state["barrier_done"]
                               and self.state["positions"] >= self.args.barrier_positions)
                if barrier_now:
                    self.state["barrier_done"] = True
                    self.state["barrier_failed"] = not memory_barrier_passes(result)
                self.save()                      # the probe is kept, so a resume does not repeat it
                if barrier_now and self.state["barrier_failed"]:
                    log.error("MEMORY_BARRIER_FAILED %s", json.dumps(result.get("belief_paired")))
                    return EXIT_MEMORY_BARRIER
                if barrier_now:
                    log.info("memory barrier passed: %s", json.dumps(result.get("belief_paired")))
        if window_logs:
            self._log(window_logs, t0)
        self.save()
        if limit is not None and self.state["positions"] < self.schedule.total_positions:
            log.info("SEQUENCE_TRAIN_CUT %d positions of %d (--max-positions %d)", self.state["positions"],
                     self.schedule.total_positions, limit)
            return 0
        if self.state["positions"] != self.schedule.total_positions:
            log.error("PASS_INCOMPLETE: %d positions trained of the %d pre-encoded", self.state["positions"],
                      self.schedule.total_positions)
            return EXIT_PASS_INCOMPLETE
        self.run_probe(final=True)
        log.info("SEQUENCE_TRAIN_DONE %d positions, %d steps in %.0f s (%d non-finite steps skipped, "
                 "%d memories reset)", self.state["positions"], self.state["steps"], time.time() - t0,
                 self.state.get("nonfinite_steps", 0), self.state.get("memory_resets", 0))
        return 0

    def _log(self, rows: List[Dict], t0: float) -> None:
        rows = [r for r in rows if r]
        if not rows:
            return
        tot = {k: sum(r[k] for r in rows) for k in rows[0] if k not in ("loss", "grad_norm", "memory_write_grad_norm")}
        el = time.time() - t0
        log.info("positions %d/%d steps %d | loss %.4f grad %.2f memory write grad %.3f | policy %.4f value %.4f "
                 "belief %.4f | lr %.2e | %.1f positions/s", self.state["positions"],
                 self.schedule.total_positions, self.state["steps"], sum(r["loss"] for r in rows) / len(rows),
                 max(r["grad_norm"] for r in rows), max(r["memory_write_grad_norm"] for r in rows),
                 tot["policy"] / max(1, tot["policy_n"]), tot["value"] / max(1, tot["value_n"]),
                 tot["belief"] / max(1, tot["belief_n"]), self._lr_now(),
                 (self.state["positions"] - self.start_positions) / max(el, 1e-9))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sequences", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True, help="the corpus directory with manifest.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true", help="continue from --out")
    ap.add_argument("--imitation-config", type=Path, default=Path("configs/imitation.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp32", action="store_true", help="no bfloat16 autocast on CUDA")
    ap.add_argument("--max-positions", type=int, default=None, help="stop after this many (a smoke run)")
    ap.add_argument("--log-seconds", type=float, default=60.0)
    for key in ("streams", "window", "warmup_steps", "value_states_per_game", "seed", "probe_every",
                "barrier_positions", "checkpoint_every", "signal_every"):
        ap.add_argument("--" + key.replace("_", "-"), type=int, default=DEFAULTS[key])
    for key, value in ARCH.items():
        ap.add_argument("--" + key.replace("_", "-"), type=int, default=value)
    for key in ("lr", "weight_decay", "grad_clip", "belief_weight"):
        ap.add_argument("--" + key.replace("_", "-"), type=float, default=DEFAULTS[key])
    ap.add_argument("--probe-ks", type=lambda s: tuple(int(x) for x in s.split(",")),
                    default=DEFAULTS["probe_ks"])
    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    trainer = Trainer(args, torch.device(args.device))
    if args.resume:
        trainer.resume(args.out)
    return trainer.run()


if __name__ == "__main__":
    raise SystemExit(main())
