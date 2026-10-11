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
gradient and in update space (`signal_telemetry.GradientProbe`). Every step
writes a row in <out>.steps.jsonl: its losses, its learning rate, and per
parameter group (wesnoth_ai/param_groups.py: the heads one by one, the
memory's initial state, slot embedding and write) the norm of its gradient
before the clip, read off the clip's own computation; the log line carries
the memory's share of it.

The holdout probe (`tools/sequence_probe.py`) runs every `--probe-every`
positions and at the end; the first probe at or past `--barrier-positions`
is the memory's crash barrier: unless the belief loss at 64 slots with the
memory carried is below the one with it reset at every decision by more
than two standard errors, paired over holdout games, the run stops (exit
3), and the checkpoint keeps the verdict, so no resume trains past it. A pass that ends short of every pre-encoded
position exits 4. A checkpoint every
`--checkpoint-every` positions holds the schedule, each slot's carried
memory and the optimizer, so `--resume` continues the pass exactly.

`--init-from CKPT` starts a new pass from another pass's weights and
optimizer state (the same encoding, streams, window and architecture),
on a schedule of this run's seed. `--decay-from F` holds the learning
rate until the fraction F of the pass, then lowers it linearly to 0 at
the pass's end (the cooldown of a warmup-stable-decay schedule, Hagele et
al., "Scaling Laws and Compute-Optimal Training Beyond Fixed Training
Durations", 2024); the checkpoint where the cooldown starts is kept as
`<out>.stable.pt`, from which a further pass can start.

`--pass-positions N` ends the pass after the first N positions of its order;
the learning rate's schedule, `--decay-from` included, runs over those N (a
lowering of the rate over part of an epoch).

`--own-sequences DIR` trains the memory player's own game-sides beside the
corpus's on `--own-streams` of the slots, toward a teacher's distribution
(tools/sequence_own.py, docs/memory_own_trajectories_prereg_20261011.md);
the telemetry then splits that loss out as the term "own". Without it the
trainer runs as it did before own streams existed. `--init-other-records`
lets `--init-from` start from a checkpoint whose records another core phase
or corpus version built, provided its vocabularies, encoding and
observation epoch are this run's. `--probe-at-start` probes the starting
weights before the first step.

`--anneal-rule T` runs the rule of tools/lr_law.py after every probe: the
law of the holdout loss against the learning-rate history, fitted to this
pass's probes and the earlier passes' (`--law-points`). While one more epoch
at the peak rate would lower the fitted loss by more than T, the pass goes
on; otherwise it stops with exit 6 (lower the rate now), or with exit 7 when
its latest probes stopped following the law (stop and look). The checkpoint
keeps the decision, so a resume stops again. The trainer keeps its rate
history's areas (S1 and S2) in its state and in every probe row; a new pass
takes them from the checkpoint it starts from, or from `--initial-areas` for
a checkpoint written before they were kept.

    python tools/sequence_train.py --sequences replays_dataset_sequences \\
        --dataset replays_dataset_imitation --out training/checkpoints/parity_memory.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from tools.lr_law import Areas, probe_points, read_points  # noqa: E402
from tools.lr_law import decide as law_decide  # noqa: E402
from tools.preencode_sequences import ENCODING, load_manifest, read_record, record_path  # noqa: E402
from tools.sequence_checkpoints import TrainerCheckpoints  # noqa: E402
from tools.sequence_own import OwnStreams  # noqa: E402
from tools.sequence_probe import fit_last_seen_rates, memory_barrier_passes, probe  # noqa: E402
from tools.sequence_setup import (ARCH, DEFAULTS, PREFETCH_SIDES, SequenceLoader,  # noqa: E402
                                  build_modules, game_facts, lr_factor, own_stream_count, save_checkpoint)
from tools.sequence_telemetry import (SIGNAL_STREAMS, SIGNAL_TERMS, TrainerTelemetry,  # noqa: E402
                                      _squares)
from wesnoth_ai.constants import OBSERVATION_EPOCH  # noqa: E402
from wesnoth_ai.imitation_loss import build_imitation_targets, imitation_loss_parts  # noqa: E402
from wesnoth_ai.own_loss import own_policy_loss  # noqa: E402
from wesnoth_ai.param_groups import GradientGroups, named_as  # noqa: E402
from wesnoth_ai.sequence_loss import belief_loss, step_labels  # noqa: E402
from wesnoth_ai.sequence_streams import GameSide, StreamSchedule, epoch_order, merge_windows  # noqa: E402

log = logging.getLogger("sequence_train")

# Training game-sides the last-seen baseline's rates are fitted on.
BASELINE_FIT_SIDES = 400
EXIT_MEMORY_BARRIER = 3
EXIT_PASS_INCOMPLETE = 4
EXIT_NONFINITE = 5
EXIT_LOWER = 6                   # the anneal rule: lower the rate from here
EXIT_REVIEW = 7                  # the anneal rule: the probes left the law, stop and look
# Non-finite steps in a row that stop the pass (a single one is skipped and counted).
NONFINITE_LIMIT = 3


class Trainer(TrainerCheckpoints, TrainerTelemetry):
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
        own_streams = own_stream_count(args)
        self.schedule = StreamSchedule(epoch_order(list(lengths), args.seed), lengths, args.streams - own_streams,
                                       args.seed)
        self.own: Optional[OwnStreams] = None
        if own_streams:
            self.own = OwnStreams(args.own_sequences, man, args.seed, own_streams,
                                  slot_offset=args.streams - own_streams)
            self.games.update(self.own.games)
        total = self.schedule.total_positions + (self.own.schedule.total_positions if self.own else 0)
        self.pass_total = total if args.pass_positions is None else min(int(args.pass_positions), total)
        if self.own is not None:
            self._check_own_shares()
        self.arch = {k: int(getattr(args, k)) for k in ARCH}
        self.encoder, self.model = build_modules(man["unit_type_to_id"], man["faction_to_id"], device,
                                                 self.arch)
        self.params = [p for p in list(self.encoder.parameters()) + list(self.model.parameters())
                       if p.requires_grad]
        self.clip_groups = GradientGroups(named_as(self.params, self.model, self.encoder))
        self.opt = torch.optim.AdamW(self.params, lr=args.lr, weight_decay=args.weight_decay)
        self.loader = (SequenceLoader(args.sequences) if self.own is None
                       else SequenceLoader(args.sequences, self.own.directory, self.own.files))
        # The telemetry's slots and terms: with own streams, the first slots
        # of each kind and the own loss as a term of its own.
        half = SIGNAL_STREAMS // 2
        self.signal_slots = (set(range(SIGNAL_STREAMS)) if self.own is None else
                             set(range(half)) | set(range(args.streams - own_streams,
                                                          args.streams - own_streams + half)))
        self.signal_terms = SIGNAL_TERMS + (() if self.own is None else ("own",))
        self.memories: Dict[int, torch.Tensor] = {}
        self.state = {"positions": 0, "steps": 0, "windows": 0, "next_probe": args.probe_every,
                      "barrier_done": False, "next_checkpoint": args.checkpoint_every,
                      "next_signal": args.signal_every, "areas": Areas().to_dict()}
        self.signal = self._signal_probe()
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
                     "arch": dict(self.arch), "decay_from": args.decay_from, "init_from": None,
                     "pass_positions": args.pass_positions, "anneal_rule": args.anneal_rule,
                     "own": None if self.own is None else self.own.meta(args.own_policy_weight)}
        self.head_shapes: Optional[Tuple[int, int, int]] = None
        log.info("%d training game-sides (%d positions), %d holdout game-sides, %d streams x %d decisions",
                 len(self.schedule.order), self.schedule.total_positions, len(self.holdout),
                 args.streams, args.window)
        if self.own is not None:
            log.info("own: %d training game-sides (%d positions) on %d of the streams at %d slots, %d holdout "
                     "game-sides", len(self.own.schedule.order), self.own.schedule.total_positions, own_streams,
                     self.own.slots, len(self.own.holdout))

    def _check_own_shares(self) -> None:
        """Each kind of game-side holds its share of the pass, so every step
        mixes them in the slots' proportion to the pass's end."""
        for name, schedule in (("corpus", self.schedule), ("own", self.own.schedule)):
            share = self.pass_total * schedule.n_streams / self.args.streams
            if schedule.total_positions < share:
                raise SystemExit(f"the {name} game-sides hold {schedule.total_positions} positions, fewer than "
                                 f"their {share:.0f} of a pass of {self.pass_total}: pass --pass-positions")

    def _window(self):
        """The next window's time steps, the own slots' beside the corpus's."""
        if self.own is None:
            return self.schedule.window(self.args.window)
        return merge_windows(self.schedule.window(self.args.window), self.own.schedule.window(self.args.window))

    def _upcoming(self) -> List[GameSide]:
        upcoming = self.schedule.upcoming(PREFETCH_SIDES)
        return upcoming if self.own is None else upcoming + self.own.schedule.upcoming(PREFETCH_SIDES)

    def _exhausted(self) -> bool:
        return self.schedule.exhausted() or (self.own is not None and self.own.schedule.exhausted())

    # ---- one window -----------------------------------------------------
    def _lr_now(self) -> float:
        warm = min(1.0, (self.state["steps"] + 1) / max(1, self.args.warmup_steps))
        return self.args.lr * warm * lr_factor(self.state["positions"], self.pass_total, self.args.decay_from)

    def _areas_step(self, lr: float) -> None:
        """The rate history's areas (tools/lr_law.py) after a step applied at `lr`."""
        if self.state.get("areas") is None:
            return
        areas = Areas.from_dict(self.state["areas"])
        areas.step(lr, warming=(self.state["steps"] + 1) < self.args.warmup_steps)
        self.state["areas"] = areas.to_dict()

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

    def _step_forward_own(self, staged, mems, targets, b_target, b_mask, own):
        """`_step_forward` for a time step that holds own rows: their
        soft-target loss joins the policy and value losses, and comes back
        per row for the log."""
        with torch.autocast(self.device.type, dtype=self.autocast, enabled=self.autocast is not None):
            streams = self.encoder.embed_staged(staged)
            out = self.model.forward_embedded(streams, memory=mems)
        out = out.float32()
        parts = imitation_loss_parts(out, targets)
        bl = belief_loss(out.belief_logits, b_target, b_mask)
        soft = own_policy_loss(out, own)
        return (parts.total + self.args.own_policy_weight * soft.sum(), bl.sum(), out.memory_padded,
                parts.log_tensor(), bl.detach(), soft.detach())

    def _own_batch(self, steps, staged):
        """The step's own rows staged for their loss (after `_prepare`, which
        staged the step), None without any."""
        if self.own is None:
            return None
        positions = [self.loader.sides(s.game_side)[s.offset] for s in steps]
        n_types, n_weapons, _ = self.head_shapes
        return self.own.batch(steps, positions, staged.sizes, n_types, n_weapons, self.device)

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
        window = self._window()
        if not window:
            return {}
        self.loader.prefetch(g.file for g in self._upcoming())
        start_memories = {s: m for s, m in self.memories.items() if s in self.signal_slots}
        total = torch.zeros((), device=self.device)
        sums = {"policy": 0.0, "policy_n": 0, "value": 0.0, "value_n": 0, "belief": 0.0, "belief_n": 0}
        if self.own is not None:
            sums.update(own=0.0, own_n=0)
        logs = []
        n_positions = 0
        for steps in window:
            staged, mems, targets, b_target, b_mask = self._prepare(steps, self.memories)
            own = self._own_batch(steps, staged)
            if own is None:
                pv, bsum, mem_padded, log_t, bl = checkpoint(self._step_forward, staged, mems, targets,
                                                             b_target, b_mask, use_reentrant=False)
            else:
                pv, bsum, mem_padded, log_t, bl, own_ce = checkpoint(
                    self._step_forward_own, staged, mems, targets, b_target, b_mask, own, use_reentrant=False)
                sums["own"] += float(own_ce.sum())
                sums["own_n"] += int(own_ce.numel())
            total = total + pv + self.args.belief_weight * bsum
            for b, s in enumerate(steps):
                self.memories[s.slot] = mem_padded[b, :s.k]
            logs.append((targets, log_t, bl))
            n_positions += len(steps)
        loss = total / float(self.args.streams * self.args.window)
        lr = self._lr_now()
        for g in self.opt.param_groups:
            g["lr"] = lr
        loss.backward()
        norm, grads = self.clip_groups.clip(self.args.grad_clip)
        finite = bool(torch.isfinite(norm)) and bool(torch.isfinite(loss.detach()))
        if finite:
            self.opt.step()
            self._areas_step(lr)
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
        held = {g.file for g in self._upcoming()}
        self.loader.keep_only(held)
        self._step_row(lr, float(loss.detach()), float(norm), grads, finite, sums)
        return {"loss": float(loss.detach()), "grad_norm": float(norm),
                "memory_write_grad_norm": grads.get("memory_write", 0.0), "positions": n_positions,
                **_squares(grads), **sums}

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
                      total_positions=self.pass_total, final=final,
                      seconds=round(time.time() - t0, 1))
        if self.state.get("areas") is not None:
            result["areas"] = {"s1": self.state["areas"]["s1"], "s2": self.state["areas"]["s2"]}
        with open(self.args.out.with_suffix(".probe.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        log.info("PROBE %s", json.dumps(result))
        return result

    def save(self) -> None:
        save_checkpoint(self.args.out, self.encoder, self.model, self.opt, self.state, self.schedule,
                        self.memories, self.meta, self.arch, self._own_schedule())

    def _own_schedule(self) -> Optional[StreamSchedule]:
        return None if self.own is None else self.own.schedule

    def _keep_stable(self) -> None:
        """Once the cooldown starts, the checkpoint there is kept as
        `<out>.stable.pt`: a further pass starts from the weights before
        the cooldown."""
        f = self.args.decay_from
        # A pass that lowers from its first position keeps none: it started from that checkpoint.
        if f is None or f == 0 or self.state.get("stable_saved") \
                or self.state["positions"] < f * self.pass_total:
            return
        self.state["stable_saved"] = True
        stable = self.args.out.with_name(self.args.out.stem + ".stable.pt")
        save_checkpoint(stable, self.encoder, self.model, self.opt, self.state, self.schedule,
                        self.memories, self.meta, self.arch, self._own_schedule())
        self.save()                              # the flag only once the stable file exists
        log.info("the cooldown starts at %d positions: the checkpoint is kept as %s",
                 self.state["positions"], stable)

    def run(self) -> int:
        if self.state.get("barrier_failed"):
            log.error("MEMORY_BARRIER_FAILED earlier in this pass (the .probe.jsonl): no resume past it")
            return EXIT_MEMORY_BARRIER
        if self.state.get("anneal"):
            log.info("ANNEAL_RULE decided %r earlier in this pass (the .anneal.jsonl): no training past it",
                     self.state["anneal"])
            return EXIT_LOWER if self.state["anneal"] == "lower" else EXIT_REVIEW
        if self.args.probe_at_start and not self.state.get("start_probed") and self.state["positions"] == 0:
            self.run_probe()
            self.state["start_probed"] = True
            self.save()
        t0, last_log = time.time(), time.time()
        self.start_positions = self.state["positions"]
        window_logs: List[Dict] = []
        limit = self.args.max_positions
        while (not self._exhausted() and self.state["positions"] < self.pass_total
               and (limit is None or self.state["positions"] < limit)):
            window_logs.append(self.train_window())
            if self.state.get("nonfinite_run", 0) >= NONFINITE_LIMIT:
                self._log(window_logs, t0)
                log.error("NONFINITE_TRAINING: %d non-finite steps in a row at %d positions",
                          self.state["nonfinite_run"], self.state["positions"])
                return EXIT_NONFINITE
            if time.time() - last_log >= self.args.log_seconds:
                self._log(window_logs, t0)
                window_logs, last_log = [], time.time()
            self._keep_stable()
            if self.state["positions"] >= self.state["next_checkpoint"]:
                self.state["next_checkpoint"] += self.args.checkpoint_every
                if self.state["positions"] < self.state["next_probe"]:
                    self.save()              # a probe due now saves after it: a resume repeats it at this position
            if self.state["positions"] >= self.state["next_probe"]:
                result = self.run_probe()
                self.state["next_probe"] += self.args.probe_every
                barrier_now = (not self.state["barrier_done"]
                               and self.state["positions"] >= self.args.barrier_positions)
                if barrier_now:
                    self.state["barrier_done"] = True
                    self.state["barrier_failed"] = not memory_barrier_passes(result)
                code = None if self.state.get("barrier_failed") else self._anneal_rule()
                self.save()                      # the probe and its decisions together: a resume repeats neither
                if barrier_now and self.state["barrier_failed"]:
                    log.error("MEMORY_BARRIER_FAILED carried against reset %s; against 0 slots %s",
                              json.dumps(result.get("belief_carried")), json.dumps(result.get("belief_paired")))
                    return EXIT_MEMORY_BARRIER
                if barrier_now:
                    log.info("memory barrier passed: carried against reset %s; against 0 slots %s",
                             json.dumps(result.get("belief_carried")), json.dumps(result.get("belief_paired")))
                if code is not None:
                    if window_logs:
                        self._log(window_logs, t0)
                    return code
        if window_logs:
            self._log(window_logs, t0)
        self.save()
        if limit is not None and self.state["positions"] < self.pass_total:
            log.info("SEQUENCE_TRAIN_CUT %d positions of %d (--max-positions %d)", self.state["positions"],
                     self.pass_total, limit)
            return 0
        if self.state["positions"] < self.pass_total:
            log.error("PASS_INCOMPLETE: %d positions trained of the pass's %d", self.state["positions"],
                      self.pass_total)
            return EXIT_PASS_INCOMPLETE
        self.run_probe(final=True)
        log.info("SEQUENCE_TRAIN_DONE %d positions, %d steps in %.0f s (%d non-finite steps skipped, "
                 "%d memories reset)", self.state["positions"], self.state["steps"], time.time() - t0,
                 self.state.get("nonfinite_steps", 0), self.state.get("memory_resets", 0))
        return 0

    def _anneal_rule(self) -> Optional[int]:
        """After a probe: the rule of tools/lr_law.py on this pass's probes
        and the earlier passes'; the exit code when it lowers the rate or
        stops the run, None while the rate holds. A decision is set in the
        state, which the caller saves with the probe."""
        if self.args.anneal_rule is None:
            return None
        this_pass = probe_points(self.args.out.with_suffix(".probe.jsonl"), self.args.law_key, source="this pass")
        epoch_steps = self.schedule.total_positions / float(self.args.streams * self.args.window)
        decision = law_decide(read_points(self.args.law_points), this_pass, self.state["areas"]["s1"],
                              self.args.lr, epoch_steps, self.args.anneal_rule)
        row = {"positions": self.state["positions"], "steps": self.state["steps"], **decision.to_dict()}
        with open(self.args.out.with_suffix(".anneal.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        log.info("ANNEAL_RULE %s", json.dumps(row))
        if decision.action == "hold":
            return None
        self.state["anneal"] = decision.action
        return EXIT_LOWER if decision.action == "lower" else EXIT_REVIEW



def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sequences", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True, help="the corpus directory with manifest.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true", help="continue from --out")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="start a new pass from this checkpoint's weights and optimizer state")
    ap.add_argument("--decay-from", type=float, default=None,
                    help="hold the learning rate until this fraction of the pass, then lower it linearly "
                         "to 0 at its end; the checkpoint there is kept as <out>.stable.pt")
    ap.add_argument("--pass-positions", type=int, default=None,
                    help="end the pass after this many positions; the rate's schedule runs over them")
    ap.add_argument("--anneal-rule", type=float, default=None,
                    help="after every probe, stop when one more epoch at the peak rate would lower the "
                         "fitted holdout loss by this much or less (tools/lr_law.py; exit 6, or 7 to look)")
    ap.add_argument("--law-points", type=Path, default=None,
                    help="the earlier passes' probe points the anneal rule fits with this pass's")
    ap.add_argument("--law-key", default="k64", help="the probe's slot reading the anneal rule fits")
    ap.add_argument("--initial-areas", default=None,
                    help="with --init-from a checkpoint that does not carry them: its rate history's areas "
                         "as JSON (tools/lr_law.py replay --areas-at)")
    ap.add_argument("--own-sequences", type=Path, default=None,
                    help="the memory player's own game-sides (tools/own_targets.py), trained beside the corpus's")
    ap.add_argument("--own-streams", type=int, default=None,
                    help="with --own-sequences: the streams own game-sides take (default: half of --streams)")
    ap.add_argument("--own-policy-weight", type=float, default=1.0,
                    help="the weight of an own position's soft-target policy loss")
    ap.add_argument("--init-other-records", action="store_true",
                    help="with --init-from: a checkpoint whose records another core phase or corpus version built "
                         "(its vocabularies, encoding and observation epoch must be this run's)")
    ap.add_argument("--probe-at-start", action="store_true", help="probe the starting weights before training")
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
    if args.resume and args.init_from is not None:
        raise SystemExit("--resume continues --out's own pass; --init-from starts a new one: not both")
    if args.decay_from is not None and not 0.0 <= args.decay_from < 1.0:
        raise SystemExit(f"--decay-from {args.decay_from}: a fraction of the pass, from 0 up to 1")
    if args.pass_positions is not None and args.pass_positions <= 0:
        raise SystemExit(f"--pass-positions {args.pass_positions}: a pass of at least one position")
    if args.initial_areas is not None and args.init_from is None:
        raise SystemExit("--initial-areas gives the areas of the checkpoint --init-from starts from")
    if args.init_other_records and args.init_from is None:
        raise SystemExit("--init-other-records qualifies --init-from")
    if args.own_streams is not None and args.own_sequences is None:
        raise SystemExit("--own-streams gives the streams of --own-sequences")
    if args.anneal_rule is not None and (args.law_points is None or not args.law_points.is_file()):
        raise SystemExit(f"--anneal-rule fits the earlier passes' probes: --law-points {args.law_points} "
                         f"is not a file")
    if args.anneal_rule is not None and args.law_key not in {f"k{k}" for k in args.probe_ks}:
        raise SystemExit(f"--law-key {args.law_key}: the probe reads {', '.join(f'k{k}' for k in args.probe_ks)}")
    trainer = Trainer(args, torch.device(args.device))
    if args.resume:
        trainer.resume(args.out)
    elif args.init_from is not None:
        trainer.init_from(args.init_from)
    if args.anneal_rule is not None and trainer.state.get("areas") is None:
        raise SystemExit("--anneal-rule needs the rate history's areas, which the starting checkpoint does "
                         "not carry: pass --initial-areas")
    return trainer.run()


if __name__ == "__main__":
    raise SystemExit(main())
