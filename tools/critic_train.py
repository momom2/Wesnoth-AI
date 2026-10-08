#!/usr/bin/env python3
"""Train one step-1 critic (docs/selfplay_program_20261008.md, "Step 1") on
the positions of tools/critic_positions.py.

    python tools/critic_train.py --positions DIR --source M --view true \\
        --fraction 0.25 --init training/checkpoints/parity3.pt --out OUT/T25.pt

One recipe for every critic: the value and aux losses of wesnoth_ai/critic.py,
AdamW at the parity recipe's peak rate after its warm-up, the rate then
lowered linearly toward 0 at the end of `--max-epochs`; batches drawn from a
fresh order of the training positions each epoch. The training games are the
source's games on the train side of the 95/5 split whose size draw is below
`--fraction`; the holdout games are all of the source's holdout games, for
every fraction. The holdout value loss is read every `--eval-every` positions
and at each epoch's end, and the checkpoint is the one with the lowest
(`<out>`); training stops after `--patience` reads without a new lowest,
after `--max-epochs`, or past `--max-minutes`, and says which.

Written as it goes: `<out stem>.holdout.jsonl` (each read), `<out
stem>.steps.jsonl` (each step: losses, rate, per-group gradient norms before
the clip, wesnoth_ai/param_groups.py), `<out stem>.signal.jsonl` (every
`--signal-every` positions, a probe of the batch just trained split into the
value and aux terms over the encoder, the trunk and the heads, in gradient and
in update space, with the steps' gradient norms since the last row:
tools/signal_telemetry.py), and `<out stem>.summary.json` (rewritten after
each read; DONE in it when training ended).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import queue
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.signal_telemetry import (SIGNAL_GROUPS, GradientProbe, StepNorms,  # noqa: E402
                                    probe_readings, signal_group, write_signal_row)
from wesnoth_ai import critic as cr  # noqa: E402
from wesnoth_ai.critic_data import unpack_raw  # noqa: E402
from wesnoth_ai.param_groups import GradientGroups  # noqa: E402

log = logging.getLogger("critic_train")

# The parity recipe's optimizer (tools/sequence_train.DEFAULTS).
DEFAULTS = {"lr": 2.8e-4, "warmup_steps": 300, "weight_decay": 1e-4, "grad_clip": 1.0, "batch": 64,
            "aux_weight": 0.25, "max_epochs": 4, "evals_per_epoch": 4, "max_eval_every": 50_000,
            "patience": 4, "max_minutes": 40.0, "signal_every": 25_000, "probe_positions": 32,
            "seed": 20261008}
SIGNAL_TERMS = ("value", "aux")
Item = Tuple[bytes, float, float]                 # (packed encoding, z, aux target)


# ---------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------

def manifest_rows(positions: Path) -> List[Dict]:
    """The builder's rows, the last one per game kept (a resumed build
    appends)."""
    rows: Dict[str, Dict] = {}
    for line in (Path(positions) / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["key"]] = row
    return list(rows.values())


def select_games(rows: Sequence[Dict], source: str, fraction: float) -> Tuple[List[Dict], List[Dict]]:
    """(training rows, holdout rows) of a source: the games with positions,
    on each side of the split, the training ones cut to `fraction` by their
    size draw."""
    ok = [r for r in rows if r["source"] == source and r["status"] == "ok" and r["positions"] > 0]
    train = [r for r in ok if r["split"] == "train" and r["size_u"] < fraction]
    hold = [r for r in ok if r["split"] == "holdout"]
    return sorted(train, key=lambda r: r["key"]), sorted(hold, key=lambda r: r["key"])


def read_shard(positions: Path, row: Dict, view: str) -> List[Item]:
    from wesnoth_ai import unpickle
    shard = unpickle.loads((Path(positions) / f"{row['shard']}.{view}.pkl").read_bytes())
    if shard["key"] != row["key"] or len(shard["raws"]) != len(shard["positions"]):
        raise ValueError(f"{row['shard']}.{view}.pkl does not hold {row['key']}'s positions")
    return [(blob, float(p["z"]), float(p["aux"])) for blob, p in zip(shard["raws"], shard["positions"])]


def load_items(positions: Path, rows: Sequence[Dict], view: str, threads: int = 8) -> List[Item]:
    with ThreadPoolExecutor(threads) as pool:
        parts = list(pool.map(lambda r: read_shard(positions, r, view), rows))
    return [item for part in parts for item in part]


# ---------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------

class Feeder:
    """Unpacks the next batches in a thread while the device trains."""

    def __init__(self, items: Sequence[Item], order: np.ndarray, batch: int, depth: int = 4):
        self._q: "queue.Queue" = queue.Queue(maxsize=depth)
        self._thread = threading.Thread(target=self._run, args=(items, order, batch), daemon=True)
        self._thread.start()

    def _run(self, items, order, batch) -> None:
        try:
            for start in range(0, len(order), batch):
                pick = order[start:start + batch]
                self._q.put(unpack_batch([items[i] for i in pick]))
        except BaseException as e:      # noqa: BLE001 - handed to the consumer
            self._q.put(e)
        self._q.put(None)

    def __iter__(self):
        while True:
            got = self._q.get()
            if got is None:
                return
            if isinstance(got, BaseException):
                raise got
            yield got


def unpack_batch(items: Sequence[Item]):
    raws = [unpack_raw(blob) for blob, _, _ in items]
    z = torch.tensor([it[1] for it in items], dtype=torch.float32)
    aux = torch.tensor([it[2] for it in items], dtype=torch.float32)
    return raws, z, aux


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

class CriticTrainer:
    def __init__(self, args, device: torch.device, type_to_id: Dict[str, int], faction_to_id: Dict[str, int],
                 train: List[Item], holdout: List[Item], provenance: Dict):
        self.args, self.device = args, device
        self.train, self.holdout = train, holdout
        torch.manual_seed(args.seed)
        arch = cr.SMALL_ARCH if args.arch == "small" else cr.FULL_ARCH
        self.encoder, self.model = cr.build_critic(arch, type_to_id, faction_to_id, device)
        self.init = None if args.init is None else cr.load_reference(self.encoder, self.model, args.init)
        if self.init is not None:
            log.info("loaded %s: missing %s, unexpected %s", args.init, self.init["missing"],
                     self.init["unexpected"])
        named = cr.trained_parameters(self.model, self.encoder)
        self.params = [p for _, p in named]
        self.groups = GradientGroups(named)
        self.opt = torch.optim.AdamW(self.params, lr=args.lr, weight_decay=args.weight_decay)
        self.probe = GradientProbe(named, signal_group, SIGNAL_GROUPS, self.opt)
        self.step_norms = StepNorms(args.grad_clip)
        self.autocast = torch.bfloat16 if device.type == "cuda" and not args.fp32 else None
        self.arch = arch
        self.epoch_steps = max(1, math.ceil(len(train) / args.batch))
        self.total_steps = self.epoch_steps * args.max_epochs
        self.eval_every = max(args.batch, min(args.max_eval_every, len(train) // max(1, args.evals_per_epoch)))
        self.state = {"steps": 0, "positions": 0, "epoch": 0, "best": math.inf, "best_positions": 0,
                      "reads": 0, "last_read": -1, "since_best": 0, "nonfinite_steps": 0, "next_signal": args.signal_every}
        self.provenance = provenance
        self.stem = args.out.with_suffix("")
        self.t0 = time.time()

    # ---- one step ---------------------------------------------------
    def lr_now(self) -> float:
        s = self.state["steps"]
        warm = min(1.0, (s + 1) / max(1, self.args.warmup_steps))
        return self.args.lr * warm * max(0.0, 1.0 - s / max(1, self.total_steps))

    def losses(self, raws, z, aux) -> Dict[str, torch.Tensor]:
        logits, aux_pred = cr.critic_forward(self.encoder, self.model, raws, self.device, self.autocast)
        return cr.critic_losses(self.model, logits, aux_pred, z.to(self.device), aux.to(self.device))

    def step(self, batch) -> Dict[str, float]:
        self.model.train()
        self.encoder.train()
        raws, z, aux = batch
        parts = self.losses(raws, z, aux)
        loss = parts["value"] + self.args.aux_weight * parts["aux"]
        lr = self.lr_now()
        for g in self.opt.param_groups:
            g["lr"] = lr
        loss.backward()
        norm, grads = self.groups.clip(self.args.grad_clip)
        self.step_norms.append(norm)
        finite = bool(torch.isfinite(norm)) and bool(torch.isfinite(loss.detach()))
        if finite:
            self.opt.step()
        else:
            self.state["nonfinite_steps"] += 1
            log.warning("non-finite step %d: update skipped", self.state["steps"])
        self.opt.zero_grad(set_to_none=True)
        self.state["steps"] += 1
        self.state["positions"] += len(raws)
        row = {"step": self.state["steps"], "positions": self.state["positions"], "lr": lr,
               "value": float(parts["value"].detach()), "aux": float(parts["aux"].detach()),
               "grad_norm": float(norm), "applied": finite, "grad": grads}
        append_jsonl(Path(f"{self.stem}.steps.jsonl"), row)
        return row

    # ---- telemetry --------------------------------------------------
    def signal_row(self, batch) -> None:
        raws, z, aux = batch
        rng = random.Random(self.args.seed * 1_000_003 + self.state["positions"])
        pick = sorted(rng.sample(range(len(raws)), min(len(raws), self.args.probe_positions)))
        row = {"kind": "probe", "step": self.state["steps"], "positions": self.state["positions"],
               "ts": time.strftime("%FT%T"), "steps": self.step_norms.drain()}
        started = time.perf_counter()
        try:
            self.model.train()
            self.encoder.train()
            with self.probe.fork_rng():
                parts = self.losses([raws[i] for i in pick], z[pick], aux[pick])
                terms = {"value": parts["value"], "aux": self.args.aux_weight * parts["aux"]}
                gradient, update, stateless = self.probe.grams(terms, 1.0)
            row.update(probe_positions=len(pick),
                       **probe_readings(gradient, update, stateless, terms=SIGNAL_TERMS, policy_terms=()))
        except Exception as e:  # noqa: BLE001 - telemetry never stops training
            row["probe_error"] = repr(e)[:300]
            log.warning("signal probe failed at step %d: %r", self.state["steps"], e)
        row["probe_ms"] = round(1000 * (time.perf_counter() - started), 1)
        write_signal_row(Path(f"{self.stem}.signal.jsonl"), row)

    # ---- holdout ----------------------------------------------------
    @torch.no_grad()
    def holdout_loss(self) -> Dict[str, float]:
        self.model.eval()
        self.encoder.eval()
        sums = {"value": 0.0, "aux": 0.0, "n": 0, "aux_n": 0}
        order = np.arange(len(self.holdout))
        for raws, z, aux in Feeder(self.holdout, order, self.args.batch):
            parts = self.losses(raws, z, aux)
            n_aux = int(torch.isfinite(aux).sum())
            sums["value"] += float(parts["value"]) * len(raws)
            sums["aux"] += float(parts["aux"]) * n_aux
            sums["n"] += len(raws)
            sums["aux_n"] += n_aux
        return {"value": sums["value"] / max(1, sums["n"]), "aux": sums["aux"] / max(1, sums["aux_n"]),
                "positions": sums["n"]}

    def read_holdout(self) -> bool:
        """Read the holdout loss; save the checkpoint at a new lowest.
        True when the patience has run out."""
        h = self.holdout_loss()
        self.state["reads"] += 1
        self.state["last_read"] = self.state["positions"]
        better = h["value"] < self.state["best"]
        if better:
            self.state.update(best=h["value"], best_positions=self.state["positions"], since_best=0)
            self.save()
        else:
            self.state["since_best"] += 1
        append_jsonl(Path(f"{self.stem}.holdout.jsonl"),
                     {"positions": self.state["positions"], "step": self.state["steps"],
                      "epoch": self.state["epoch"], "holdout_value": h["value"], "holdout_aux": h["aux"],
                      "holdout_positions": h["positions"], "best": better, "minutes": self.minutes()})
        log.info("positions %d epoch %d: holdout value %.4f aux %.4f%s", self.state["positions"],
                 self.state["epoch"], h["value"], h["aux"], " (best)" if better else "")
        self.write_summary(None)
        return self.state["since_best"] >= self.args.patience

    def minutes(self) -> float:
        return (time.time() - self.t0) / 60.0

    def save(self) -> None:
        from wesnoth_ai.constants import OBSERVATION_EPOCH
        payload = {"arch": dict(self.arch), "observation_epoch": int(OBSERVATION_EPOCH),
                   "observation_parity": True, "memory_slots": 0, "relevant_set_version": 2,
                   "relevant_set_hexes": True, "fog_hides_enemy_villages": True, "terrain_multi_hot": True,
                   "aux_score": True, "model_state": self.model.state_dict(),
                   "encoder_state": self.encoder.state_dict(),
                   "unit_type_to_id": dict(self.encoder.unit_type_to_id),
                   "faction_to_id": dict(self.encoder.faction_to_id),
                   "critic": {"view": self.args.view, "source": self.args.source, "fraction": self.args.fraction,
                              "arch_name": self.args.arch, "aux_scale": cr.AUX_SCALE,
                              "aux_weight": self.args.aux_weight},
                   "training": dict(self.state), "provenance": self.provenance}
        self.args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.args.out.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, self.args.out)

    def write_summary(self, ended_by: Optional[str]) -> None:
        summary = {"state": dict(self.state), "ended_by": ended_by, "minutes": self.minutes(),
                   "train_positions": len(self.train), "holdout_positions": len(self.holdout),
                   "epoch_steps": self.epoch_steps, "eval_every": self.eval_every,
                   "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(self.args).items()},
                   "init": self.init, "provenance": self.provenance}
        if ended_by is not None:
            summary["DONE"] = True
        tmp = Path(f"{self.stem}.summary.json.tmp")
        tmp.write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, Path(f"{self.stem}.summary.json"))

    # ---- the loop ---------------------------------------------------
    def run(self) -> str:
        write_signal_row(Path(f"{self.stem}.signal.jsonl"),
                         {"kind": "start", "positions": 0, "terms": list(SIGNAL_TERMS),
                          "groups": list(SIGNAL_GROUPS), "every": self.args.signal_every,
                          "probe_positions": self.args.probe_positions, "clip": self.args.grad_clip})
        log.info("%d training positions (%d steps an epoch), %d holdout positions; holdout read every %d",
                 len(self.train), self.epoch_steps, len(self.holdout), self.eval_every)
        next_read = self.eval_every
        for epoch in range(self.args.max_epochs):
            self.state["epoch"] = epoch
            order = np.random.default_rng(self.args.seed + epoch).permutation(len(self.train))
            for batch in Feeder(self.train, order, self.args.batch):
                row = self.step(batch)
                if self.state["positions"] >= self.state["next_signal"]:
                    self.signal_row(batch)
                    self.state["next_signal"] += self.args.signal_every
                if self.state["steps"] % 50 == 0:
                    log.info("step %d positions %d lr %.2e value %.4f aux %.4f grad %.3f %.1f min",
                             row["step"], row["positions"], row["lr"], row["value"], row["aux"],
                             row["grad_norm"], self.minutes())
                if self.state["positions"] >= next_read:
                    next_read += self.eval_every
                    if self.read_holdout():
                        return self.finish("patience")
                    if self.minutes() >= self.args.max_minutes:
                        return self.finish("max_minutes")
            if self.state["positions"] != self.state["last_read"] and self.read_holdout():
                return self.finish("patience")
            if self.minutes() >= self.args.max_minutes:
                return self.finish("max_minutes")
        return self.finish("max_epochs")

    def finish(self, ended_by: str) -> str:
        self.write_summary(ended_by)
        log.info("CRITIC_TRAIN_DONE %s: ended by %s after %d positions, best holdout value %.4f at %d",
                 self.args.out, ended_by, self.state["positions"], self.state["best"],
                 self.state["best_positions"])
        return ended_by


def append_jsonl(path: Path, row: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=float) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--positions", type=Path, required=True)
    ap.add_argument("--source", choices=("M", "H"), required=True)
    ap.add_argument("--view", choices=("true", "obs"), required=True)
    ap.add_argument("--fraction", type=float, default=1.0, help="share of the source's training games")
    ap.add_argument("--arch", choices=("full", "small"), default="full")
    ap.add_argument("--init", type=Path, default=None, help="the reference checkpoint a full critic starts from")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp32", action="store_true", help="no bfloat16 autocast on CUDA")
    for key, value in DEFAULTS.items():
        ap.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if args.arch == "full" and args.init is None:
        raise SystemExit("a full critic starts from the reference: pass --init")
    if args.arch == "small" and args.init is not None:
        raise SystemExit("the small critic starts from scratch")
    meta = json.loads((args.positions / "summary.json").read_text(encoding="utf-8"))
    type_to_id, faction_to_id = meta["unit_type_to_id"], meta["faction_to_id"]
    rows = manifest_rows(args.positions)
    train_rows, hold_rows = select_games(rows, args.source, args.fraction)
    if not train_rows or not hold_rows:
        raise SystemExit(f"no training ({len(train_rows)}) or holdout ({len(hold_rows)}) games for "
                         f"source {args.source} at fraction {args.fraction}")
    t0 = time.time()
    train, holdout = load_items(args.positions, train_rows, args.view), load_items(args.positions, hold_rows, args.view)
    log.info("loaded %d + %d positions of %d + %d games in %.0f s", len(train), len(holdout), len(train_rows),
             len(hold_rows), time.time() - t0)
    from wesnoth_ai import __version__
    provenance = {"code_version": __version__, "positions_dir": str(args.positions),
                  "positions_fingerprint": meta.get("fingerprint"), "train_games": len(train_rows),
                  "holdout_games": len(hold_rows), "seed": args.seed}
    trainer = CriticTrainer(args, torch.device(args.device), type_to_id, faction_to_id, train, holdout, provenance)
    trainer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
