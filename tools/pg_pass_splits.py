"""The gradient pass's split files (tools/pg_pass.py `GradientPass`, its
`partial_dir`): each split's sums written as the split completes, and taken
back by a pass started again when the file is this pass's. A split's key
hashes what its sums depend on (the checkpoint, the settings that shape
them, the split's game-sides, labels and lengths). Under fp16 a split file
carries its loss-scale record, so a resumed pass carries the scale on from
the last split it takes and records the uncut pass's counts.

`SplitFiles` is GradientPass's base: it reads and writes the pass's own
sums (`splits`, `sq`, `split_windows`, `head_sums`, `head_fisher`,
`replay`, `rows`, `shadow`, the probes, the floor and the loss scale),
which GradientPass keeps.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict
from typing import Dict, List, Optional

import torch

from wesnoth_ai.pg_loss import CLASSES
from wesnoth_ai.sequence_streams import GameSide as StreamSide

log = logging.getLogger("pg_pass")


class SplitFiles:
    """A split's file: its key, its sums written, and taken back on resume
    (module docstring)."""

    def _split_games(self, i: int) -> List[str]:
        return sorted(g for g, sides in self.games.items() if sides[1].split == i)

    def _split_key(self, i: int) -> str:
        """What a split's written sums depend on: the checkpoint, the
        settings that shape the sum, the split's game-sides, labels and
        lengths."""
        config = {k: v for k, v in asdict(self.config).items()
                  if k not in ("workers", "prefetch", "verify", "floor_chunk")}
        sides = [[g, s, self.R[(g, s)], self.lengths[StreamSide(g, s)]] for g in self._split_games(i) for s in (1, 2)]
        body = json.dumps({"checkpoint": self.checkpoint_sha256, "config": config, "sides": sides}, sort_keys=True)
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    def _tally(self) -> Dict:
        probes = torch.zeros(len(CLASSES), device=self.device) if self.probes.grad is None else self.probes.grad
        ls = self.loss_scale
        return {"head_sums": self.head_sums.clone(), "head_fisher": self.head_fisher.clone(),
                "replay": self.replay.clone(), "rows": self.rows, "shadow": self.shadow, "probes": probes.clone(),
                "scale_counts": None if ls is None else (ls.overflows, ls.growths)}

    def _split_scale(self, before: Dict) -> Optional[Dict]:
        """A split's loss-scale record: its overflows and growths, the
        smallest scale its windows ran at, and the scale and finite run it
        ends on (the next split's start); None without a loss scale."""
        ls = self.loss_scale
        if ls is None:
            return None
        overflows, growths = before["scale_counts"]
        return {"overflows": ls.overflows - overflows, "growths": ls.growths - growths,
                "smallest": self._split_smallest, "scale": ls.scale, "finite_run": ls.finite_run}

    def _write_split(self, i: int, before: Dict) -> None:
        if self.partial_dir is None:
            return
        now = self._tally()
        sides = [(g, s) for g in self._split_games(i) for s in (1, 2)]
        floor = [[g, s, *v] for (g, s), v in (self.floor.of(sides) if self.floor else {}).items()]
        replay = now["replay"] - before["replay"]
        replay[1] = self._split_max
        part = {"key": self._split_key(i), "split": i, "windows": self.split_windows[i],
                "sums": {n: t.cpu() for n, t in self.splits[i].items()}, "sq": {n: t.cpu() for n, t in self.sq[i].items()},
                "head_sums": (now["head_sums"] - before["head_sums"]).cpu(),
                "head_fisher": (now["head_fisher"] - before["head_fisher"]).cpu(), "replay": replay.cpu(),
                "rows": now["rows"] - before["rows"], "shadow": now["shadow"] - before["shadow"],
                "probes": (now["probes"] - before["probes"]).cpu(), "floor": floor,
                "loss_scale": self._split_scale(before)}
        self.partial_dir.mkdir(parents=True, exist_ok=True)
        path = self.partial_dir / f"split{i}.pt"
        tmp = path.with_name(path.name + ".tmp")
        torch.save(part, tmp)
        os.replace(tmp, path)

    def _load_split(self, i: int) -> bool:
        """Take split i's written sums when they are this pass's, and with a
        loss scale its record: the counts added, the scale and its finite
        run carried on from the split's end."""
        path = None if self.partial_dir is None else self.partial_dir / f"split{i}.pt"
        if path is None or not path.is_file():
            return False
        part = torch.load(path, map_location="cpu", weights_only=True)
        if part.get("key") != self._split_key(i):
            log.warning("%s is another pass's split: it is read again", path)
            return False
        scale = part.get("loss_scale")
        if self.loss_scale is not None and scale is None:
            log.warning("%s was written without its loss scale: it is read again", path)
            return False
        if self.loss_scale is not None:
            ls = self.loss_scale
            ls.overflows += int(scale["overflows"])
            ls.growths += int(scale["growths"])
            ls.smallest = min(ls.smallest, float(scale["smallest"]))
            ls.scale, ls.finite_run = float(scale["scale"]), int(scale["finite_run"])
        dev = self.device
        self.splits[i] = {n: t.to(dev) for n, t in part["sums"].items()}
        self.sq[i] = {n: t.to(dev) for n, t in part["sq"].items()}
        self.split_windows[i] = int(part["windows"])
        self.head_sums += part["head_sums"].to(dev)
        self.head_fisher += part["head_fisher"].to(dev)
        replay = part["replay"].to(dev)
        self.replay[[0, 2, 3]] += replay[[0, 2, 3]]
        self.replay[1] = torch.maximum(self.replay[1], replay[1])
        self.rows += int(part["rows"])
        self.shadow += int(part["shadow"])
        self.probe_carry += part["probes"].to(dev)
        if self.floor is not None:
            self.floor.merge({(g, int(s)): v for g, s, *v in part["floor"]})
        log.info("split %d taken from %s", i, path)
        return True
