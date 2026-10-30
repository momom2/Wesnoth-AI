"""The evaluator's readings kept on disk (tools/pg_eval.py, `EvalConfig.cache`),
so a restarted stage C driver resumes instead of forwarding again: each
reading the evaluator would forward is first looked up here, and each one
it forwards is written here.

A reading is keyed by everything it depends on:
- the context: the checkpoint's SHA-256, the held-out game-sides in their
  order (game, side, start, stop, half), the KL subsample, the `EvalConfig`
  fields that change a forward (not `workers`, `prefetch`, `verify` or
  `cache`), and CACHE_VERSION;
- the direction: the SHA-256 of its parameters' names, shapes and float32
  bytes;
- the step s, exactly (`float.hex`), and for a half reading its halves.

So a reading comes back only for the same network, data, configuration and
step; anything else is a miss, and the evaluator forwards as before. Files
are written to a temporary name and renamed, so a killed process leaves no
partial reading. The stage C driver's directions are built on the CPU from
the pass's sums, so they are bit-identical from one start to the next.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

CACHE_VERSION = 1                       # bump when a reading's meaning changes
_UNREAD_FIELDS = ("workers", "prefetch", "verify", "cache")      # EvalConfig fields no forward depends on


def context_key(checkpoint_sha256: str, sides: Sequence, kl_sides: Sequence, config) -> str:
    """The key of an evaluation context (module docstring)."""
    fields = {k: v for k, v in asdict(config).items() if k not in _UNREAD_FIELDS}
    spec = {"version": CACHE_VERSION, "checkpoint": checkpoint_sha256, "config": fields,
            "sides": [[gs.game, gs.side, gs.start, gs.stop, gs.half] for gs in sides],
            "kl_sides": [[gs.game, gs.side] for gs in kl_sides]}
    return hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:24]


def direction_key(direction: Mapping[str, torch.Tensor]) -> str:
    """The SHA-256 of a direction's names, shapes and float32 values."""
    h = hashlib.sha256()
    for name in sorted(direction):
        t = direction[name].detach().to(device="cpu", dtype=torch.float32).contiguous()
        h.update(name.encode())
        h.update(repr(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()[:24]


def step_key(s: float) -> str:
    return float(s).hex()


class ReadingCache:
    """One context's readings under root/<context key>/."""

    def __init__(self, root: Path, key: str):
        self.dir = Path(root) / key
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0

    def _replace(self, name: str, write) -> None:
        path = self.dir / name
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            write(f)
        os.replace(tmp, path)

    def arrays(self, name: str) -> Optional[Dict[str, np.ndarray]]:
        path = self.dir / f"{name}.npz"
        if not path.exists():
            return None
        with np.load(path) as z:
            out = {k: z[k] for k in z.files}
        self.hits += 1
        return out

    def put_arrays(self, name: str, **arrays: np.ndarray) -> None:
        self._replace(f"{name}.npz", lambda f: np.savez(f, **arrays))

    def value(self, name: str) -> Optional[float]:
        path = self.dir / f"{name}.json"
        if not path.exists():
            return None
        self.hits += 1
        return float(json.loads(path.read_text(encoding="utf-8"))["value"])

    def put_value(self, name: str, value: float) -> None:
        self._replace(f"{name}.json", lambda f: f.write(json.dumps({"value": float(value)}).encode()))

    def laws(self) -> Optional[List[Dict[str, torch.Tensor]]]:
        path = self.dir / "laws.pt"
        if not path.exists():
            return None
        self.hits += 1
        return torch.load(path, map_location="cpu", weights_only=True)

    def put_laws(self, laws: List[Dict[str, torch.Tensor]]) -> None:
        self._replace("laws.pt", lambda f: torch.save(laws, f))
