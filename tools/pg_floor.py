"""The replay identity's precision floor (tools/pg_pass.py; docs/
training_design_20261019/current_design.md §5, C0).

The box generator serves in bf16 on the packed varlen trunk
(tools/eval_inference_server.py: the embedding in fp32, the trunk under bf16
autocast, each batch's numerics depending on its composition), while the
pass recomputes the law in fp32 on the padded trunk. Two differences sit
between the logged law and the trainer's, and the floor separates them on
the decisions of a few game-sides, each forwarded three times in a batch of
the floor's own:

    rounding   KL(packed trunk in bf16 || packed trunk in fp32): the serving
               precision's rounding alone, on the trunk the generator serves;
    path       KL(packed trunk in fp32 || padded trunk in fp32): the two
               trunks compute one function, so this must sit at fp32's
               rounding, below PATH_KL_LIMIT; past it the pass fails (the
               departure the replay identity exists to catch, which the
               floor must not absorb).

The replay limit then rises by twice the floor's upper bound, its mean plus
FLOOR_UPPER_SE SEs clustered by game-side: max(base, base + 2 upper). Above
FLOOR_LIMIT_CAP the pass refuses: a serving precision that far from fp32
cannot certify the replay at the step search's scale, and stage 1 should
serve in fp32.

The game-sides are drawn by a hash of their names, FLOOR_SIDES of them (all
of a smaller pass): with 400 the SE is a twentieth of the spread of a
game-side's mean rounding, so the upper bound overstates the floor by a
tenth of that spread; the three forwards on 2 % of a 20,000-game pass's
decisions cost about 6 % of its forwards, 2 % of its time.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import torch

from wesnoth_ai.memory import MemoryBatch
from wesnoth_ai.pg_loss import legal_kl
from wesnoth_ai.sequence_streams import unit_draw

FLOOR_SIDES = 400
FLOOR_UPPER_SE = 2.0
# 1e-3 nats per decision: ten times the base limit and about the arms' radius
# (K 1/4 over 237.9 decisions a side-game, 0.00105 a decision; design §1.2):
# a limit past it would let through a replay miss the size of a step.
FLOOR_LIMIT_CAP = 1e-3
# fp32 rounding between two orders of the same sums moves a logit by about
# 1e-6 relative; a KL of 1e-6 nats a decision, 1 % of the base limit, is far
# above that and far below a departure the gate must see.
PATH_KL_LIMIT = 1e-6


def draw_sides(games: Iterable[str], seed: int, target: int = FLOOR_SIDES) -> set:
    """The game-sides of `games` the floor reads, drawn by a hash of their
    names with the probability that gives `target` of them."""
    games = list(games)
    fraction = min(1.0, target / max(1, 2 * len(games)))
    return {(g, s) for g in games for s in (1, 2) if unit_draw("pg_floor", seed, g, s) < fraction}


def packed_forward(encoder, model, staged, memory, dtype: Optional[torch.dtype], device: torch.device):
    """The packed trunk's law: the embedding in fp32, the trunk under
    autocast at `dtype` (fp32 when None), as the server forwards it."""
    streams = encoder.embed_staged(staged)
    with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
        return model.forward_embedded(streams, memory=memory, packed=True).float32()


def padded_forward(encoder, model, staged, memory):
    """The padded trunk's law in fp32: the pass's own forward."""
    return model.forward_embedded(encoder.embed_staged(staged), memory=memory, packed=False).float32()


Forward = Callable[..., object]


class FloorReader:
    """The rounding and path readings over the drawn game-sides' decisions,
    summed by game-side (module docstring). `serving` and `packed` replace
    the packed trunk's forward at the serving precision and in fp32: a
    device whose bf16 is not the server's, or an injected difference."""

    def __init__(self, encoder, model, device: torch.device, sides: set, dtype: str,
                 serving: Optional[Forward] = None, packed: Optional[Forward] = None):
        self.encoder, self.model, self.device, self.sides = encoder, model, device, sides
        self.dtype = dtype
        precision = getattr(torch, dtype)
        self.serving = serving or (lambda e, m, st, mem: packed_forward(e, m, st, mem, precision, device))
        self.packed = packed or (lambda e, m, st, mem: packed_forward(e, m, st, mem, None, device))
        # game-side -> [rounding sum, path sum, decisions, largest path KL]
        self.sums: Dict[Tuple[str, int], list] = {}

    def read(self, steps, positions, mems: MemoryBatch, stage_step) -> None:
        """The step's rows of drawn game-sides."""
        rows = [i for i, s in enumerate(steps) if (s.game_side.file, s.game_side.side) in self.sides]
        if not rows:
            return
        sub = [positions[i] for i in rows]
        with torch.no_grad():
            memory = MemoryBatch(padded=mems.padded[rows].detach(), counts=[mems.counts[i] for i in rows])
            staged = self.encoder.stage_raws([p.raw for p in sub], device=self.device)
            batch = stage_step(sub, staged, self.model, self.device)
            padded = padded_forward(self.encoder, self.model, staged, memory)
            packed = self.packed(self.encoder, self.model, staged, memory)
            served = self.serving(self.encoder, self.model, staged, memory)
            rounding = legal_kl(served, packed, batch.views, batch.rows).double().cpu().tolist()
            path = legal_kl(packed, padded, batch.views, batch.rows).double().cpu().tolist()
        for i, r, p in zip(rows, rounding, path):
            key = (steps[i].game_side.file, steps[i].game_side.side)
            entry = self.sums.setdefault(key, [0.0, 0.0, 0, 0.0])
            entry[0] += r
            entry[1] += p
            entry[2] += 1
            entry[3] = max(entry[3], p)

    def of(self, sides: Iterable[Tuple[str, int]]) -> Dict[Tuple[str, int], list]:
        """The sums of `sides` (a split's, to write it aside)."""
        return {k: list(v) for k, v in self.sums.items() if k in set(sides)}

    def merge(self, sums: Dict[Tuple[str, int], Sequence]) -> None:
        for k, v in sums.items():
            self.sums[tuple(k)] = list(v)

    def stats(self) -> Optional[Dict]:
        """The floor's mean over its decisions with its SE clustered by
        game-side (a ratio estimator), its upper bound, and the path
        reading; None when no game-side was read."""
        if not self.sums:
            return None
        values = list(self.sums.values())
        rounding = torch.tensor([v[0] for v in values], dtype=torch.float64)
        counts = torch.tensor([v[2] for v in values], dtype=torch.float64)
        mean = float(rounding.sum() / counts.sum())
        m = len(values)
        se = (float(((rounding - mean * counts) ** 2).sum() * m / (m - 1)) ** 0.5 / float(counts.sum())
              if m > 1 else float("nan"))
        return {"kl": mean, "se": se, "upper": mean + FLOOR_UPPER_SE * se if math.isfinite(se) else float("nan"),
                "path_kl": sum(v[1] for v in values) / float(counts.sum()), "path_max": max(v[3] for v in values),
                "decisions": int(counts.sum()), "game_sides": m, "dtype": self.dtype}


def replay_limit(base: float, floor: Optional[Dict]) -> float:
    """The replay identity's limit over a measured floor: max(base, base + 2
    upper), the floor's upper bound (module docstring); the base without one."""
    if floor is None:
        return base
    upper = floor["upper"] if math.isfinite(floor["upper"]) else floor["kl"]
    return max(base, base + 2.0 * upper)


def refusal(floor: Optional[Dict], limit: float) -> Optional[str]:
    """Why the pass refuses its floor, or None: the trunks disagree, or the
    limit passes FLOOR_LIMIT_CAP."""
    if floor is None:
        return None
    if not floor["path_kl"] <= PATH_KL_LIMIT:
        return (f"path: the packed and padded trunks disagree in fp32 by {floor['path_kl']:.3g} nats a decision "
                f"(largest {floor['path_max']:.3g}), above {PATH_KL_LIMIT:g}")
    if not limit <= FLOOR_LIMIT_CAP:
        return (f"floor: its upper bound {floor['upper']:.3g} lifts the replay limit to {limit:.3g}, above "
                f"{FLOOR_LIMIT_CAP:g}; stage 1 should serve in fp32")
    return None
