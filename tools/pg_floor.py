"""The replay identity's precision floor (tools/pg_pass.py; docs/
training_design_20261019/current_design.md §5, C0).

The generator serves at a reduced precision on the packed varlen trunk
(tools/eval_inference_server.py: the embedding in fp32, the trunk under an
autocast, each batch's numerics depending on its composition): bf16 on a
4090 box, fp16 on the local RX 6600 (--infer-fp16, "The serving precision"
below). The pass recomputes the law in fp32 on the padded trunk. Two
differences sit between the logged law and the trainer's, and the floor
separates them on the decisions of a few game-sides, each forwarded three
times in a batch of the floor's own:

    rounding   KL(packed trunk at the serving precision || packed trunk in
               fp32): the serving precision's rounding alone, on the trunk
               the generator serves;
    path       KL(packed trunk in fp32 || padded trunk in fp32): the two
               trunks compute one function, so this must sit at fp32's
               rounding, below PATH_KL_LIMIT; past it the pass fails (the
               departure the replay identity exists to catch, which the
               floor must not absorb).

The replay limit is then, as `replay_limit` codes it,

    limit = max(base, base + 2 (mean + 2 SE)),   base 1e-4 (pg_pass.REPLAY_KL_LIMIT),

mean the rounding's per-decision mean over the drawn decisions and SE its
SE clustered by game (both seats of a game can be drawn, and they share the
game's positions). Above FLOOR_LIMIT_CAP the pass refuses: a serving
precision that far from fp32 cannot certify the replay at the step search's
scale, and stage 1 should serve in fp32.

The game-sides are drawn by a hash of their names, FLOOR_SIDES of them (all
of a smaller pass): 400 of a 20,000-game pass's 40,000, 1 % of its
decisions. Over about 300 to 400 games the SE is about a twentieth of the
spread of a game's mean rounding, so the bound overstates the floor by
about a tenth of that spread; the three forwards on 1 % of the decisions
cost about 3 % of the pass's forwards, 1 % of its time.

The serving precision. Each match result records it
(pg_positions.serving_precision): bfloat16 on a 4090 box, float16 on the
local RX 6600 (scripts/selfplay_t1_local.py: --infer-fp16, the packed trunk
on the HIP kernels of wesnoth_ai/hip_kernels.py and the fused add-and-norm
loop of WesnothModel._run_hip_fp16_trunk). The floor is read at the
precision the labelled games were served in (`resolve_floor_dtype`; the
pass's command line takes it from the records, "auto"), and the limit is
the formula above with the mean read at that precision: nothing in it
depends on which precision that is. Games served in float32 carry no
serving rounding: auto reads their path alone (PATH_ONLY), so the base
limit holds and a packed trunk that departs from the padded one still
fails the pass. One pass takes the games of one serving precision. Games
served on another numerics path beside the autocast (the bf16 residual
stream, the fused ReLU, fp8 products: pg_positions.OTHER_NUMERICS) carry a
rounding the floor's autocast does not read, and are refused.

- **float16 against bfloat16.** A rounding moves a value by up to its
  format's unit roundoff relative to it: 2^-8 for bfloat16's 8 significant
  bits, 2^-11 for float16's 11, an eighth. A small perturbation d of a
  law's logits moves it by KL = Var_p(d) / 2 + O(d^3), quadratic in d, so
  float16's floor is about 1/64 of bfloat16's on the same positions, by
  the formats alone, as long as fp16's narrower range (normals from 6e-5
  to 65504) holds the trunk's values. Measured on the CPU's autocast
  (2026-10-29, the local stage 1's first game, 507 decisions,
  docs/stage1_local_handoff_20261029.md): 3.77e-6 at float16, 2.44e-4 at
  bfloat16, a ratio of 65. A bfloat16 floor on fp16-served games lifts
  the limit by about 64 times the rounding fp16 explains: too loose. A
  float16 floor on bf16-served games sits below the logs' rounding: too
  strict. `resolve_floor_dtype` refuses both.
- **What the floor does not read.** It reads one forward's rounding on
  the trainer's own memory, by the autocast on the pass's device: on a
  CUDA box the flash varlen kernel, not the HIP kernels and the fused
  add-and-norm that served the local logs, and not the memory the served
  law carried along the game, which its own reduced-precision forwards
  wrote. The limit's factor 2 over the bound is the margin for those. On
  the local logs (the first five games, 3,021 decisions, against an fp32
  recompute) the replay's second-order reading on the first game is
  5.5e-6, about 1.5 times the CPU's float16 floor, and |logged − trainer|
  is as large in a game-side's first ten decisions (0.0015) as after its
  hundredth (0.0017 to 0.0019): no drift through the memory. Their
  first-order KL estimate, 5.0e-5 +- 5.3e-5, is noise: one game's alone
  has an SE of about 1.3e-4, so a pass over a game or two can miss the
  limit by chance; C0's 200 games read it to about 1e-5. C0 records the
  two side by side (c0_pass.json: `replay.kl`, `replay.half_mean_square`,
  `floor.kl`).
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
# The precisions a match server serves at (tools/eval_inference_server.py),
# which the floor reads; FLOOR_DTYPES adds the pass's two other choices.
SERVING_DTYPES = ("bfloat16", "float16")
FLOOR_DTYPES = SERVING_DTYPES + ("auto", "off")
# What auto reads on games served in float32: no serving rounding (they
# carry none, so nothing lifts the base limit), the path alone (the packed
# and padded trunks' fp32 agreement, which no serving precision decides).
PATH_ONLY = "float32"


class FloorDtypeError(ValueError):
    """The floor's precision is not the one the labelled games were served in."""


def resolve_floor_dtype(requested: str, servings: Iterable[Optional[str]]) -> str:
    """The precision the floor reads ("off": none), given the serving
    precision of each labelled game (pg_positions.serving_precision; None,
    a game without a match result, is left to refuse_foreign):

        auto       the games' precision; PATH_ONLY ("float32") when they
                   were served in float32: no rounding read, so the base
                   limit holds, and the path still read and checked;
                   refused when the games were served in more than one
                   precision, float32 among them (one pass, one floor: a
                   reduced precision's floor would lift the limit for the
                   decisions of fp32-served games, which carry no rounding);
        bfloat16, float16
                   that precision; refused when a game was served in the
                   other one (module docstring: too loose or too strict).
                   Games served in float32 take either, alone or beside
                   games of the named precision, as the suite's CPU games
                   do: the floor then reads a rounding they do not carry,
                   the caller's choice;
        off        no floor and no path check.

    Games served on another numerics path beside the autocast
    (pg_positions.OTHER_NUMERICS: "bfloat16+infer_bf16_residual", say) are
    refused under every choice but off: the floor's autocast reads the
    plain trunk's rounding, not theirs.
    """
    if requested not in FLOOR_DTYPES:
        raise FloorDtypeError(f"--floor-dtype {requested!r} is none of {FLOOR_DTYPES}")
    if requested == "off":
        return "off"
    served = sorted({s for s in servings if s is not None})
    other = [s for s in served if s not in SERVING_DTYPES and s != PATH_ONLY]
    if other:
        raise FloorDtypeError(f"the labelled games were served on numerics the floor does not read "
                              f"({', '.join(other)}): its autocast reads the plain trunk's rounding, not "
                              f"theirs; pass games served without them, or --floor-dtype off (no floor, no "
                              f"path check)")
    if requested == "auto":
        if len(served) > 1:
            raise FloorDtypeError(f"the labelled games were served in {' and '.join(served)}: one pass reads one "
                                  f"precision floor; pass the games of one serving precision")
        return served[0] if served else PATH_ONLY          # "float32" is PATH_ONLY itself
    other = [s for s in served if s in SERVING_DTYPES and s != requested]
    if other:
        raise FloorDtypeError(f"the labelled games were served in {other[0]}, and a {requested} floor reads another "
                              f"rounding than theirs (tools/pg_floor.py): --floor-dtype {other[0]}, or auto")
    return requested


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
    device whose bf16 is not the server's, or an injected difference. At
    PATH_ONLY (games served in float32) no serving forward runs and the
    rounding is not read (`serving` is unused): the path alone. `chunk`
    rows at a time (None: a step's drawn rows at once) bounds each forward:
    on the RX 6600 the padded fp32 forward takes about 3 s at 64 rows, the
    kind of call that reset its display on 2026-10-28."""

    def __init__(self, encoder, model, device: torch.device, sides: set, dtype: str,
                 serving: Optional[Forward] = None, packed: Optional[Forward] = None,
                 chunk: Optional[int] = None):
        self.encoder, self.model, self.device, self.sides = encoder, model, device, sides
        self.dtype = dtype
        self.chunk = chunk
        self.reads_rounding = dtype != PATH_ONLY
        if self.reads_rounding:
            precision = getattr(torch, dtype)
            self.serving = serving or (lambda e, m, st, mem: packed_forward(e, m, st, mem, precision, device))
        else:
            self.serving = None
        self.packed = packed or (lambda e, m, st, mem: packed_forward(e, m, st, mem, None, device))
        # game-side -> [rounding sum, path sum, decisions, largest path KL]
        self.sums: Dict[Tuple[str, int], list] = {}

    def reads(self, steps) -> bool:
        """Whether `read` forwards anything at this step: a row of a drawn
        game-side."""
        return any((s.game_side.file, s.game_side.side) in self.sides for s in steps)

    def read(self, steps, positions, mems: MemoryBatch, stage_step) -> None:
        """The step's rows of drawn game-sides."""
        rows = [i for i, s in enumerate(steps) if (s.game_side.file, s.game_side.side) in self.sides]
        if not rows:
            return
        size = self.chunk or len(rows)
        for start in range(0, len(rows), size):
            self._read_rows(steps, positions, mems, stage_step, rows[start:start + size])

    def _read_rows(self, steps, positions, mems: MemoryBatch, stage_step, rows) -> None:
        sub = [positions[i] for i in rows]
        with torch.no_grad():
            memory = MemoryBatch(padded=mems.padded[rows].detach(), counts=[mems.counts[i] for i in rows])
            staged = self.encoder.stage_raws([p.raw for p in sub], device=self.device)
            batch = stage_step(sub, staged, self.model, self.device)
            padded = padded_forward(self.encoder, self.model, staged, memory)
            packed = self.packed(self.encoder, self.model, staged, memory)
            if self.serving is None:
                rounding = [0.0] * len(rows)                 # PATH_ONLY: unread (stats reports None)
            else:
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
        """The floor's mean over its decisions with its SE clustered by game
        (a ratio estimator over the games' sums), its upper bound, and the
        path reading; the three rounding fields None at PATH_ONLY (not
        read); None when no game-side was read."""
        if not self.sums:
            return None
        by_game: Dict[str, list] = {}
        for (game, _side), v in self.sums.items():
            entry = by_game.setdefault(game, [0.0, 0])
            entry[0] += v[0]
            entry[1] += v[2]
        rounding = torch.tensor([v[0] for v in by_game.values()], dtype=torch.float64)
        counts = torch.tensor([v[1] for v in by_game.values()], dtype=torch.float64)
        m = len(by_game)
        values = list(self.sums.values())
        path = {"path_kl": sum(v[1] for v in values) / float(counts.sum()), "path_max": max(v[3] for v in values),
                "decisions": int(counts.sum()), "game_sides": len(values), "games": m, "dtype": self.dtype}
        if not self.reads_rounding:
            return {"kl": None, "se": None, "upper": None, **path}
        mean = float(rounding.sum() / counts.sum())
        se = (float(((rounding - mean * counts) ** 2).sum() * m / (m - 1)) ** 0.5 / float(counts.sum())
              if m > 1 else float("nan"))
        return {"kl": mean, "se": se, "upper": mean + FLOOR_UPPER_SE * se if math.isfinite(se) else float("nan"),
                **path}


def replay_limit(base: float, floor: Optional[Dict]) -> float:
    """The replay identity's limit over a measured floor: max(base, base +
    2 (mean + 2 SE)) (module docstring), the floor's mean alone when its SE
    is not read (one game); the base without a floor or with the path alone
    (PATH_ONLY: no rounding read)."""
    if floor is None or floor["kl"] is None:
        return base
    upper = floor["upper"] if math.isfinite(floor["upper"]) else floor["kl"]
    return max(base, base + 2.0 * upper)


def refusal(floor: Optional[Dict], limit: float) -> Optional[str]:
    """Why the pass refuses its floor, or None: the trunks disagree, or the
    limit passes FLOOR_LIMIT_CAP (a limit the floor lifted: not one read on
    the path alone, whose limit is the caller's base)."""
    if floor is None:
        return None
    if not floor["path_kl"] <= PATH_KL_LIMIT:
        return (f"path: the packed and padded trunks disagree in fp32 by {floor['path_kl']:.3g} nats a decision "
                f"(largest {floor['path_max']:.3g}), above {PATH_KL_LIMIT:g}")
    if floor["kl"] is None:
        return None
    if not limit <= FLOOR_LIMIT_CAP:
        return (f"floor: its upper bound {floor['upper']:.3g} lifts the replay limit to {limit:.3g}, above "
                f"{FLOOR_LIMIT_CAP:g}; stage 1 should serve in fp32")
    return None
