"""Material tiebreaker for drawn games (MCTS value signal).

Problem this solves: AlphaZero distills the terminal outcome z onto
every visited state. While the policy is too weak to ever kill a
leader, 100% of games end as turn-cap draws, every z is exactly 0,
and the value head receives no gradient at all -- which in turn
starves PUCT of meaningful Q estimates. Chicken-and-egg: the search
can't learn to fight because it never sees a nonzero outcome.

Fix (user-approved 2026-06-11): a draw is scored by MATERIAL
DIFFERENTIAL instead of a flat 0, mapped into a small symmetric
range (-cap, +cap) that stays well inside the win/loss values of
+/-1. Components (from the scoring side's perspective):

    villages   -- count of villages owned
    gold       -- side's current gold
    unit value -- sum of living units' recruit cost (gold-weighted
                  army strength; a Wose ~ 20g, a Gryphon Rider ~ 24g)

The same function scores BOTH the trainer's z target for drawn
games (mcts_policy.finalize_game) and the search's value at
turn-cap-terminal leaves (mcts._terminal_value), so what the
search optimizes at the horizon is exactly what the value head
learns.

Scale rationale (defaults, re-derived 2026-07-21): BANKED gold gets
weight 0 (2026-07-20): MP Wesnoth has no carryover, so end-of-game
gold is worthless in the real game, and pricing it at par with units
taught the prior 2.8x hoarding vs the SL baseline (BACKLOG
2026-07-20) -- only units score, converting gold into army is
strictly rewarded. Village and unit weights are calibrated to the
MEASURED distribution of drawn games (medians: |dV|/total ~= 0.5,
|d unit-gold| ~= 56): the previous weights put the median draw in
tanh saturation, so differentials barely moved z; the current
weights place it at score ~2 (tanh(0.4) ~= 38% of cap), the linear
regime. `cap=0.3` keeps the best possible draw far below a real
win: the search must always prefer a leaderkill (+1) over any
material lead.

All knobs live in `configs/draw_tiebreak.json` (or any JSON passed
via --draw-tiebreak-config); a modder can re-weight or disable
without touching model code.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wesnoth_ai.classes import GameState


@dataclass
class DrawTiebreakConfig:
    """Weights + bounds for the material draw score. See module
    docstring for the derivation of the defaults."""
    cap:               float = 0.3
    # Village term is Δvillages / MAP_TOTAL_VILLAGES (2026-07-12):
    # normalized so "half the map's villages" means the same signal
    # on every map. De-saturation (2026-07-21, user directive):
    # measured drawn-game medians under the previous weights
    # (village=10 -> 5.0 pts, units=0.05 -> ~2.8 pts, score_scale 5)
    # sat in tanh saturation -- differentials barely moved z.
    # village/5 and units/3 put the median drawn game at score ~2
    # (tanh(0.4) ~= 0.38 of cap, z ~= 0.11), the LINEAR regime, so
    # the tiebreak now GRADES draws instead of clipping them.
    # Derivation + measurements: docs/design_constants.md.
    weight_village:    float = 2.0
    # weight_gold = 0 since 2026-07-20: banked gold is WORTHLESS in
    # MP Wesnoth (no carryover), and pricing it equal to unit value
    # made the margin indifferent between hoarding and recruiting --
    # measured to teach the prior 2.8x hoarding vs the SL baseline
    # (BACKLOG 2026-07-20; A/B in bank_ab). Units score, gold
    # doesn't: conversion is now strictly rewarded. Mid-game gold's
    # instrumental value is the value head's job to learn.
    weight_gold:       float = 0.0
    weight_unit_value: float = 0.016667   # = 0.05/3, de-saturation
    score_scale:       float = 5.0

    @classmethod
    def from_json(cls, path: Path) -> "DrawTiebreakConfig":
        """Load overrides from a JSON file. Unknown keys (and keys
        starting with "_", reserved for comments) are ignored so
        config files can carry documentation."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        kwargs = {k: float(v) for k, v in data.items()
                  if not k.startswith("_") and k in cls.__dataclass_fields__}
        return cls(**kwargs)


def _map_total_villages(gs: "GameState") -> int:
    """Total village count of the map (terrain truth), cached on
    global_info as an underscore attr so MCTS deepcopies carry it
    and the hex scan runs once per game."""
    n = getattr(gs.global_info, "_map_total_villages", None)
    if n is None:
        from wesnoth_ai.classes import Terrain
        n = sum(1 for h in gs.map.hexes
                if Terrain.VILLAGE in h.terrain_types)
        try:
            setattr(gs.global_info, "_map_total_villages", n)
        except Exception:                            # noqa: BLE001
            pass
    return max(1, int(n))


def _side_material(gs: "GameState", side: int) -> tuple:
    """(villages, gold, unit_value) owned by `side`."""
    owner = getattr(gs.global_info, "_village_owner", None) or {}
    villages = sum(1 for s in owner.values() if s == side)
    # gs.sides is 0-indexed by (side - 1).
    gold = gs.sides[side - 1].current_gold if len(gs.sides) >= side else 0
    unit_value = sum(u.cost for u in gs.map.units if u.side == side)
    return villages, gold, unit_value


def draw_tiebreak_z(
    gs:   "GameState",
    side: int,
    cfg:  DrawTiebreakConfig,
) -> float:
    """Material score of a DRAWN final state from `side`'s
    perspective, in [-cfg.cap, +cfg.cap] (tanh saturates to the cap
    at float precision for runaway differentials). Antisymmetric
    between the two sides (z(side1) == -z(side2)) so the value
    targets stay zero-sum; smooth gradient near equality."""
    opponent = 3 - side   # 2p ladder games only
    v_us,  g_us,  u_us  = _side_material(gs, side)
    v_opp, g_opp, u_opp = _side_material(gs, opponent)
    score = (
        cfg.weight_village    * (v_us - v_opp) / _map_total_villages(gs)
        + cfg.weight_gold       * (g_us - g_opp)
        + cfg.weight_unit_value * (u_us - u_opp)
    )
    if cfg.score_scale <= 0:
        return 0.0
    return cfg.cap * math.tanh(score / cfg.score_scale)


def material_margin(
    gs:   "GameState",
    side: int,
    cfg:  DrawTiebreakConfig,
) -> float:
    """Auxiliary training target (KataGo 'score'/margin analog): the
    signed final MATERIAL margin from `side`'s perspective, squashed to
    (-1, +1) by tanh. Identical material score to `draw_tiebreak_z`
    (villages + weighted gold + weighted unit value) but WITHOUT the
    draw `cap` -- a denser signal than the win/loss z, distinguishing a
    crushing result from a narrow one regardless of who won. Computed
    for EVERY recorded state (not just draws). Antisymmetric between
    sides. `score_scale<=0` -> 0.0 (margin signal disabled)."""
    if cfg.score_scale <= 0:
        return 0.0
    opponent = 3 - side
    v_us,  g_us,  u_us  = _side_material(gs, side)
    v_opp, g_opp, u_opp = _side_material(gs, opponent)
    score = (
        cfg.weight_village    * (v_us - v_opp) / _map_total_villages(gs)
        + cfg.weight_gold       * (g_us - g_opp)
        + cfg.weight_unit_value * (u_us - u_opp)
    )
    return math.tanh(score / cfg.score_scale)
