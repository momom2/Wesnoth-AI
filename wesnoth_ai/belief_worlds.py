"""The worlds of the rollout re-grade in belief worlds
(docs/belief_worlds_prereg_20261012.md; the tool is tools/rollout_regrade.py
`--world`). Each is built for the mover from the simulator right after its
end_turn (the opponent's turn begun), on a fork drawing its dice from the
rollout's salt, with the steps of wesnoth_ai/lookahead_world.py.

| arm | the opponent's units the mover cannot see | the rest |
|---|---|---|
| `true` (T) | where they are | true |
| `true_units` (TU) | where they are | the estimates |
| `roster_belief` (RB) | placed by the belief head | the estimates |
| `roster_uniform` (RU) | placed uniformly | the estimates |
| `last_seen` (LS) | the sighted ones at their last sighting's hex, the others removed | the estimates |
| `none` (O2) | removed | the estimates |
| `observed` | removed | the re-grade's observed world (lookahead_world) |

"The estimates": villages owned as the mover's display shows them, and the
opponent's gold and base income set to the mover's at the start of its turn
(`turn_start`: read from the position's boundary, after the mover's
init_side paid its income, so every candidate of a position shares it).
The `observed` arm is the look-ahead player's observed world unchanged,
whose opponent takes the mover's gold after the candidate. In every arm but
T the units of the scenario's other sides that the mover cannot see are
removed, and the opponent's fog is recomputed from its units of the world
with its sighting record emptied.

Placement (RB, RU) draws from its own stream, `placement_rng(salt)`: the
opponent's hidden units in a random order, each on a hex that the mover does
not see, that holds no unit of the world (the units already placed
included, the hidden units not yet placed excluded) and whose movement cost
for that unit is below UNREACHABLE_COST. RB draws with probability
proportional to the belief head's probability over the qualifying hexes that
have a token (`belief_by_hex`); a unit with none of them takes RU's rule
and counts as a fallback. RU draws uniformly over every qualifying hex of
the map. A unit with no qualifying hex at all is left out (removed).

Last seen (LS): the mover's sighting record at its end_turn decision (rows
(id, type, hp, max hp, x, y)); each row whose unit is an opponent unit
hidden in the world's true state puts that unit, its own record with it, at
the sighting's hex with the sighting's hit points (at most its maximum,
counted when clamped), when the mover does not see that hex and no unit of
the world holds it; otherwise the unit is left out. Rows are taken in id
order. The opponent's other hidden units are removed.

A placed unit is moved with `GameCore.update_unit` and keeps its id, so the
mover's sighting record does not mark it gone: what the mover observes of
the world is what it observes of the game (tests/test_belief_worlds.py). A
side whose leader was removed stays alive for the game-over check
(`WesnothSim.hidden_leader_sides`).
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from wesnoth_ai.classes import opponent_of
from wesnoth_ai.lookahead_world import (World, build_world, display_villages, hidden_units, leader_sides,
                                        refog_opponent, remove_units, set_economy)
from wesnoth_ai.rules.terrain_resolver import UNREACHABLE_COST

ARM_LABELS = {"true": "T", "true_units": "TU", "roster_belief": "RB", "roster_uniform": "RU",
              "last_seen": "LS", "none": "O2", "observed": "observed"}
ARMS = tuple(ARM_LABELS)
ESTIMATED_ARMS = ("true_units", "roster_belief", "roster_uniform", "last_seen", "none")
PLACE_SALT = ":place"

Hex = Tuple[int, int]
SightRow = Tuple[str, str, int, int, int, int]


@dataclass
class Placement:
    """What a world did with the opponent's units the mover cannot see:
    hidden = kept + placed + left_out + removed."""
    hidden: int = 0              # the opponent's units the mover cannot see in the true state
    kept: int = 0                # left where they stand (T, TU)
    placed: int = 0              # put on a hex by the arm's rule (RB, RU, LS)
    fallbacks: int = 0           # of the placed, RB's units that took RU's rule
    left_out: int = 0            # removed for want of a hex (RB, RU) or of their sighting's (LS)
    removed: int = 0             # removed by the arm's rule (O2, observed; LS: never sighted)
    hp_clamped: int = 0          # LS: sightings above the unit's maximum hit points
    leader_hidden: bool = False  # the opponent's leader is among the hidden

    def record(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------
# The worlds
# ---------------------------------------------------------------------

def build_arm_world(post_sim, mover: int, arm: str, salt: str, *, turn_start: Optional[Tuple[int, int]] = None,
                    belief: Optional[Mapping[Hex, float]] = None,
                    sightings: Optional[Sequence[SightRow]] = None) -> Tuple[World, Placement]:
    """The world of `arm` for `mover` from `post_sim` (module docstring),
    its dice drawn from `salt`, and what it did with the hidden units.
    `turn_start`: the mover's (gold, base income) at the start of its turn,
    required by the estimated arms; `belief`: RB's probability per hex;
    `sightings`: LS's sighting record."""
    if arm not in ARM_LABELS:
        raise ValueError(f"unknown world {arm!r}; one of {ARMS}")
    mover = int(mover)
    opp = opponent_of(mover)
    theirs = sorted((d for d in hidden_units(post_sim.core.core, mover) if int(d["side"]) == opp),
                    key=lambda d: d["id"])
    placement = Placement(hidden=len(theirs), leader_hidden=any(d["is_leader"] for d in theirs))
    if arm == "observed":
        placement.removed = len(theirs)
        return build_world(post_sim, mover, "observed", salt), placement
    world = build_world(post_sim, mover, "godview", salt)
    world.determinization = arm
    if arm == "true":
        placement.kept = len(theirs)
        return world, placement
    if turn_start is None:
        raise ValueError(f"the {arm} world needs the mover's turn-start gold and income")
    cs = world.sim.core
    core = cs.core
    others = [d for d in hidden_units(core, mover) if int(d["side"]) != opp]
    remove_units(core, others)
    gone = list(others)
    if arm == "true_units":
        placement.kept = len(theirs)
    elif arm == "none":
        remove_units(core, theirs)
        gone += theirs
        placement.removed = len(theirs)
    elif arm == "last_seen":
        if sightings is None:
            raise ValueError("the last_seen world needs the mover's sighting record")
        gone += _apply(core, theirs, last_seen_hexes(cs, mover, theirs, sightings, placement))
    else:
        if arm == "roster_belief" and belief is None:
            raise ValueError("the roster_belief world needs the belief head's probabilities")
        weights = belief if arm == "roster_belief" else None
        gone += _apply(core, theirs, roster_hexes(cs, mover, theirs, placement_rng(salt), weights, placement))
    display_villages(cs, mover)
    set_economy(core, opp, int(turn_start[0]), int(turn_start[1]))
    refog_opponent(core, mover)
    bad = core.invariant_violation()
    if bad is not None:
        raise RuntimeError(f"the {arm} world breaks an invariant: {bad}")
    world.removed_ids = [d["id"] for d in gone]
    world.sim.hidden_leader_sides = frozenset(leader_sides(gone))
    return world, placement


def _apply(core, units: Sequence[dict], plan: Mapping[str, Optional[Dict]]) -> List[dict]:
    """Move each unit of `plan` to its fields (x, y, and hit points when
    given) and remove the others of `units`; returns the removed."""
    removed = [d for d in units if plan.get(d["id"]) is None]
    remove_units(core, removed)
    for d in units:
        fields = plan.get(d["id"])
        if fields is not None:
            core.update_unit(d["id"], fields)
    return removed


def placement_rng(salt: str) -> np.random.Generator:
    """The placement's own stream for the rollout salted `salt`."""
    digest = hashlib.sha256((salt + PLACE_SALT).encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


# ---------------------------------------------------------------------
# Where a unit may stand
# ---------------------------------------------------------------------

def free_hexes(cs, mover: int, lifted: Sequence[dict]) -> np.ndarray:
    """bool [H] in the core's hex order: the hexes the mover does not see
    and no unit of the world holds, the `lifted` units left out."""
    core = cs.core
    index = cs.geometry().pos_index
    free = np.asarray(core.seen_export(mover)) == 0
    skip = {d["id"] for d in lifted}
    for d in core.units_export():
        j = index.get((int(d["x"]), int(d["y"])))
        if d["id"] not in skip and j is not None:
            free[j] = False
    return free


def passable_for(core, unit: dict) -> np.ndarray:
    """bool [H]: the hexes whose movement cost for `unit` is below
    UNREACHABLE_COST (its movement class, unslowed: slowing doubles only
    the costs below it)."""
    return np.asarray(core.class_arrays(int(unit["class_id"]))[0]) < UNREACHABLE_COST


def belief_weights(cs, belief: Mapping[Hex, float]) -> np.ndarray:
    """float64 [H]: the belief per hex in the core's hex order, 0 on a hex
    without a token."""
    index = cs.geometry().pos_index
    out = np.zeros(len(index), dtype=np.float64)
    for hex_, p in belief.items():
        j = index.get((int(hex_[0]), int(hex_[1])))
        if j is not None:
            out[j] = float(p)
    return out


def _draw(rng: np.random.Generator, weights: np.ndarray) -> int:
    """An index drawn with probability proportional to `weights` (>= 0,
    some positive): a zero weight is never drawn."""
    cumulative = np.cumsum(weights)
    return int(np.searchsorted(cumulative, rng.random() * cumulative[-1], side="right"))


def roster_hexes(cs, mover: int, units: Sequence[dict], rng: np.random.Generator,
                 belief: Optional[Mapping[Hex, float]], placement: Placement) -> Dict[str, Optional[Dict]]:
    """Per unit id, its new hex (RB with `belief`, RU without) or None when
    no hex qualifies; the counts go to `placement`."""
    core = cs.core
    keys = cs.geometry().keys
    free = free_hexes(cs, mover, units)
    weights = None if belief is None else belief_weights(cs, belief)
    plan: Dict[str, Optional[Dict]] = {}
    for k in rng.permutation(len(units)).tolist():
        d = units[k]
        ok = free & passable_for(core, d)
        j = None
        if weights is not None:
            w = np.where(ok, weights, 0.0)
            if w.sum() > 0:
                j = _draw(rng, w)
        if j is None and ok.any():
            j = int(rng.choice(np.flatnonzero(ok)))
            if weights is not None:
                placement.fallbacks += 1
        if j is None:
            plan[d["id"]] = None
            placement.left_out += 1
            continue
        free[j] = False
        plan[d["id"]] = {"x": int(keys[j][0]), "y": int(keys[j][1])}
        placement.placed += 1
    return plan


def last_seen_hexes(cs, mover: int, units: Sequence[dict], sightings: Sequence[SightRow],
                    placement: Placement) -> Dict[str, Optional[Dict]]:
    """Per unit id, its last sighting's hex and hit points, or None (left
    out, or never sighted); the counts go to `placement`."""
    index = cs.geometry().pos_index
    by_id = {d["id"]: d for d in units}
    free = free_hexes(cs, mover, units)
    plan: Dict[str, Optional[Dict]] = {}
    for uid, _type, hp, _max_hp, x, y in sorted(tuple(r) for r in sightings):
        d = by_id.get(uid)
        if d is None or uid in plan:
            continue
        j = index.get((int(x), int(y)))
        if j is None or not free[j]:
            plan[uid] = None
            placement.left_out += 1
            continue
        free[j] = False
        hp = int(hp)
        if hp > int(d["max_hp"]):
            hp = int(d["max_hp"])
            placement.hp_clamped += 1
        plan[uid] = {"x": int(x), "y": int(y), "current_hp": hp}
        placement.placed += 1
    placement.removed = sum(1 for d in units if d["id"] not in plan)
    return plan


# ---------------------------------------------------------------------
# The belief head
# ---------------------------------------------------------------------

def belief_by_hex(policy, cs) -> Dict[Hex, float]:
    """The belief head of `policy` (a policy base: `_inference_encoder`,
    `_inference_model`) for the side to move at the core `cs`, its memory
    at 0 slots: per hex that has a token, the probability that a unit the
    side cannot see stands there. A token's hex is the map hex whose
    `tok_of_hex` entry names it, the index the belief targets are trained
    on (wesnoth_ai/belief_targets.py)."""
    import torch
    from wesnoth_ai.game_core import view_of
    from wesnoth_ai.memory import MemoryState
    encoder, model = policy._inference_encoder, policy._inference_model
    view = view_of(cs.fork())
    encoder.register_names(view)
    raw = encoder.raw_of(view)
    obs = raw.observation
    if obs is None or obs.tok_of_hex is None:
        raise ValueError("the encoding carries no observation with its token index (the parity recipe's)")
    memory = MemoryState(0, None) if getattr(model, "memory_slots", 0) else None
    with torch.no_grad():
        out = model(encoder.encode_from_raw(raw), memory=memory)
    if out.belief_logits is None:
        raise ValueError("the model has no belief head")
    logits = out.belief_logits.float().reshape(-1).cpu().numpy().astype(np.float64)
    if logits.shape[0] != raw.hex_xs.shape[0]:
        raise ValueError(f"{logits.shape[0]} belief logits for {raw.hex_xs.shape[0]} hex tokens")
    probs = 1.0 / (1.0 + np.exp(-logits))
    keys = obs.geometry.keys
    return {keys[m]: float(probs[t]) for m, t in enumerate(np.asarray(obs.tok_of_hex).tolist()) if t >= 0}
