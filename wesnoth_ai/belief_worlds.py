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

Placement (RB, RU) draws from its own stream, `placement_rng(salt)`. A hex
qualifies for a unit when the mover does not see it, no unit of the world
holds it (the units already placed included, the hidden units not yet
placed excluded) and the unit's movement cost there is below
UNREACHABLE_COST. The opponent's leader, when hidden, goes first to its
side's starting position (`side_start`: the map's start marker) when that
hex qualifies; otherwise it is placed as the others. RB reads the belief
head (`belief_by_hex`) as each hex token's own probability that a hidden
unit stands there: every token hex is occupied with its probability, one
independent draw each; the units still unplaced, in a random order, take
the occupied hexes that qualify for them, in a random order (an occupied hex
left over stays empty), and the units left over take RU's rule over the
qualifying hexes without a token, counted as fallbacks. RU draws each
unit's hex, in a random order, uniformly over every qualifying hex of the
map. A unit with no hex left under its rule is left out (removed).

Last seen (LS): the mover's sighting record at its end_turn decision (rows
(id, type, hp, max hp, x, y)); each row whose unit is an opponent unit
hidden in the world's true state puts that unit, its own record with it, at
the sighting's hex with the sighting's hit points (at most its maximum,
counted when clamped), when the mover does not see that hex and no unit of
the world holds it; otherwise the unit is left out. Rows are taken in id
order. A hidden leader not placed so then goes to its side's starting
position when that hex qualifies (with its own hit points). The opponent's
other hidden units are removed.

A placed unit is moved with `GameCore.update_unit` and keeps its id, so the
mover's sighting record does not mark it gone: what the mover observes of
the world is what it observes of the game (tests/test_belief_worlds.py). A
side whose leader was removed stays alive for the game-over check
(`WesnothSim.hidden_leader_sides`).
"""
from __future__ import annotations

import functools
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
PLACING_ARMS = ("roster_belief", "roster_uniform", "last_seen")
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
    fallbacks: int = 0           # of the placed, RB's units that took RU's rule off the token hexes
    occupied: int = 0            # RB: the token hexes its draws occupied
    left_out: int = 0            # removed for want of a hex (RB, RU) or of their sighting's (LS)
    removed: int = 0             # removed by the arm's rule (O2, observed; LS: never sighted)
    hp_clamped: int = 0          # LS: sightings above the unit's maximum hit points
    leader_hidden: bool = False  # the opponent's leader is among the hidden
    leader_at_start: bool = False  # RB, RU, LS: the hidden leader went to its side's starting position

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
    start = side_start(cs, opp) if arm in PLACING_ARMS and placement.leader_hidden else None
    if arm == "true_units":
        placement.kept = len(theirs)
    elif arm == "none":
        remove_units(core, theirs)
        gone += theirs
        placement.removed = len(theirs)
    elif arm == "last_seen":
        if sightings is None:
            raise ValueError("the last_seen world needs the mover's sighting record")
        gone += _apply(core, theirs, last_seen_hexes(cs, mover, theirs, sightings, placement, start))
    else:
        if arm == "roster_belief" and belief is None:
            raise ValueError("the roster_belief world needs the belief head's probabilities")
        weights = belief if arm == "roster_belief" else None
        gone += _apply(core, theirs, roster_hexes(cs, mover, theirs, placement_rng(salt), weights, placement,
                                                  start))
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


@functools.lru_cache(maxsize=16)
def _map_starts(map_data: str) -> Dict[int, Hex]:
    from wesnoth_ai.rules.wml_state import map_starting_positions
    return map_starting_positions(map_data)


def side_start(cs, side: int) -> Hex:
    """`side`'s starting position, 0-indexed: its start marker on the map
    of the game's record (wml_state.map_starting_positions over the map
    data the state keeps)."""
    start = _map_starts(str(cs.statics.get("_raw_map_data") or "")).get(int(side))
    if start is None:
        raise ValueError(f"the game's map names no starting position for side {side}")
    return start


def token_hexes(cs, belief: Mapping[Hex, float]) -> Tuple[np.ndarray, np.ndarray]:
    """(the map indices of the hexes that have a token, ascending; the
    belief at each), in the core's hex order."""
    index = cs.geometry().pos_index
    pairs = sorted((index[(int(h[0]), int(h[1]))], float(p)) for h, p in belief.items()
                   if (int(h[0]), int(h[1])) in index)
    return np.array([j for j, _ in pairs], dtype=np.int64), np.array([p for _, p in pairs], dtype=np.float64)


def _leader(units: Sequence[dict]) -> Optional[dict]:
    return next((d for d in units if d["is_leader"]), None)


def _start_index(cs, unit: dict, start: Optional[Hex], free: np.ndarray) -> Optional[int]:
    """The map index of `start` when it qualifies for `unit`, else None."""
    if start is None:
        return None
    j = cs.geometry().pos_index.get((int(start[0]), int(start[1])))
    return j if j is not None and free[j] and passable_for(cs.core, unit)[j] else None


def _uniform(rng: np.random.Generator, ok: np.ndarray) -> Optional[int]:
    """A qualifying index drawn uniformly, or None when none qualifies."""
    choices = np.flatnonzero(ok)
    return int(rng.choice(choices)) if len(choices) else None


def roster_hexes(cs, mover: int, units: Sequence[dict], rng: np.random.Generator,
                 belief: Optional[Mapping[Hex, float]], placement: Placement,
                 start: Optional[Hex] = None) -> Dict[str, Optional[Dict]]:
    """Per unit id, its new hex (RB with `belief`, RU without) or None when
    no hex is left for it under its rule; the counts go to `placement`.
    `start`: the hidden leader's starting position."""
    core = cs.core
    keys = cs.geometry().keys
    free = free_hexes(cs, mover, units)
    plan: Dict[str, Optional[Dict]] = {}

    def place(d: dict, j: Optional[int]) -> bool:
        if j is None:
            return False
        free[j] = False
        plan[d["id"]] = {"x": int(keys[j][0]), "y": int(keys[j][1])}
        placement.placed += 1
        return True

    rest = list(units)
    leader = _leader(units)
    if leader is not None and place(leader, _start_index(cs, leader, start, free)):
        placement.leader_at_start = True
        rest = [d for d in rest if d is not leader]
    order = [rest[k] for k in rng.permutation(len(rest)).tolist()]
    if belief is None:
        left = [d for d in order if not place(d, _uniform(rng, free & passable_for(core, d)))]
    else:
        left = _belief_hexes(cs, order, rng, belief, free, place, placement)
    for d in left:
        plan[d["id"]] = None
    placement.left_out += len(left)
    return plan


def _belief_hexes(cs, order: Sequence[dict], rng: np.random.Generator, belief: Mapping[Hex, float],
                  free: np.ndarray, place, placement: Placement) -> List[dict]:
    """RB's rule for the units of `order`: the token hexes occupied by one
    draw each, taken in a random order by the units they qualify for, the
    others off the tokens by RU's rule. Returns the units left out."""
    core = cs.core
    tokens, probs = token_hexes(cs, belief)
    occupied = tokens[rng.random(len(tokens)) < probs]
    placement.occupied = len(occupied)
    open_hexes = [int(occupied[k]) for k in rng.permutation(len(occupied)).tolist()]
    unplaced = []
    for d in order:
        ok = free & passable_for(core, d)
        j = next((h for h in open_hexes if ok[h]), None)
        if place(d, j):
            open_hexes.remove(j)
        else:
            unplaced.append(d)
    off_token = np.ones(len(free), dtype=bool)
    off_token[tokens] = False
    left = []
    for d in unplaced:
        if place(d, _uniform(rng, free & passable_for(core, d) & off_token)):
            placement.fallbacks += 1
        else:
            left.append(d)
    return left


def last_seen_hexes(cs, mover: int, units: Sequence[dict], sightings: Sequence[SightRow],
                    placement: Placement, start: Optional[Hex] = None) -> Dict[str, Optional[Dict]]:
    """Per unit id, its last sighting's hex and hit points (the hidden
    leader not placed so: its starting position when that hex qualifies),
    or None; the counts go to `placement`."""
    index = cs.geometry().pos_index
    by_id = {d["id"]: d for d in units}
    free = free_hexes(cs, mover, units)
    plan: Dict[str, Optional[Dict]] = {}
    left = set()
    for uid, _type, hp, _max_hp, x, y in sorted(tuple(r) for r in sightings):
        d = by_id.get(uid)
        if d is None or uid in plan or uid in left:
            continue
        j = index.get((int(x), int(y)))
        if j is None or not free[j]:
            left.add(uid)
            continue
        free[j] = False
        hp = int(hp)
        if hp > int(d["max_hp"]):
            hp = int(d["max_hp"])
            placement.hp_clamped += 1
        plan[uid] = {"x": int(x), "y": int(y), "current_hp": hp}
        placement.placed += 1
    leader = _leader(units)
    if leader is not None and leader["id"] not in plan:
        j = _start_index(cs, leader, start, free)
        if j is not None:
            plan[leader["id"]] = {"x": int(start[0]), "y": int(start[1])}
            placement.placed += 1
            placement.leader_at_start = True
            left.discard(leader["id"])
    placement.left_out = len(left)
    placement.removed = sum(1 for d in units if d["id"] not in plan and d["id"] not in left)
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
