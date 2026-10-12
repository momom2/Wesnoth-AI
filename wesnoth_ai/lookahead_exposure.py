"""The material-and-exposure value of a state, in gold, from a side's view
(docs/exposure_evaluator_prereg_20261012.md): M = material - exposure, read
off the state's Rust core in the world the look-ahead expands in
(wesnoth_ai/lookahead_world.py; in the observed world no unit the side
cannot see is on the board, so none contributes).

**Material.** Each unit's cost times its hit points over its maximum, the
side's units less its opponent's (the two player sides only; a unit's cost
is its current type's, the unit database's), plus the side's gold less the
opponent's as the world holds it (the observed world sets the opponent's
gold to the side's own at the decision). A recruit moves its cost from the
gold into a unit at full hit points, so the material is the same before and
after it.

**Exposure.** What the opponent's units can take on the opponent's next
turn. `GameCore.threats` (rust/wesnoth_core/src/threats.rs) gives, for each
opponent unit and each of the side's units it can attack, the attack's exact
expected hit-point loss to the target and the hex and weapons it is made
with: the target is a unit the opponent sees, not petrified; the attack hex
is next to the target, one the attacker reaches at its full movement through
the core's reach with zone of control, its own hex included (an attacker
already adjacent attacks with no move); the attacker's weapon is the one the
engine's attack dialog preselects (harm weight 0.5,
src/mouse_events.cpp:1286 at the 1.18.4 tag), the counter the engine's
`choose_defender_weapon`; per (attacker, target) the hex with the largest
expected loss. A loss is priced at the target's cost over its maximum hit
points, so a death costs the target's whole remaining value. The attacks are
then assigned greedily (`assign_greedily`): repeatedly the (attacker,
target) whose loss, capped by what is left of the target's value, is
largest; each attacker once, each target's total at most its value.

**Where the opponent's movement comes from.** Every opponent unit moves with
its full movement (`max_moves`) in every state. In the state after end_turn
the opponent's init_side has refreshed its units, so that is their movement
left there too; a mid-turn state holds what the opponent's units ended their
last turn with, which its next turn restores. Everything else is read as the
state stands: hit points, statuses (a unit slowed now fights slowed), the
time of day of the state's turn, the opponent's fog (its units' vision,
which covers the hexes they reach wherever vision costs equal movement
costs), the other units on the board (each attack is valued alone: two
attackers may be counted from one hex). In the state after end_turn the
material also holds what the opponent's init_side gave it, its income and
its healing, which no mid-turn state holds yet.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

from wesnoth_ai.classes import opponent_of

# The attack dialog preselects the weapon `better_attack` prefers at harm
# weight 0.5 (mouse_handler::fill_weapon_choices, src/mouse_events.cpp:1286,
# 1.18.4): the weapon a player attacks with unless they pick another.
DIALOG_HARM_WEIGHT = 0.5


@dataclass(frozen=True)
class Threat:
    """One opponent unit's best attack on one of the side's units: the
    attack hex, the attacker's weapon, the defender's counter (-1: none),
    the target's expected loss in gold and its death probability."""
    attacker: str
    target: str
    hex: Tuple[int, int]
    weapon: int
    counter: int
    loss: float
    death: float


@dataclass(frozen=True)
class Valuation:
    """A state's material and exposure for a side, in gold, and the
    attacks the core could not value exactly (`fallbacks`: weapon choices
    a heuristic made; `unvalued`: attack hexes whose fight the strike DP
    could not enumerate, left out)."""
    material: float
    exposure: float
    threats: int
    fallbacks: int
    unvalued: int

    @property
    def m(self) -> float:
        return self.material - self.exposure


def unit_value(unit: Mapping) -> float:
    """A unit's material: its cost times its hit points over its maximum."""
    return float(unit["cost"]) * int(unit["current_hp"]) / max(1, int(unit["max_hp"]))


def material(units: Sequence[Mapping], sides: Sequence[tuple], side: int) -> float:
    """The side's material less its opponent's (module docstring), from a
    core's `units_export()` and `sides_export()`."""
    them = opponent_of(side)
    total = 0.0
    for u in units:
        s = int(u["side"])
        if s == side:
            total += unit_value(u)
        elif s == them:
            total -= unit_value(u)
    return total + float(sides[side - 1][2]) - float(sides[them - 1][2])


def threats(core, side: int, units: Sequence[Mapping]) -> Tuple[List[Threat], int, int]:
    """(the opponent's attacks on the side's units with their losses in
    gold, the weapon choices a heuristic made, the attack hexes left
    unvalued), from the GameCore `core` whose units are `units`."""
    rows, fallbacks, unvalued = core.threats(opponent_of(side), side, DIALOG_HARM_WEIGHT)
    by_id = {u["id"]: u for u in units}
    out = []
    for attacker, target, x, y, weapon, counter, hp_loss, death in rows:
        t = by_id[target]
        gold_per_hp = float(t["cost"]) / max(1, int(t["max_hp"]))
        out.append(Threat(attacker, target, (int(x), int(y)), int(weapon), int(counter),
                          float(hp_loss) * gold_per_hp, float(death)))
    return out, int(fallbacks), int(unvalued)


def assign_greedily(attacks: Sequence[Threat], values: Mapping[str, float]) -> Tuple[float, List[Tuple[Threat, float]]]:
    """(the total, the attacks taken with what each counted): repeatedly the
    attack whose loss, capped by what is left of its target's value, is
    largest, the first listed among equals; each attacker once. A lazy
    max-heap: what is left of a target only falls, so an attack whose
    capped loss fell since it was pushed goes back with the new one."""
    left: Dict[str, float] = {t.target: float(values[t.target]) for t in attacks}
    heap = [(-t.loss, k) for k, t in enumerate(attacks) if t.loss > 0]
    heapq.heapify(heap)
    used = set()
    taken: List[Tuple[Threat, float]] = []
    total = 0.0
    while heap:
        key, k = heapq.heappop(heap)
        t = attacks[k]
        if t.attacker in used:
            continue
        gain = min(t.loss, left[t.target])
        if gain <= 0:
            continue
        if gain < -key:
            heapq.heappush(heap, (-gain, k))
            continue
        used.add(t.attacker)
        left[t.target] -= gain
        total += gain
        taken.append((t, gain))
    return total, taken


def valuation(core, side: int) -> Valuation:
    """The state's material and exposure for `side` (module docstring) on
    the GameCore `core`."""
    units = core.units_export()
    mat = material(units, core.sides_export(), side)
    attacks, fallbacks, unvalued = threats(core, side, units)
    values = {u["id"]: unit_value(u) for u in units if int(u["side"]) == side}
    exposure, _ = assign_greedily(attacks, values)
    return Valuation(mat, exposure, len(attacks), fallbacks, unvalued)
