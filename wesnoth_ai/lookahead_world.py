"""The worlds the look-ahead player expands its candidates in, and the
outcome states of each candidate with their exact probabilities
(tools/lookahead_player.py).

**The observed world** (`determinization="observed"`), built on a fork of
the simulator for the side `side` about to decide, holds what that side
knows and nothing else:

  - every unit of another side that `side` does not see
    (`GameCore.visible_ids`: fogged, or a hider not uncovered) is removed;
    the side's sighting record keeps the ones it saw, marked gone, as the
    core keeps a unit that left the board out of the side's sight;
  - a village is owned as the side's display shows it: its own villages,
    and another side's only on a hex it sees (the engine draws no enemy
    flag on a fogged hex); the others are unowned, and every side's
    village count follows;
  - the opponent's gold and base income are set to the side's own: both
    sides of every pool scenario start with the same gold and base income,
    and the side cannot see the opponent's gold under fog (the enemy row of
    the status table shows its name only), so its own is the estimate;
    the opponent's upkeep and income then follow from the units and
    villages of this world;
  - the opponent's fog is recomputed from its units of this world
    (`refog_side`) and its sighting record emptied, so nothing the removed
    units saw reaches what the opponent observes in this world;
  - a side whose leader was removed stays alive for the game-over check
    (`WesnothSim.hidden_leader_sides`).

The side's own units, fog, sighting record and seen types are untouched,
so its observation of the world equals its observation of the game
(tests/test_lookahead_player.py). A move in this world clears fog onto
hexes that hold no hidden unit, and a village under fog shows the owner the
side's display shows; nothing hidden is revealed to the evaluator. A unit
created in this world (a recruit, a plague corpse) takes the id after the
largest on the world's board (`GameCore.next_uid`), which can repeat a
removed unit's id; the encoder reads unit ids only to break sort ties
(docs/observation_parity_20260926.md).

Left as they are: the opponent's seen types (the evidence of the side's
faction it gathered, read only when the side chose Random, which eval
games never do) and the scenario's event variables.

**God view** (`determinization="godview"`) expands on the true state: an
upper bound, never a fair player.

In both, the world draws its dice from its own stream, salted per decision
(`salt`), so a recruit's trait roll is one draw that does not foresee the
game's next draw. Advancements follow the simulator's rule on that stream.

**Outcomes.** A move, a recruit or end_turn gives one state (end_turn:
after the opponent's init_side, the neutral sides' turns between them
played). An attack gives every distinct state its fight can end in: the
pre-move step() would make first, then every hit/miss sequence of the
strikes run on a fork of the core with the strikes scripted, its
probability the product of the strikes' chances, sequences merged by the
core's state key; the probabilities sum to 1 (asserted). A state where a
leader died carries the game's result (+1 won, -1 lost, 0 neither) for
the deciding side.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

from wesnoth_ai.classes import PLAYER_SIDES, opponent_of

PROBABILITY_TOLERANCE = 1e-9


class ExpansionError(Exception):
    """A candidate whose outcomes could not be built; `reason` names why
    (the player counts it per reason)."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


@dataclass
class Outcome:
    """One outcome state of a candidate: its probability, the core holding
    it, who moves there, and the game's result for the deciding side when
    the game ended there (None otherwise)."""
    prob: float
    core: object
    side_to_move: int
    terminal: Optional[float] = None


@dataclass
class World:
    """A fork of the simulator to expand candidates on, for `side`."""
    sim: object
    side: int
    determinization: str
    removed_ids: List[str] = field(default_factory=list)


def build_world(sim, side: int, determinization: str, salt: str) -> World:
    """A fork of `sim` for `side` to expand its candidates on (module
    docstring), its dice drawn from the stream `salt`."""
    if sim.core is None:
        raise ExpansionError("no_core", "the look-ahead expands on the Rust core")
    fork = sim.fork()
    fork._is_search_fork = True
    fork._seed_salt = salt
    core = fork.core.core
    if core.globals_export()["advance_uniform"]:
        core.set_advance_salt(salt)
    removed: List[str] = []
    if determinization == "observed":
        removed, hidden_leaders = _observe(fork.core, side)
        fork.hidden_leader_sides = frozenset(hidden_leaders)
    elif determinization != "godview":
        raise ValueError(f"unknown determinization {determinization!r}")
    return World(sim=fork, side=int(side), determinization=determinization, removed_ids=removed)


def _observe(cs, side: int):
    """Turn the core `cs` into the observed world of `side` in place;
    returns the removed unit ids and the sides whose leader was removed."""
    core = cs.core
    visible = set(core.visible_ids(side))
    hidden = [d for d in core.units_export() if int(d["side"]) != side and d["id"] not in visible]
    for d in hidden:
        core.remove_unit(d["id"])
    seen = core.seen_export(side)
    index = cs.geometry().pos_index
    owners = [(int(x), int(y), int(s)) for x, y, s in core.village_owner_export()
              if int(s) == side or seen[index[(int(x), int(y))]]]
    core.set_village_owner(owners)
    count = Counter(s for _, _, s in owners)
    sides = core.sides_export()
    opp = opponent_of(side)
    own_gold, own_income = sides[side - 1][2], sides[side - 1][3]
    rows = []
    for k, (player, recruits, gold, income, _villages, faction) in enumerate(sides):
        if k + 1 == opp:
            gold, income = own_gold, own_income
        rows.append((player, list(recruits), int(gold), int(income), int(count.get(k + 1, 0)), faction))
    core.set_sides(rows)
    if opp in PLAYER_SIDES:
        core.refog_side(opp)
        core.set_sightings(opp, [])
        core.set_sightings_gone(opp, [])
    return [d["id"] for d in hidden], {int(d["side"]) for d in hidden if d["is_leader"]}


# ---------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------

def expand(world: World, action: dict, *, max_attack_leaves: int = 512) -> List[Outcome]:
    """The outcome states of `action` in `world`, with their probabilities;
    ExpansionError when they cannot be built."""
    kind = action.get("type", "end_turn")
    if kind == "attack":
        return _attack_outcomes(world, action, max_attack_leaves)
    child = world.sim.fork()
    child.step(action)
    if child.last_step_rejected:
        raise ExpansionError("refused", f"{kind}: {child.last_step_refusal}")
    played = child.command_history[0].kind if child.command_history else ""
    if played != kind:
        # The simulator ends the turn on an action it cannot translate.
        raise ExpansionError("not_applied", f"{kind} was played as {played or 'nothing'}")
    return [Outcome(1.0, child.core, int(child.current_side), _sim_result(child, world.side))]


def _sim_result(sim, side: int) -> Optional[float]:
    """The game's result for `side` when `sim` is over, else None."""
    if not sim.done:
        return None
    if sim.winner == side:
        return 1.0
    return -1.0 if sim.winner in PLAYER_SIDES else 0.0


def _core_result(core, side: int, hidden_leaders) -> Optional[float]:
    """The game's result for `side` after a fight on `core`: the
    simulator's leader rule (WesnothSim._check_game_over)."""
    alive = set(core.leader_sides()) | set(hidden_leaders)
    if all(s in alive for s in PLAYER_SIDES):
        return None
    if side in alive:
        return 1.0
    return -1.0 if any(s in alive for s in PLAYER_SIDES) else 0.0


def _approach(world: World, action: dict):
    """(a fork with the attacker next to its target, the adjacent attack
    command): the pre-move step() makes first, then the command it would
    apply."""
    child = world.sim.fork()
    start = action["start_hex"]
    hex_ = child.attack_hex_for(action)
    if hex_ is None:
        raise ExpansionError("no_attack_hex", f"{(start.x, start.y)}->{action['target_hex']}")
    if (hex_.x, hex_.y) != (start.x, start.y):
        mover = child.core.core.unit_id_at(start.x, start.y, world.side)
        child.step({"type": "move", "start_hex": start, "target_hex": hex_})
        if child.last_step_rejected or child.core.core.unit_id_at(hex_.x, hex_.y, world.side) != mover:
            raise ExpansionError("approach_failed", f"{mover} did not land on {(hex_.x, hex_.y)}")
        if child.done:
            return child, None
    cmd = child.attack_command({**action, "start_hex": hex_})
    if cmd is None:
        raise ExpansionError("no_attack_command", repr(action))
    return child, cmd


def _attack_outcomes(world: World, action: dict, max_leaves: int) -> List[Outcome]:
    """Every distinct state the attack can end in (module docstring)."""
    child, cmd = _approach(world, action)
    if cmd is None:
        return [Outcome(1.0, child.core, int(child.current_side), _sim_result(child, world.side))]
    ax, ay, dx, dy, a_weapon, d_weapon = (int(v) for v in cmd[1:7])
    base = child.core
    side = int(child.current_side)
    hidden = getattr(child, "hidden_leader_sides", frozenset())
    leaves = {}
    total = 0.0
    n_sequences = 0
    stack: List[List[bool]] = [[]]
    while stack:
        prefix = stack.pop()
        fight = base.core.fork()
        calls = fight.apply_attack_scripted(ax, ay, dx, dy, a_weapon, d_weapon, list(prefix), [])
        if calls is None:
            raise ExpansionError("attack_missing_unit", repr(cmd))
        strikes = fight.last_checkup_strikes_export()
        if int(calls) * 4 != len(strikes):
            raise ExpansionError("strike_record", f"{calls} draws, {len(strikes) // 4} strikes")
        k = len(prefix)
        if calls > k:
            chance = strikes[4 * k]
            if chance >= 100:
                stack.append(prefix + [True])
            elif chance <= 0:
                stack.append(prefix + [False])
            else:
                stack.append(prefix + [True])
                stack.append(prefix + [False])
            continue
        n_sequences += 1
        if n_sequences > max_leaves:
            raise ExpansionError("attack_leaves", f"more than {max_leaves} hit/miss sequences")
        prob = 1.0
        for j in range(calls):
            p_hit = strikes[4 * j] / 100.0
            prob *= p_hit if strikes[4 * j + 1] else 1.0 - p_hit
        total += prob
        key = int(fight.state_key())
        hit = leaves.get(key)
        if hit is None:
            leaves[key] = [prob, base.fork(core=fight)]
        else:
            hit[0] += prob
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=PROBABILITY_TOLERANCE):
        raise AssertionError(f"an attack's outcome probabilities sum to {total!r}, not 1")
    return [Outcome(p, cs, side, _core_result(cs.core, world.side, hidden)) for p, cs in leaves.values()]
