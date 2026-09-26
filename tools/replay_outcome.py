"""The outcome of a recorded game, read from its extracted record: the
imitation corpus's winner labeller.

The rules, in order, over the record as extracted (cut at the game's end,
`replay_control.find_game_end`):

  1. LEADER_DEATH: a player side that started with a leader has none at
     the end of the record; that side lost.
  2. SURRENDER: the server names a side as surrendering (the record's
     `game_end`); that side lost, unless it was ahead on material at
     the surrender -- its non-leader units' cost times HP fraction plus
     its gold above AHEAD_MARGIN times the other side's -- which makes
     the game ABANDONED: no winner, so no policy weight and no value
     label. A surrender whose two sources name different sides is
     ABANDONED too.
  3. LEFT: the record was cut where a side's turn was taken by the other
     side's player (a leave or disconnect without return); that side
     lost. Whether these games label anything is the corpus config's
     choice (`outcome_classes`).
  4. INCONCLUSIVE otherwise.

Replaces the script that wrote training/logs/replay_outcomes.jsonl.gz,
which is in no commit; on a sample it named the surrendering side as
the winner in 8 of 108 surrender games (docs/corpus_v2_20260926.md).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Set

from tools.replay_control import PLAYED_BY_OPPONENT, SURRENDER
from wesnoth_ai.classes import PLAYER_SIDES, opponent_of
from wesnoth_ai.material import material_of_units

LEADER_DEATH = "leader_death"
ABANDONED = "abandoned"
LEFT = "left"
INCONCLUSIVE = "inconclusive"

# A surrendering side counts as ahead when its material exceeds the
# other side's by this factor (the 2026-09-26 crawl's threshold).
AHEAD_MARGIN = 1.05


@dataclass
class Outcome:
    """`outcome` is one of the classes above; `winner_side` is set for
    LEADER_DEATH, SURRENDER and LEFT. `material` is each player side's
    material at the end of the record."""
    outcome: str
    winner_side: Optional[int] = None
    leader_death_side: int = 0
    surrender_side: Optional[int] = None
    n_turns: int = 0
    material: Dict[int, float] = field(default_factory=dict)

    def as_row(self) -> dict:
        row = asdict(self)
        row["material"] = {str(k): round(v, 1) for k, v in self.material.items()}
        return row


def side_material(game_state, side: int) -> float:
    """The side's non-leader units' cost times HP fraction, plus its gold."""
    units = [u for u in game_state.map.units if u.side == side and not u.is_leader]
    return material_of_units(units, side) + float(game_state.sides[side - 1].current_gold)


def replay_to_end(record: dict):
    """(state at the start, state at the end) of an extracted record."""
    import copy

    from tools.replay_dataset import (_apply_command, _build_initial_gamestate,
                                      _setup_scenario_events)
    gs = _build_initial_gamestate(record)
    _setup_scenario_events(gs, record.get("scenario_id", ""))
    start = copy.copy(gs.map.units)
    for cmd in record.get("commands", []):
        _apply_command(gs, cmd)
    return start, gs


def label_outcome(record: dict) -> Outcome:
    """The outcome of an extracted record (see the module docstring)."""
    start_units, gs = replay_to_end(record)
    return decide_outcome(
        led={u.side for u in start_units if u.is_leader},
        alive={u.side for u in gs.map.units if u.is_leader},
        material={s: side_material(gs, s) for s in PLAYER_SIDES},
        game_end=record.get("game_end") or {},
        n_turns=int(gs.global_info.turn_number))


def decide_outcome(led: Set[int], alive: Set[int], material: Dict[int, float],
                   game_end: dict, n_turns: int = 0) -> Outcome:
    """The rules of the module docstring over what the record's end
    shows: the sides with a leader at the start (`led`) and at the end
    (`alive`), each player side's material at the end, and the
    record's `game_end`."""
    base = dict(n_turns=n_turns, material=material)
    dead = [s for s in PLAYER_SIDES if s in led and s not in alive]
    if len(dead) == 1:
        return Outcome(LEADER_DEATH, winner_side=opponent_of(dead[0]),
                       leader_death_side=dead[0], **base)
    if game_end.get("reason") == SURRENDER:
        side = game_end.get("surrender_side")
        if side not in PLAYER_SIDES or not game_end.get("surrender_sides_agree", True):
            return Outcome(ABANDONED, surrender_side=side, **base)
        if material[side] > AHEAD_MARGIN * material[opponent_of(side)]:
            return Outcome(ABANDONED, surrender_side=side, **base)
        return Outcome(SURRENDER, winner_side=opponent_of(side), surrender_side=side, **base)
    taken = game_end.get("played_by_opponent_side")
    if game_end.get("reason") == PLAYED_BY_OPPONENT and taken in PLAYER_SIDES:
        return Outcome(LEFT, winner_side=opponent_of(taken), **base)
    return Outcome(INCONCLUSIVE, **base)
