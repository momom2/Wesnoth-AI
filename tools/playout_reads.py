"""Two side readings of a playout, cheaper and less noisy than its
outcome (docs/turn_value_prereg_20260925.md).

  horizon  the value and the HP margin at the playout's start and at
           each player side's turn start after it, up to a count: the
           position a few side-turns later, as an auxiliary target.
  luck     over the playout's attacks, the realized change of the HP
           margin and of the kill margin minus the change the exact
           outcome distribution expects from the position before each
           attack. Its mean is zero given that position, so it can be
           subtracted from the outcome as a control variate.

Both are read after the playout, by replaying its commands from a copy
of its start position on the applier the simulator plays with
(`tools/replay_dataset._apply_command`, walked as tools/game_record.py
walks a record).

The zero mean holds only if the expectation is computed from the same
position under the same resolution rules as the applier. Three rules
are read from the command and the position rather than assumed: the
defender's counter-weapon (the command's must be the one
`choose_counter_weapon` picks, which is what the enumerator assumes);
the advancement choice (the first option, or with the simulator's
uniform channel on, uniform over the options the unit is offered); and
no queued advancement choice (the applier would pop it). An attack for
which one of them fails, or which the enumerator refuses, is skipped
and counted.

    reads = playout_reads(post_gs, [rc.cmd for rc in sim.command_history],
                          sim.recruit_rejections, mover=mover, value_of=value_of,
                          horizon_reads=4, luck=True, advance_salt=salt)
"""
from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

from tools.combat_outcomes import choose_counter_weapon, enumerate_attack_outcomes
from tools.game_record import _replay_steps
from wesnoth_ai.classes import GameState, Position, Unit

PLAYER_SIDES = (1, 2)

UnitKey = Tuple[str, int]                      # (id, side): a unit's identity on the board


# ---------------------------------------------------------------------
# Replaying a playout
# ---------------------------------------------------------------------

def start_state(post_gs: GameState, advance_salt: Optional[str] = None) -> GameState:
    """A deep copy of the playout's start position, ready to replay.

    With the simulator's uniform advancement channel on, an advancement
    is drawn from the channel's salt, which the simulator sets to its
    own seed salt at its first step; `advance_salt` must be that salt
    (the playout simulator's `_seed_salt`), or the replay would advance
    units differently from the playout."""
    gs = copy.deepcopy(post_gs)
    gi = gs.global_info
    if getattr(gi, "_advance_uniform", False):
        if advance_salt is None:
            raise ValueError("the uniform advancement channel is on: pass the playout "
                             "simulator's seed salt as advance_salt")
        gi._advance_salt = advance_salt
    return gs


def replay(gs: GameState, commands: Sequence[list],
           rejections: Sequence[Sequence[int]]) -> Iterator[Tuple[str, int, GameState, list]]:
    """("before", k, state, command) and ("after", k, state, command)
    for each command k of a playout applied to `gs`, the recruit
    rejections (command index, x, y) applied where they happened. The
    state is `gs` itself, mutated as the replay goes."""
    pseudo_record = {"commands": commands, "rejections": rejections}
    yield from _replay_steps(pseudo_record, gs, verify=False)


# ---------------------------------------------------------------------
# Horizon reads
# ---------------------------------------------------------------------

def hp_margin(gs: GameState, mover: int) -> int:
    """The mover's units' total hit points minus every other side's."""
    return sum(u.current_hp if u.side == mover else -u.current_hp for u in gs.map.units)


def _horizon_entry(gs: GameState, mover: int,
                   value_of: Optional[Callable[[GameState], Optional[float]]]) -> list:
    value = None if value_of is None else value_of(gs)
    return [None if value is None else float(value), hp_margin(gs, mover)]


# ---------------------------------------------------------------------
# Luck of one attack
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class AttackForecast:
    """One attack as its exact outcome distribution expects it, from
    the mover's side: the two combatants, their hit points before, and
    the expected change of the HP margin and of the kill margin over
    the two of them."""
    attacker: UnitKey
    defender: UnitKey
    hp_before: Tuple[int, int]
    signs: Tuple[int, int]                     # +1 for a unit of the mover's side, else -1
    hp_change: float
    kill_change: float


def _sign(side: int, mover: int) -> int:
    return 1 if side == mover else -1


def _kill_change(signs: Tuple[int, int], attacker_died: bool, defender_died: bool) -> int:
    """+1 per opponent unit killed, -1 per mover unit killed."""
    return -signs[0] * int(attacker_died) - signs[1] * int(defender_died)


def _unit_at(gs: GameState, x: int, y: int) -> Optional[Unit]:
    return next((u for u in gs.map.units if u.position.x == x and u.position.y == y), None)


def _advancement_rule(gs: GameState) -> Callable[[GameState, Unit, List[str]], List[float]]:
    """The advancement choice the applier makes on `gs`
    (`replay_dataset._advance_unit_once` with no queued choice), as the
    enumerator's probabilities over the unit type's `advances_to`: the
    first option, or with the uniform channel on, uniform over the
    options the unit is offered (a pick_advance narrowing, else all).
    The enumerator applies index i as the i-th offered option."""
    uniform = bool(getattr(gs.global_info, "_advance_uniform", False))

    def probs(_gs: GameState, unit: Unit, targets: List[str]) -> List[float]:
        if not uniform:
            return [1.0] + [0.0] * (len(targets) - 1)
        offered = [t for t in (getattr(unit, "_pickadvance", None) or ()) if t in targets]
        n = len(offered) or len(targets)
        return [1.0 / n] * n + [0.0] * (len(targets) - n)
    return probs


def forecast_attack(gs: GameState, cmd: list, mover: int) -> Optional[AttackForecast]:
    """The expected change of an attack command applied to `gs`, or
    None when the expectation would not describe what the applier does:
    an empty seed (the applier skips the attack), a queued advancement
    choice, a combatant missing, a defender weapon other than the one
    `choose_counter_weapon` picks, or a fight the enumerator refuses.

    The command is ["attack", ax, ay, dx, dy, weapon, defender weapon
    (-1: no retaliation), seed, advancement choices?], positions
    0-indexed (`replay_dataset._apply_command`)."""
    ax, ay, dx, dy, weapon = (int(v) for v in cmd[1:6])
    defender_weapon = int(cmd[6]) if len(cmd) > 6 else -1
    seed = cmd[7] if len(cmd) > 7 else ""
    queued = (len(cmd) > 8 and cmd[8]) or getattr(gs.global_info, "_advance_choices", None)
    if not seed or queued:
        return None
    att, dfd = _unit_at(gs, ax, ay), _unit_at(gs, dx, dy)
    if att is None or dfd is None:
        return None
    if choose_counter_weapon(gs, att, dfd, weapon) != defender_weapon:
        return None
    action = {"type": "attack", "start_hex": Position(ax, ay), "target_hex": Position(dx, dy),
              "attack_index": weapon}
    dist = enumerate_attack_outcomes(gs, action, advancement_choice=_advancement_rule(gs))
    if dist is None:
        return None
    signs = (_sign(att.side, mover), _sign(dfd.side, mover))
    hp_change = kill_change = 0.0
    for key, p in dist.probs.items():
        attacker_hp, defender_hp = key[0], key[1]
        hp_change += p * (signs[0] * (attacker_hp - att.current_hp)
                          + signs[1] * (defender_hp - dfd.current_hp))
        kill_change += p * _kill_change(signs, attacker_hp <= 0, defender_hp <= 0)
    return AttackForecast(attacker=(att.id, att.side), defender=(dfd.id, dfd.side),
                          hp_before=(att.current_hp, dfd.current_hp), signs=signs,
                          hp_change=hp_change, kill_change=kill_change)


def realized_change(gs_after: GameState, forecast: AttackForecast) -> Tuple[int, int]:
    """(HP margin change, kill margin change) the attack made, read from
    the position after it. A combatant is found by id and side (a plague
    corpse may take a dead unit's id, never its side); one that is gone
    was killed."""
    by_key = {(u.id, u.side): u for u in gs_after.map.units}
    after = [by_key.get(forecast.attacker), by_key.get(forecast.defender)]
    hp_change = sum(sign * ((u.current_hp if u is not None else 0) - before)
                    for sign, u, before in zip(forecast.signs, after, forecast.hp_before))
    return hp_change, _kill_change(forecast.signs, after[0] is None, after[1] is None)


class _LuckTally:
    """Realized minus expected, summed over a playout's attacks."""

    def __init__(self):
        self.hp = 0.0
        self.kills = 0.0
        self.attacks = 0
        self.skipped = 0

    def before(self, gs: GameState, cmd: list, mover: int) -> Optional[AttackForecast]:
        self.attacks += 1
        forecast = forecast_attack(gs, cmd, mover)
        if forecast is None:
            self.skipped += 1
        return forecast

    def after(self, gs: GameState, forecast: AttackForecast) -> None:
        hp_change, kill_change = realized_change(gs, forecast)
        self.hp += hp_change - forecast.hp_change
        self.kills += kill_change - forecast.kill_change

    def as_dict(self) -> Dict:
        return {"hp": self.hp, "kills": self.kills, "attacks": self.attacks,
                "skipped": self.skipped}


# ---------------------------------------------------------------------
# Both readings of one playout
# ---------------------------------------------------------------------

def playout_reads(post_gs: GameState, commands: Sequence[list],
                  rejections: Sequence[Sequence[int]], *, mover: int,
                  value_of: Optional[Callable[[GameState], Optional[float]]],
                  horizon_reads: int, luck: bool,
                  advance_salt: Optional[str] = None) -> Dict:
    """The side readings of a playout that started at `post_gs` and
    played `commands` (the playout simulator's command list) with
    `rejections` (its recruit rejections, (command index, x, y)).

    "horizon": up to `horizon_reads` entries [value, hp_margin]. Entry
    0 is read on `post_gs`; entry i >= 1 on the position right after
    the i-th init_side of a player side (the neutral side's are
    skipped). `value_of` returns the value already signed to the
    mover's side (None allowed, as is `value_of` None); it is given the
    replay's own state, which it must neither keep nor change.
    `hp_margin` is `hp_margin(state, mover)`. A list shorter than
    `horizon_reads` means the game ended first.

    "luck" (only when `luck`): {"hp", "kills", "attacks", "skipped"},
    the summed realized-minus-expected changes of the HP margin and the
    kill margin over the attacks read, the count of attack commands,
    and the count of those skipped (`forecast_attack` returned None).

    `advance_salt`: see `start_state`."""
    if horizon_reads <= 0 and not luck:
        return {"horizon": []}
    gs = start_state(post_gs, advance_salt)
    horizon = [_horizon_entry(gs, mover, value_of)] if horizon_reads > 0 else []
    tally = _LuckTally() if luck else None
    forecast: Optional[AttackForecast] = None
    for moment, _k, state, cmd in replay(gs, commands, rejections):
        kind = cmd[0]
        if moment == "before":
            if tally is not None and kind == "attack":
                forecast = tally.before(state, cmd, mover)
            continue
        if kind == "attack" and forecast is not None:
            tally.after(state, forecast)
            forecast = None
        elif kind == "init_side" and int(cmd[1]) in PLAYER_SIDES and len(horizon) < horizon_reads:
            horizon.append(_horizon_entry(state, mover, value_of))
        if tally is None and len(horizon) >= horizon_reads:
            break
    reads: Dict = {"horizon": horizon}
    if tally is not None:
        reads["luck"] = tally.as_dict()
    return reads
