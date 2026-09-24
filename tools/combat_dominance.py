"""Combat rewrites certified by distributional dominance, strict and
relaxed (docs/xod_dominance_design_20260924.md).

A rewrite changes part of a realized side-turn (a weapon, an attack hex,
the order of a setup move or of two attacks) and is compared with what
was played on the exact outcome distributions of both. The comparison
is a vector of dimensions, each carrying enough to be read under any of
the 36 combinations of relaxations:

  binary dims  (a unit alive, a status, a level-up)   never relaxed
  hp dims      almost-dominance ratio eps             R2 admits eps <= its level
  xp dims      residual experience                    R1 drops them (keeps level-ups)
  pos dims     hex distance and the guard             R3 admits distance <= 2 (guarded or literal)
  vis dims     what the side sees and shows           R4 drops them

A combination admits a rewrite when no dimension it keeps is worse and
at least one dimension is better (tier D), or, for tier O, when the
rewrite keeps an option the played turn spent (the attack before a
surround move, whose mover keeps its movement on the kill branch). A
relaxation only removes constraints: a dimension it drops can no longer
block, and a better value on it still counts, so a looser combination
admits every rewrite a tighter one does.

Classes (section 5 of the design): W weapon, H attack hex, A ability
setup (backstab, leadership), K attack before a non-enabling move,
Q order of two attacks on one target, F the kill given to the unit it
levels (a Q rewrite that only R1 admits and that raises a level-up).
The generators read a realized side-turn: its start state and its
ordered commands.
"""
from __future__ import annotations

import copy
import itertools
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

from tools.combat_outcomes import (_kill_xp, enumerate_attack_outcomes)  # noqa: E402
from tools.replay_dataset import _apply_command, _stats_for  # noqa: E402
from wesnoth_ai.classes import GameState, Position  # noqa: E402

EPS = 1e-9
Hex = Tuple[int, int]

# ---------------------------------------------------------------------
# Dimensions and their comparison
# ---------------------------------------------------------------------
GT, EQ, LT, INCOMP = ">", "=", "<", "incomp"


@dataclass
class Dim:
    """One compared dimension of a rewrite: its symbol (candidate against
    the played turn) and what the relaxations read. `eps` is the share of
    the area between the two distribution functions on the wrong side
    (Leshno and Levy 2002), 0 when the candidate dominates, 1 when the
    played turn does."""
    name: str
    kind: str          # binary, hp, xp, pos, vis, count
    sym: str
    eps: float = 0.0
    distance: int = 0  # pos: hexes between the candidate's position and the played one
    guard: bool = True  # pos: the guarded form's conditions hold


def compare_marginals(mc: Dict[float, float], mb: Dict[float, float],
                      more_is_better: bool) -> Tuple[str, float]:
    """(symbol, eps) of candidate marginal `mc` against played `mb` over
    a numeric value. The candidate dominates when its distribution
    function lies at or below the played one everywhere (more is better)
    or at or above it (less is better). eps is the area where it does
    not, over the whole area between the two."""
    vals = sorted(set(mc) | set(mb))
    fc = fb = 0.0
    wrong = total = 0.0
    c_worse = c_better = False
    for a, b in zip(vals, vals[1:] + [None]):
        fc += mc.get(a, 0.0)
        fb += mb.get(a, 0.0)
        if b is None:
            break
        gap = (fc - fb) if more_is_better else (fb - fc)   # > 0: candidate worse here
        width = b - a
        if gap > EPS:
            c_worse = True
            wrong += gap * width
        elif gap < -EPS:
            c_better = True
        total += abs(gap) * width
    if not c_worse and not c_better:
        return EQ, 0.0
    if not c_worse:
        return GT, 0.0
    if not c_better:
        return LT, 1.0
    return INCOMP, (wrong / total if total > 0 else 1.0)


def _marginal(dist: List[Tuple[object, float]], fn: Callable) -> Dict[float, float]:
    m: Dict[float, float] = {}
    for x, p in dist:
        v = fn(x)
        m[v] = m.get(v, 0.0) + p
    return m


def numeric_dim(name: str, kind: str, cand, base, fn, more_is_better: bool) -> Dim:
    sym, eps = compare_marginals(_marginal(cand, fn), _marginal(base, fn), more_is_better)
    return Dim(name, kind, sym, eps)


# ---------------------------------------------------------------------
# Combinations of relaxations
# ---------------------------------------------------------------------
@dataclass(frozen=True, order=True)
class Combo:
    r1: bool = False               # experience only through level-ups
    r2: float = 0.0                # almost dominance on hp dims, 0 = off
    r3: str = ""                   # "", "guarded", "literal"
    r4: bool = False               # visibility ignored

    def name(self) -> str:
        parts = [p for p in ("r1" if self.r1 else "",
                             f"r2-{self.r2:g}" if self.r2 else "",
                             {"guarded": "r3g", "literal": "r3l"}.get(self.r3, ""),
                             "r4" if self.r4 else "") if p]
        return "+".join(parts) or "R0"

    def within(self, other: "Combo") -> bool:
        """Every axis at most as loose as `other`'s."""
        order3 = {"": 0, "guarded": 1, "literal": 2}
        return (self.r1 <= other.r1 and self.r2 <= other.r2
                and order3[self.r3] <= order3[other.r3] and self.r4 <= other.r4)


R2_LEVELS = (0.05, 0.15)
COMBOS: Tuple[Combo, ...] = tuple(
    Combo(r1, r2, r3, r4) for r1, r2, r3, r4 in itertools.product(
        (False, True), (0.0,) + R2_LEVELS, ("", "guarded", "literal"), (False, True)))


def dim_passes(d: Dim, c: Combo) -> Optional[bool]:
    """True: at least as good; False: worse; None: dropped by `c`."""
    if d.kind == "xp" and c.r1:
        return None
    if d.kind == "vis" and c.r4:
        return None
    if d.kind == "pos":
        if d.distance == 0:
            return True
        if not c.r3 or d.distance > 2:
            return False
        return c.r3 == "literal" or d.guard
    if d.sym in (GT, EQ):
        return True
    if d.kind == "hp" and c.r2 and d.eps <= c.r2 + EPS:
        return True
    return False


def admits(dims: List[Dim], tier: str, c: Combo) -> bool:
    if any(dim_passes(d, c) is False for d in dims):
        return False
    return tier == "O" or any(d.sym == GT for d in dims)


def minimal_combos(dims: List[Dim], tier: str) -> List[Combo]:
    """The smallest combinations that admit the rewrite."""
    ok = [c for c in COMBOS if admits(dims, tier, c)]
    return [c for c in ok if not any(o != c and o.within(c) for o in ok)]


# ---------------------------------------------------------------------
# Rewrites
# ---------------------------------------------------------------------
@dataclass
class Rewrite:
    klass: str
    tier: str                          # D or O
    turn: int
    side: int
    window: Tuple[int, ...]            # command indices within the side-turn
    detail: Dict[str, object]
    dims: List[Dim]
    gains: Dict[str, float] = field(default_factory=dict)

    def admitted(self) -> Dict[str, bool]:
        return {c.name(): admits(self.dims, self.tier, c) for c in COMBOS}


@dataclass
class SideTurn:
    game_id: str
    turn: int
    side: int
    pre_state: GameState
    commands: List[list]
    first_index: int                   # index of the side-turn's first command in the game


def side_turns(game_id: str, start: GameState, commands: List[list]) -> Iterator[SideTurn]:
    """The player side-turns of a game: the state after each player
    init_side and the commands up to its end_turn."""
    gs = start
    current: Optional[SideTurn] = None
    for k, cmd in enumerate(commands):
        kind = cmd[0] if cmd else ""
        if current is not None and kind != "init_side":
            current.commands.append(list(cmd))
        _apply_command(gs, cmd)
        if kind == "init_side" and len(cmd) > 1 and cmd[1] in (1, 2):
            current = SideTurn(game_id, gs.global_info.turn_number, int(cmd[1]),
                               copy.deepcopy(gs), [], k + 1)
        elif kind == "end_turn" and current is not None:
            yield current
            current = None
        elif kind == "init_side":
            current = None


# ---------------------------------------------------------------------
# Fights
# ---------------------------------------------------------------------
def _unit_at(gs: GameState, pos: Hex):
    return next((u for u in gs.map.units if (u.position.x, u.position.y) == pos), None)


def _level(name: str) -> int:
    try:
        return int(_stats_for(name).get("level", 1))
    except (TypeError, ValueError):
        return 1


def attack_action(cmd: list, weapon: Optional[int] = None) -> dict:
    return {"type": "attack", "start_hex": Position(cmd[1], cmd[2]),
            "target_hex": Position(cmd[3], cmd[4]),
            "attack_index": int(cmd[5] if weapon is None else weapon)}


def fight(gs: GameState, action: dict):
    """The exact outcome distribution of one attack, advancement resolved
    uniformly as self-play does; None when it cannot be enumerated."""
    return enumerate_attack_outcomes(gs, action, advancement_choice="uniform")


def fight_dims(cand, base, att, dfd) -> List[Dim]:
    """The attack's own dimensions, attacker = own unit. Keys of
    `combat_outcomes.OutcomeDistribution`: (a_hp, d_hp, a_slowed,
    d_slowed, a_poisoned, d_poisoned, a_petrified, d_petrified, a_type,
    d_type); a dead unit's hp reads -1."""
    c, b = list(cand.probs.items()), list(base.probs.items())
    a_lvl, d_lvl = _level(att.name), _level(dfd.name)

    def own_xp(k):
        if k[0] <= 0:
            return -1
        if k[8] != att.name:
            return 1_000_000                   # levelled: counted by the level-up dim
        return att.current_exp + (_kill_xp(d_lvl) if k[1] <= 0 else d_lvl)

    def enemy_xp(k):
        if k[1] <= 0:
            return -1
        if k[9] != dfd.name:
            return 1_000_000
        return dfd.current_exp + (_kill_xp(a_lvl) if k[0] <= 0 else a_lvl)

    return [
        numeric_dim("enemy_alive", "binary", c, b, lambda k: int(k[1] > 0), False),
        numeric_dim("enemy_hp", "hp", c, b, lambda k: k[1] if k[1] > 0 else -1, False),
        numeric_dim("own_alive", "binary", c, b, lambda k: int(k[0] > 0), True),
        numeric_dim("own_hp", "hp", c, b, lambda k: k[0] if k[0] > 0 else -1, True),
        numeric_dim("enemy_slowed", "binary", c, b, lambda k: int(bool(k[3])), True),
        numeric_dim("enemy_poisoned", "binary", c, b, lambda k: int(bool(k[5])), True),
        numeric_dim("enemy_petrified", "binary", c, b, lambda k: int(bool(k[7])), True),
        numeric_dim("own_slowed", "binary", c, b, lambda k: int(bool(k[2])), False),
        numeric_dim("own_poisoned", "binary", c, b, lambda k: int(bool(k[4])), False),
        numeric_dim("own_levelup", "binary", c, b, lambda k: int(k[0] > 0 and k[8] != att.name), True),
        numeric_dim("enemy_levelup", "binary", c, b, lambda k: int(k[1] > 0 and k[9] != dfd.name), False),
        numeric_dim("own_xp", "xp", c, b, own_xp, True),
        numeric_dim("enemy_xp", "xp", c, b, enemy_xp, False),
    ]


def fight_gains(cand, base) -> Dict[str, float]:
    def p_kill(d):
        return sum(p for k, p in d.probs.items() if k[1] <= 0)

    def e_hp(d, i):
        return sum(p * max(0, k[i]) for k, p in d.probs.items())
    return {"d_p_kill": p_kill(cand) - p_kill(base),
            "d_target_hp": e_hp(cand, 1) - e_hp(base, 1),
            "d_attacker_hp": e_hp(cand, 0) - e_hp(base, 0)}
