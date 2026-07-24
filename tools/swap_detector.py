"""Swap detector v1 -- distributional pathwise dominance.

See docs/swap_detector_design.md. We compare two ORDERINGS of a
side-turn's actions by their OUTCOME DISTRIBUTIONS (no RNG coupling):
a reordering is a strict improvement iff, per comparison dimension, its
distribution stochastically dominates the baseline's (product order =
Tier-1), strict on >= 1 dimension.

v1 ships:
  - the per-dimension comparison vector + first-order stochastic
    dominance verifier over two `combat_outcomes.OutcomeDistribution`s
    (gate 2 = the thief-backstab theorem),
  - a loader that reconstructs an exported bundle into side-turns,
  - the `backstab_setup` generator + the run harness.
"""
from __future__ import annotations

import argparse
import copy as _copy
import glob as _glob
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

from tools.abilities import is_backstab_active, opposite_hex       # noqa: E402
from tools.combat_outcomes import (                                # noqa: E402
    OutcomeDistribution, enumerate_attack_outcomes,
)
from tools.replay_extract import extract_replay                    # noqa: E402
from tools.replay_dataset import (                                 # noqa: E402
    _build_initial_gamestate, _setup_scenario_events,
    _apply_command, _stats_for, _rebuild_unit,
)
from wesnoth_ai.classes import GameState, Position                 # noqa: E402

_EPS = 1e-9


# =====================================================================
# Distributional-dominance verifier
# =====================================================================

class Sym(Enum):
    """Per-dimension comparison symbol: candidate vs baseline."""
    GT = ">"            # candidate strictly dominates
    EQ = "="            # identical distributions on this dim
    LT = "<"            # baseline strictly dominates
    INCOMP = "incomp"   # neither stochastically dominates


class Verdict(Enum):
    STRICTLY_BETTER = "strictly_better"   # >= on all dims, > on >= 1
    EQUAL = "equal"
    WORSE = "worse"                        # baseline dominates
    INCOMPARABLE = "incomparable"
    INCONCLUSIVE = "inconclusive"          # engine returned None


@dataclass(frozen=True)
class Dim:
    name:           str
    value:          Callable[[tuple], float]
    more_is_better: bool


# OutcomeKey = (a_hp, d_hp, a_sl, d_sl, a_po, d_po, a_pe, d_pe,
#              a_type, d_type), attacker = a_*, defender = d_*. For an
# attack the acting side OWNS the attacker.
ATTACK_DIMS: Tuple[Dim, ...] = (
    Dim("enemy_hp",        lambda k: max(0, k[1]), more_is_better=False),
    Dim("enemy_alive",     lambda k: 1 if k[1] > 0 else 0, more_is_better=False),
    Dim("own_hp",          lambda k: max(0, k[0]), more_is_better=True),
    Dim("own_alive",       lambda k: 1 if k[0] > 0 else 0, more_is_better=True),
    Dim("enemy_poisoned",  lambda k: 1 if k[5] else 0, more_is_better=True),
    Dim("enemy_slowed",    lambda k: 1 if k[3] else 0, more_is_better=True),
    Dim("enemy_petrified", lambda k: 1 if k[7] else 0, more_is_better=True),
    Dim("own_poisoned",    lambda k: 1 if k[4] else 0, more_is_better=False),
    Dim("own_slowed",      lambda k: 1 if k[2] else 0, more_is_better=False),
)


def _marginal(dist: OutcomeDistribution,
              value: Callable[[tuple], float]) -> Dict[float, float]:
    m: Dict[float, float] = {}
    for key, p in dist.probs.items():
        v = value(key)
        m[v] = m.get(v, 0.0) + p
    return m


def _dominates(m_cand: Dict[float, float], m_base: Dict[float, float],
               more_is_better: bool) -> Optional[bool]:
    """First-order stochastic dominance of candidate over baseline.
    True = strict, False = equal, None = candidate does NOT dominate.

    more_is_better: candidate dominates iff its value is stochastically
    HIGHER -> its CDF at-or-below the baseline's everywhere.
    """
    vals = sorted(set(m_cand) | set(m_base))
    cc = bc = 0.0
    strict = False
    for v in vals:
        cc += m_cand.get(v, 0.0)
        bc += m_base.get(v, 0.0)
        if more_is_better:
            if cc > bc + _EPS:
                return None
            if cc < bc - _EPS:
                strict = True
        else:
            if cc < bc - _EPS:
                return None
            if cc > bc + _EPS:
                strict = True
    return strict


def _sym(dist_cand: OutcomeDistribution, dist_base: OutcomeDistribution,
         dim: Dim) -> Sym:
    m_c = _marginal(dist_cand, dim.value)
    m_b = _marginal(dist_base, dim.value)
    fwd = _dominates(m_c, m_b, dim.more_is_better)
    if fwd is True:
        return Sym.GT
    if fwd is False:
        return Sym.EQ
    rev = _dominates(m_b, m_c, dim.more_is_better)
    return Sym.LT if rev is not None else Sym.INCOMP


@dataclass
class Comparison:
    verdict: Verdict
    vector:  Dict[str, str]
    def is_improvement(self) -> bool:
        return self.verdict is Verdict.STRICTLY_BETTER


def compare_distributions(
    dist_base: Optional[OutcomeDistribution],
    dist_cand: Optional[OutcomeDistribution],
    dims: Tuple[Dim, ...] = ATTACK_DIMS,
) -> Comparison:
    """Product-order verdict of candidate vs baseline over `dims`. A
    `None` distribution (engine couldn't enumerate exactly) -> INCONCLUSIVE
    (never sampled)."""
    if dist_base is None or dist_cand is None:
        return Comparison(Verdict.INCONCLUSIVE, {})
    syms = {d.name: _sym(dist_cand, dist_base, d) for d in dims}
    vals = set(syms.values())
    vec = {name: s.value for name, s in syms.items()}
    if Sym.INCOMP in vals:
        return Comparison(Verdict.INCOMPARABLE, vec)
    has_gt = Sym.GT in vals
    has_lt = Sym.LT in vals
    if has_gt and has_lt:
        return Comparison(Verdict.INCOMPARABLE, vec)
    if has_gt:
        return Comparison(Verdict.STRICTLY_BETTER, vec)
    if has_lt:
        return Comparison(Verdict.WORSE, vec)
    return Comparison(Verdict.EQUAL, vec)


# =====================================================================
# Replay loader: bundle -> side-turns (pre-state + ordered actions)
# =====================================================================

@dataclass
class SideTurn:
    game_id:   str
    turn:      int
    side:      int
    pre_state: GameState        # state at the START of this side-turn
    actions:   List[list]       # compact command tuples for this side


def _bundle_bz2(path: Path) -> Optional[Path]:
    """The .bz2 for a bundle: a .tar (single .bz2 member -> temp file) or
    a .bz2 directly. Returns the path, or None."""
    path = Path(path)
    if path.suffix == ".bz2":
        return path
    if path.suffix == ".tar":
        with tarfile.open(path) as tf:
            members = [m for m in tf.getmembers() if m.name.endswith(".bz2")]
            if not members:
                return None
            tmp = Path(tempfile.mkdtemp(prefix="swapdet_"))
            tf.extract(members[0], tmp, filter="data")
            return tmp / members[0].name
    return None


def load_side_turns(bundle_path: Path) -> Iterator[SideTurn]:
    """Reconstruct a game from an exported bundle and yield each side-turn.
    Mirrors diff_replay's walk (extract -> build -> _apply_command loop),
    snapshotting at every `init_side` and grouping the side's move/attack/
    recruit commands until `end_turn`."""
    bz2_path = _bundle_bz2(bundle_path)
    if bz2_path is None:
        return
    data = extract_replay(bz2_path)
    if not data or not data.get("commands"):
        return
    game_id = data.get("game_id", Path(bundle_path).stem)
    gs = _build_initial_gamestate(data)
    _setup_scenario_events(gs, data.get("scenario_id", ""))

    side: Optional[int] = None
    pre: Optional[GameState] = None
    acts: List[list] = []

    def _flush() -> Optional[SideTurn]:
        if pre is not None and acts and side is not None:
            return SideTurn(game_id, int(pre.global_info.turn_number),
                            int(side), pre, list(acts))
        return None

    for cmd in data["commands"]:
        kind = cmd[0]
        if kind == "init_side":
            done = _flush()
            if done is not None:
                yield done
            _apply_command(gs, cmd)               # healing / income / ToD
            side = cmd[1]
            pre = _copy.deepcopy(gs)              # START-of-side-turn snapshot
            acts = []
            continue
        if kind in ("move", "attack", "recruit", "recall"):
            acts.append(cmd)
        _apply_command(gs, cmd)
        if kind == "end_turn":
            done = _flush()
            if done is not None:
                yield done
            pre, acts = None, []
    done = _flush()
    if done is not None:
        yield done


# =====================================================================
# backstab_setup generator + verifier
# =====================================================================

def _unit_at(gs: GameState, pos: Tuple[int, int]):
    return next((u for u in gs.map.units
                 if (u.position.x, u.position.y) == pos), None)


def _weapon_has_backstab(name: str, weapon_idx: int) -> bool:
    atks = _stats_for(name).get("attacks", [])
    if not (0 <= weapon_idx < len(atks)):
        return False
    return "backstab" in (atks[weapon_idx].get("specials", []) or [])


def _move_dest(cmd: list) -> Optional[Tuple[int, int]]:
    if cmd[0] != "move" or not cmd[1] or not cmd[2]:
        return None
    return (cmd[1][-1], cmd[2][-1])


def _move_start(cmd: list) -> Optional[Tuple[int, int]]:
    if cmd[0] != "move" or not cmd[1] or not cmd[2]:
        return None
    return (cmd[1][0], cmd[2][0])


@dataclass
class Finding:
    game_id:      str
    turn:         int
    side:         int
    motif:        str
    attacker:     str
    defender:     str
    attacker_pos: Tuple[int, int]
    defender_pos: Tuple[int, int]
    verdict:      str
    vector:       Dict[str, str]


def backstab_setup_findings(st: SideTurn) -> Tuple[List[Finding], int]:
    """Scan one side-turn for backstab_setup improvements. Returns
    (firings, n_inconclusive). A firing: a backstab-weapon attack that
    was NOT a backstab, where a later same-turn move puts a flanker on
    the opposite hex -- so reordering (move first) activates the backstab
    and its outcome distribution strictly dominates the baseline."""
    gs = _copy.deepcopy(st.pre_state)
    move_dests = [_move_dest(a) for a in st.actions]
    findings: List[Finding] = []
    inconclusive = 0
    for i, cmd in enumerate(st.actions):
        if cmd[0] == "attack":
            ax, ay, dx, dy, w = cmd[1], cmd[2], cmd[3], cmd[4], cmd[5]
            att = _unit_at(gs, (ax, ay))
            dfd = _unit_at(gs, (dx, dy))
            if (att is not None and dfd is not None
                    and _weapon_has_backstab(att.name, w)
                    and not is_backstab_active(att, dfd, gs.map.units)):
                opp = opposite_hex((dx, dy), (ax, ay))
                if opp is not None and _unit_at(gs, opp) is None:
                    for j in range(len(st.actions)):
                        if j != i and move_dests[j] == opp:
                            fin, inc = _verify_backstab(
                                gs, cmd, opp, st.actions[j], st)
                            inconclusive += inc
                            if fin is not None:
                                findings.append(fin)
                            break
        _apply_command(gs, cmd)
    return findings, inconclusive


def _verify_backstab(gs: GameState, attack_cmd: list,
                     opp: Tuple[int, int], move_cmd: list,
                     st: SideTurn) -> Tuple[Optional[Finding], int]:
    ax, ay, dx, dy, w = attack_cmd[1:6]
    action = {"type": "attack",
              "start_hex": Position(ax, ay),
              "target_hex": Position(dx, dy), "attack_index": int(w)}
    d_base = enumerate_attack_outcomes(gs, action, advancement_choice="uniform")

    cand = _copy.deepcopy(gs)
    m_start = _move_start(move_cmd)
    mover = _unit_at(cand, m_start) if m_start else None
    if mover is None:
        return None, 0
    cand.map.units.discard(mover)
    cand.map.units.add(_rebuild_unit(mover, position=Position(opp[0], opp[1])))
    d_cand = enumerate_attack_outcomes(cand, action, advancement_choice="uniform")

    cmp = compare_distributions(d_base, d_cand)
    if cmp.verdict is Verdict.INCONCLUSIVE:
        return None, 1
    if cmp.verdict is Verdict.STRICTLY_BETTER:
        att = _unit_at(gs, (ax, ay))
        dfd = _unit_at(gs, (dx, dy))
        return Finding(
            st.game_id, st.turn, st.side, "backstab_setup",
            att.name, dfd.name, (ax, ay), (dx, dy),
            cmp.verdict.value, cmp.vector), 0
    return None, 0


# =====================================================================
# Harness
# =====================================================================

def run_over_bundles(bundle_paths: List[Path]) -> dict:
    findings: List[Finding] = []
    n_games = n_turns = n_attacks = n_inconclusive = errors = 0
    for bp in bundle_paths:
        try:
            saw = False
            for st in load_side_turns(bp):
                saw = True
                n_turns += 1
                n_attacks += sum(1 for a in st.actions if a[0] == "attack")
                fs, inc = backstab_setup_findings(st)
                findings.extend(fs)
                n_inconclusive += inc
            if saw:
                n_games += 1
        except Exception as e:                    # noqa: BLE001
            errors += 1
            print(f"swap_detector: {Path(bp).name} failed: {e!r}")
    return {
        "games": n_games, "side_turns": n_turns, "attacks": n_attacks,
        "backstab_inconclusive": n_inconclusive, "errors": errors,
        "findings": findings,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Swap detector v1 (backstab_setup)")
    ap.add_argument("--bundles", default="training/validate_exports/bundles",
                    help="dir of .tar bundles (or a glob)")
    ap.add_argument("--limit", type=int, default=0, help="max bundles (0=all)")
    args = ap.parse_args(argv)

    pat = args.bundles
    if Path(pat).is_dir():
        pat = str(Path(pat) / "*.tar")
    paths = sorted(_glob.glob(pat))
    if args.limit:
        paths = paths[:args.limit]
    print(f"swap_detector: scanning {len(paths)} bundle(s)")

    rep = run_over_bundles([Path(p) for p in paths])
    print("\n=== swap_detector v1 (backstab_setup) ===")
    print(f"games:            {rep['games']}")
    print(f"side-turns:       {rep['side_turns']}")
    print(f"attacks:          {rep['attacks']}")
    print(f"backstab firings: {len(rep['findings'])}")
    print(f"inconclusive:     {rep['backstab_inconclusive']}")
    print(f"load errors:      {rep['errors']}")
    for f in rep["findings"][:40]:
        print(f"  FIRE g={f.game_id} t{f.turn} s{f.side} "
              f"{f.attacker}{f.attacker_pos}->{f.defender}{f.defender_pos} "
              f"{f.vector}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
