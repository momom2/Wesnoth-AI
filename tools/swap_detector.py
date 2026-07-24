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

from tools.abilities import (                                      # noqa: E402
    is_backstab_active, opposite_hex, hex_neighbors, leadership_bonus,
)
from tools.combat_outcomes import (                                # noqa: E402
    OutcomeDistribution, enumerate_attack_outcomes,
)
from tools.pathfind_sim import ReachContext, unit_reach            # noqa: E402
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
                            fin, inc = _verify_reorder(
                                gs, cmd, opp, st.actions[j], st,
                                "backstab_setup")
                            inconclusive += inc
                            if fin is not None:
                                findings.append(fin)
                            break
        _apply_command(gs, cmd)
    return findings, inconclusive


def _has_leadership(name: str) -> bool:
    return "leadership" in (_stats_for(name).get("abilities", []) or [])


def _unit_level(name: str) -> int:
    try:
        return int(_stats_for(name).get("level", 1))
    except (TypeError, ValueError):
        return 1


def leadership_setup_findings(st: SideTurn) -> Tuple[List[Finding], int]:
    """A leadership-ability ally, strictly HIGHER level than the attacker,
    that moves ADJACENT to the attacker only AFTER the attack: reordering
    it first activates the +25%-per-level leadership damage bonus, whose
    outcome distribution strictly dominates the baseline."""
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
                    and leadership_bonus(att, gs.map.units) == 0):
                adj = set(hex_neighbors(ax, ay))
                att_lvl = _unit_level(att.name)
                for j in range(len(st.actions)):
                    if j == i:
                        continue
                    md = move_dests[j]
                    if md is not None and md in adj and _unit_at(gs, md) is None:
                        mv = _unit_at(gs, _move_start(st.actions[j]))
                        if (mv is not None and _has_leadership(mv.name)
                                and _unit_level(mv.name) > att_lvl):
                            fin, inc = _verify_reorder(
                                gs, cmd, md, st.actions[j], st,
                                "leadership_setup")
                            inconclusive += inc
                            if fin is not None:
                                findings.append(fin)
                            break
        _apply_command(gs, cmd)
    return findings, inconclusive


def _verify_reorder(gs: GameState, attack_cmd: list,
                    relocate_hex: Tuple[int, int], move_cmd: list,
                    st: SideTurn, motif: str) -> Tuple[Optional[Finding], int]:
    """Verify a pair-reorder: relocate `move_cmd`'s mover onto
    `relocate_hex` before `attack_cmd`, then compare the attack's outcome
    distribution WITH the mover in place (candidate) vs the actual state
    (baseline). Returns (Finding, inc): Finding iff candidate strictly
    dominates; inc=1 iff the engine was inconclusive."""
    ax, ay, dx, dy, w = attack_cmd[1:6]
    action = {"type": "attack", "start_hex": Position(ax, ay),
              "target_hex": Position(dx, dy), "attack_index": int(w)}
    d_base = enumerate_attack_outcomes(gs, action, advancement_choice="uniform")

    cand = _copy.deepcopy(gs)
    m_start = _move_start(move_cmd)
    mover = _unit_at(cand, m_start) if m_start else None
    if mover is None:
        return None, 0
    cand.map.units.discard(mover)
    cand.map.units.add(_rebuild_unit(
        mover, position=Position(relocate_hex[0], relocate_hex[1])))
    d_cand = enumerate_attack_outcomes(cand, action, advancement_choice="uniform")

    cmp = compare_distributions(d_base, d_cand)
    if cmp.verdict is Verdict.INCONCLUSIVE:
        return None, 1
    if cmp.verdict is Verdict.STRICTLY_BETTER:
        att = _unit_at(gs, (ax, ay))
        dfd = _unit_at(gs, (dx, dy))
        return Finding(
            st.game_id, st.turn, st.side, motif,
            att.name, dfd.name, (ax, ay), (dx, dy),
            cmp.verdict.value, cmp.vector), 0
    return None, 0


# =====================================================================
# Harness
# =====================================================================

def _reach(gs: GameState, mover):
    """Actual-movement reachability for `mover` in `gs` (god-view: the
    detector reasons over the TRUE board -- it is not the fogged policy).
    UnitReach: {hex: cost}, {hex: remaining_mp}, `landable` set."""
    ctx = ReachContext.for_side(gs, mover.side, god_view=True, exclude_unit=mover)
    return unit_reach(mover, gs, ctx, budget=int(mover.current_moves))


def pos_mp_dominates(gs: GameState, mover, dest: Tuple[int, int],
                     mp_target: int) -> bool:
    """(position, MP) dominance (user criterion 2026-07-24): a unit at its
    current hex with its current MP is >= a unit at `dest` with
    `mp_target` MP iff it can ACTUALLY reach `dest` (terrain/ZoC/blockers)
    and land there with >= mp_target MP left. Same hex -> cost 0 ->
    more-MP dominates."""
    if dest == (mover.position.x, mover.position.y):
        return int(mover.current_moves) >= mp_target   # already here, m MP
    r = _reach(gs, mover)
    return dest in r.landable and r.mp.get(dest, -1) >= mp_target


def _banked_mp(gs: GameState, mover, target_pos: Tuple[int, int],
               dest: Tuple[int, int]) -> Optional[float]:
    """MP the mover would spend reaching `dest` with the target REMOVED
    (the kill branch: the dead target no longer blocks/ZoCs, so this is a
    lower bound on the alive cost) -- i.e. the MP banked by NOT making the
    surround move on the kill branch. None if `dest` is unreachable even
    then."""
    g2 = _copy.deepcopy(gs)
    t = _unit_at(g2, target_pos)
    if t is not None:
        g2.map.units.discard(t)
    m2 = _unit_at(g2, (mover.position.x, mover.position.y))
    if m2 is None:
        return None
    r = _reach(g2, m2)
    return r.cost.get(dest) if dest in r.landable else None


def attacks_before_commit_findings(st: SideTurn) -> Tuple[List[Finding], int]:
    """A killable attack (P(kill) > 0) followed LATER in the turn by a
    move that ends ADJACENT to the same target -- a 'surround/support'
    move the kill would waste. Reordering the attack first BANKS that
    mover's MP on the kill branch.

    Unlike backstab/leadership this is NOT a Tier-1 dominance verdict: on
    the kill branch the mover ends on a DIFFERENT hex (it stays put), and
    'pure position' is product-incomparable by design -- so this is a
    structural / MP-lex-view OPPORTUNITY, reported with its kill
    probability (the banking gain), not a stochastic-dominance certificate.
    Heuristic: we can't read the mover's intent, only that its MP is at
    risk on the kill branch."""
    gs = _copy.deepcopy(st.pre_state)
    findings: List[Finding] = []
    inconclusive = 0
    for i, cmd in enumerate(st.actions):
        if cmd[0] == "attack":
            ax, ay, dx, dy, w = cmd[1], cmd[2], cmd[3], cmd[4], cmd[5]
            att = _unit_at(gs, (ax, ay))
            dfd = _unit_at(gs, (dx, dy))
            if att is not None and dfd is not None:
                action = {"type": "attack", "start_hex": Position(ax, ay),
                          "target_hex": Position(dx, dy),
                          "attack_index": int(w)}
                dist = enumerate_attack_outcomes(
                    gs, action, advancement_choice="uniform")
                if dist is None:
                    inconclusive += 1
                else:
                    p_kill = sum(p for k, p in dist.probs.items()
                                 if k[1] <= 0)          # defender dead
                    if p_kill > 1e-9:
                        tgt_adj = set(hex_neighbors(dx, dy))
                        for j in range(i + 1, len(st.actions)):
                            m = st.actions[j]
                            if m[0] != "move":
                                continue
                            dest, mst = _move_dest(m), _move_start(m)
                            if (dest in tgt_adj and mst is not None
                                    and mst != (ax, ay)):
                                mv = _unit_at(gs, mst)
                                if mv is None:
                                    break
                                # Validate + quantify via the (pos,MP)
                                # criterion: the banked mover (stays at X,
                                # keeps its MP) dominates the committed one
                                # (at Y with n MP) because it can still
                                # reach Y. Gain = P(kill) x MP banked.
                                banked = _banked_mp(gs, mv, (dx, dy), dest)
                                if banked is None:
                                    break     # surround hex not reachable
                                findings.append(Finding(
                                    st.game_id, st.turn, st.side,
                                    "attacks_before_commit",
                                    att.name, dfd.name, (ax, ay), (dx, dy),
                                    "banking_opportunity",
                                    {"kill_prob": f"{p_kill:.2f}",
                                     "mover": mv.name,
                                     "banks_mp": f"{banked:.0f}",
                                     "gain_mp": f"{p_kill * banked:.2f}"}))
                                break
        _apply_command(gs, cmd)
    return findings, inconclusive


# Motif registry: name -> side-turn generator. Extend here.
GENERATORS = {
    "backstab_setup":        backstab_setup_findings,
    "leadership_setup":      leadership_setup_findings,
    "attacks_before_commit": attacks_before_commit_findings,
}


def run_over_bundles(bundle_paths: List[Path]) -> dict:
    findings: List[Finding] = []
    n_games = n_turns = n_attacks = errors = 0
    inconclusive = {m: 0 for m in GENERATORS}
    for bp in bundle_paths:
        try:
            saw = False
            for st in load_side_turns(bp):
                saw = True
                n_turns += 1
                n_attacks += sum(1 for a in st.actions if a[0] == "attack")
                for m, gen in GENERATORS.items():
                    fs, inc = gen(st)
                    findings.extend(fs)
                    inconclusive[m] += inc
            if saw:
                n_games += 1
        except Exception as e:                    # noqa: BLE001
            errors += 1
            print(f"swap_detector: {Path(bp).name} failed: {e!r}")
    return {
        "games": n_games, "side_turns": n_turns, "attacks": n_attacks,
        "inconclusive": inconclusive, "errors": errors, "findings": findings,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Improvement (swap) detector")
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
    atk = rep["attacks"] or 1
    print("\n=== improvement detector ===")
    print(f"games:       {rep['games']}")
    print(f"side-turns:  {rep['side_turns']}")
    print(f"attacks:     {rep['attacks']}")
    print(f"load errors: {rep['errors']}")
    by_motif: Dict[str, List[Finding]] = {}
    for f in rep["findings"]:
        by_motif.setdefault(f.motif, []).append(f)
    print(f"\n{'motif':20s} {'fires':>6s} {'rate/attack':>12s} {'inconcl':>8s}")
    for m in GENERATORS:
        n = len(by_motif.get(m, []))
        print(f"{m:20s} {n:6d} {n / atk:12.4%} {rep['inconclusive'][m]:8d}")
    print(f"\ntotal firings: {len(rep['findings'])}")

    def _gain(f: Finding) -> float:
        try:
            return float(f.vector.get("gain_mp", 0.0))
        except (TypeError, ValueError):
            return 0.0

    for f in rep["findings"]:            # Tier-1 (theorem-grade): show all
        if f.motif in ("backstab_setup", "leadership_setup"):
            nz = {k: v for k, v in f.vector.items() if v != "="}
            print(f"  [{f.motif}] g={f.game_id} t{f.turn} s{f.side} "
                  f"{f.attacker}{f.attacker_pos}->{f.defender}{f.defender_pos} {nz}")
    abc = sorted((f for f in rep["findings"]
                  if f.motif == "attacks_before_commit"),
                 key=_gain, reverse=True)
    print(f"\n  top attacks_before_commit by banked-MP gain (of {len(abc)}):")
    for f in abc[:12]:
        print(f"    g={f.game_id} t{f.turn} s{f.side} "
              f"{f.attacker}->{f.defender} {f.vector}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
