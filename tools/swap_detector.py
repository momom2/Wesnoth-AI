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
from itertools import product as _product
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
from wesnoth_ai import combat as _cb                               # noqa: E402
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
# Sim-driven outcome enumeration (faithful materialization)
# =====================================================================
# The distributional side-turn verifier needs the FULL end-state
# distribution of a reordered turn, which means materializing every
# combat outcome. We do NOT re-implement the post-combat bookkeeping
# (XP, feeding, petrify, death, status write-back) -- that would drift
# from the sim. Instead we drive the sim's OWN applier
# (replay_dataset._apply_command) with a scripted hit/miss RNG and
# enumerate strike patterns by DFS, reading each strike's probability
# from the engine's own checkup. Result: bit-faithful children, proven
# against the exact DP by test_swap_detector's parity test.

class _EnumRNG:
    """Scripted PRNG for combat outcome enumeration. `resolve_attack`
    draws one `get_next_random() % 100` per strike (`hit = r < cth`). We
    force the first len(prefix) strikes (True->hit->0, False->miss->99);
    any draw BEYOND the prefix defaults to hit (0) and still bumps
    `calls`, so the DFS driver learns another strike exists to branch on."""
    __slots__ = ("prefix", "calls")

    def __init__(self, prefix: List[bool]):
        self.prefix = prefix
        self.calls = 0

    def get_next_random(self) -> int:
        i = self.calls
        self.calls += 1
        if i < len(self.prefix):
            return 0 if self.prefix[i] else 99
        return 0

    def get_random_int(self, low: int, high: int) -> int:   # defensive
        return low


def _apply_attack_scripted(gs: GameState, attack_cmd: list, rng: _EnumRNG) -> None:
    """Apply `attack_cmd` to `gs` with the sim's combat RNG replaced by
    `rng` (offline, single-threaded: a temporary global swap of
    combat.MTRng, which _apply_command's attack branch constructs exactly
    once)."""
    orig = _cb.MTRng
    _cb.MTRng = lambda *_a, **_k: rng      # noqa: E731
    try:
        _apply_command(gs, attack_cmd)
    finally:
        _cb.MTRng = orig


def _attack_action(attack_cmd: list) -> dict:
    ax, ay, dx, dy, w = attack_cmd[1:6]
    return {"type": "attack", "start_hex": Position(ax, ay),
            "target_hex": Position(dx, dy), "attack_index": int(w)}


def _advance_targets(name: str) -> list:
    return list(_stats_for(name).get("advances_to", []) or [])


def enumerate_children_via_sim(
    gs: GameState, attack_cmd: list, *, advancement_choice=None,
    max_leaves: int = 512,
) -> Optional[List[Tuple[GameState, float]]]:
    """Faithful outcome distribution for an attack command, as a list of
    (child_state, prob) normalized to 1 -- or None if the exact DP bails
    (complexity blow-up; or a levelling fight when `advancement_choice` is
    None) or the leaf count blows up.

    All post-combat bookkeeping is the sim's; we only enumerate which
    strikes land. DFS over strike hit/miss prefixes: at each node run the
    fight with the prefix forced (rest default-hit); if the engine drew a
    strike beyond the prefix, branch it (per its own chance-to-hit), else
    it's a leaf whose probability is the product of its realized strikes'
    per-strike chances.

    `advancement_choice="uniform"`: a leaf whose fight levels a unit is
    re-run once per advancement-target combination (the sim decides
    ELIGIBILITY; we only enumerate the CHOICE by forcing it via the
    `_advance_choices` queue, weighted 1/n uniform, attacker-first).
    Advancement's own draw lives on the separate salted channel, so it
    never disturbs the scripted combat RNG."""
    if enumerate_attack_outcomes(gs, _attack_action(attack_cmd),
                                 advancement_choice=advancement_choice) is None:
        return None
    ax, ay, dx, dy = attack_cmd[1], attack_cmd[2], attack_cmd[3], attack_cmd[4]
    att0, dfd0 = _unit_at(gs, (ax, ay)), _unit_at(gs, (dx, dy))
    if att0 is None or dfd0 is None:
        return None
    pre_type = {att0.id: att0.name, dfd0.id: dfd0.name}
    order = (att0.id, dfd0.id)             # attacker-first choice consumption

    def _run(prefix: List[bool], choices: Optional[list] = None):
        g = _copy.deepcopy(gs)
        if choices:
            g.global_info._advance_choices = list(choices)
        rng = _EnumRNG(prefix)
        _apply_attack_scripted(g, attack_cmd, rng)
        strikes = [s for s in
                   (getattr(g.global_info, "_last_checkup_strikes", []) or [])
                   if "chance" in s]
        return g, rng, strikes

    leaves: List[Tuple[GameState, float]] = []
    stack: List[List[bool]] = [[]]
    while stack:
        prefix = stack.pop()
        g2, rng, strikes = _run(prefix)
        if rng.calls != len(strikes):
            return None                    # unexpected extra draws (defensive)
        k = len(prefix)
        if rng.calls > k:                  # a strike past the prefix exists
            c = float(strikes[k]["chance"])
            if c >= 100.0:
                stack.append(prefix + [True])
            elif c <= 0.0:
                stack.append(prefix + [False])
            else:
                stack.append(prefix + [True])
                stack.append(prefix + [False])
            continue
        # leaf: the prefix fully resolved the fight.
        prob = 1.0
        for s in strikes:
            c = float(s["chance"]) / 100.0
            prob *= c if s["hits"] else (1.0 - c)
        if prob <= _EPS:
            continue
        # Advancement-choice enumeration (uniform). g2 used the default
        # targets[0]; re-run per combo only when a real (multi-target)
        # choice exists.
        advancers = []                     # (id, n_targets) in attacker-first order
        if advancement_choice == "uniform":
            for uid in order:
                u = _unit_by_id(g2, uid)
                if u is not None and u.name != pre_type[uid]:
                    advancers.append((uid, max(1, len(_advance_targets(pre_type[uid])))))
        n_combos = 1
        for _, n in advancers:
            n_combos *= n
        if advancers and n_combos > 1:
            wsplit = prob / float(n_combos)
            counts = [n for _, n in advancers]
            for combo in _product(*[range(n) for n in counts]):
                g3, _r3, _s3 = _run(prefix, list(combo))
                leaves.append((g3, wsplit))
                if len(leaves) > max_leaves:
                    return None
        else:
            leaves.append((g2, prob))
            if len(leaves) > max_leaves:
                return None
    total = sum(p for _, p in leaves)
    if not leaves or total <= 0:
        return None
    return [(g, p / total) for g, p in leaves]


# =====================================================================
# Side-turn reconstruction + distributional state comparison
# =====================================================================

def reconstruct_side_turn_dist(
    pre_state: GameState, actions: List[list], *, advancement_choice=None,
    max_particles: int = 512,
) -> Optional[List[Tuple[GameState, float]]]:
    """Forward distributional reconstruction of an ORDERED action list into
    the exact joint distribution over end-states: [(state, prob)] normalized
    to 1, or None if any attack is inconclusive (advancement / blow-up) or
    the particle set exceeds `max_particles`.

    Deterministic actions (move / recruit / recall) apply via the sim's own
    `_apply_command` to every particle; each attack branches every particle
    through the faithful sim-driven enumerator. The caller supplies a
    COHERENT ordering (each command's coordinates valid in that order); the
    baseline as-recorded list always is, and generators build reordered
    lists that are."""
    particles: List[Tuple[GameState, float]] = [(_copy.deepcopy(pre_state), 1.0)]
    for cmd in actions:
        if cmd[0] == "attack":
            nxt: List[Tuple[GameState, float]] = []
            for st, p in particles:
                children = enumerate_children_via_sim(
                    st, cmd, advancement_choice=advancement_choice)
                if children is None:
                    return None
                for child, pc in children:
                    nxt.append((child, p * pc))
                if len(nxt) > max_particles:
                    return None
            particles = nxt
        else:
            for st, _ in particles:
                _apply_command(st, cmd)           # deterministic, in place
    total = sum(p for _, p in particles)
    if not particles or total <= 0:
        return None
    return [(st, p / total) for st, p in particles]


def _pmarginal(particles: List[Tuple[GameState, float]],
               value_fn: Callable[[GameState], object]) -> Dict[object, float]:
    m: Dict[object, float] = {}
    for st, p in particles:
        v = value_fn(st)
        m[v] = m.get(v, 0.0) + p
    return m


def _num_sym(pc, pb, value_fn, more_is_better: bool) -> Sym:
    """Per-dimension symbol from first-order stochastic dominance of the
    candidate marginal over the baseline marginal (numeric value)."""
    mc, mb = _pmarginal(pc, value_fn), _pmarginal(pb, value_fn)
    fwd = _dominates(mc, mb, more_is_better)
    if fwd is True:
        return Sym.GT
    if fwd is False:
        return Sym.EQ
    rev = _dominates(mb, mc, more_is_better)
    return Sym.LT if rev is not None else Sym.INCOMP


def _marg_close(ma: Dict[object, float], mb: Dict[object, float]) -> bool:
    for k in set(ma) | set(mb):
        if abs(ma.get(k, 0.0) - mb.get(k, 0.0)) > 1e-9:
            return False
    return True


def _rollup(vec_syms: Dict[str, Sym]) -> Comparison:
    vec = {name: s.value for name, s in vec_syms.items()}
    vals = set(vec_syms.values())
    if Sym.INCOMP in vals:
        return Comparison(Verdict.INCOMPARABLE, vec)
    has_gt, has_lt = Sym.GT in vals, Sym.LT in vals
    if has_gt and has_lt:
        return Comparison(Verdict.INCOMPARABLE, vec)
    if has_gt:
        return Comparison(Verdict.STRICTLY_BETTER, vec)
    if has_lt:
        return Comparison(Verdict.WORSE, vec)
    return Comparison(Verdict.EQUAL, vec)


def compare_state_distributions(
    base_parts: Optional[List[Tuple[GameState, float]]],
    cand_parts: Optional[List[Tuple[GameState, float]]],
    side: int,
) -> Comparison:
    """Product-order dominance of two reconstructed side-turn end-state
    DISTRIBUTIONS (each from `reconstruct_side_turn_dist`) from `side`'s
    view. Per unit (matched by id): first-order stochastic dominance on
    health (hp, dead = -1), poisoned, slowed and XP -- with the good
    direction MIRRORED for enemy units (we want them lower-HP / more-
    debuffed / less XP) -- plus own gold. PURE POSITION is compared by
    marginal equality only (instrumental; the reachability (pos,MP)
    criterion needs a concrete board and lives in `compare_states` /
    per-branch banking, not here), so a unit that ends on a different hex
    with positive probability reads INCOMPARABLE -- conservative and sound.
    A None distribution -> INCONCLUSIVE (never sampled)."""
    if base_parts is None or cand_parts is None:
        return Comparison(Verdict.INCONCLUSIVE, {})

    id_side: Dict[object, int] = {}
    for parts in (base_parts, cand_parts):
        for st, _ in parts:
            for u in st.map.units:
                id_side.setdefault(u.id, u.side)

    def health(uid):
        return lambda st: (lambda u: -1 if u is None else int(u.current_hp))(
            _unit_by_id(st, uid))

    def status(uid, name):
        return lambda st: (lambda u: 0 if u is None
                           else (1 if name in (u.statuses or set()) else 0))(
            _unit_by_id(st, uid))

    def xp(uid):
        return lambda st: (lambda u: 0 if u is None
                           else int(getattr(u, "current_exp", 0)))(
            _unit_by_id(st, uid))

    def pos(uid):
        return lambda st: (lambda u: None if u is None
                           else (u.position.x, u.position.y))(
            _unit_by_id(st, uid))

    syms: Dict[str, Sym] = {}
    for uid, uside in id_side.items():
        own = (uside == side)
        syms[f"hp:{uid}"] = _num_sym(cand_parts, base_parts, health(uid), own)
        syms[f"pois:{uid}"] = _num_sym(
            cand_parts, base_parts, status(uid, "poisoned"), not own)
        syms[f"slow:{uid}"] = _num_sym(
            cand_parts, base_parts, status(uid, "slowed"), not own)
        syms[f"xp:{uid}"] = _num_sym(cand_parts, base_parts, xp(uid), own)
        mc = _pmarginal(cand_parts, pos(uid))
        mb = _pmarginal(base_parts, pos(uid))
        syms[f"pos:{uid}"] = Sym.EQ if _marg_close(mc, mb) else Sym.INCOMP

    def gold(st):
        s = next((s for s in st.sides if s.player == side), None)
        return getattr(s, "current_gold", 0) if s else 0
    syms["gold"] = _num_sym(cand_parts, base_parts, gold, True)

    return _rollup(syms)


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


def _unit_by_id(gs: GameState, uid):
    return next((u for u in gs.map.units if u.id == uid), None)


def _unit_state_sym(base_gs: GameState, cand_gs: GameState, b, c) -> Sym:
    """Per-unit dominance of candidate vs baseline end-state, product order
    over (existence, HP, (pos,MP)). `b`/`c` are the same unit (by id) in
    the baseline / candidate state, or None if absent (dead/never-there)."""
    if b is None and c is None:
        return Sym.EQ
    if b is None:                 # exists in candidate only -> better for us
        return Sym.GT
    if c is None:                 # exists in baseline only -> worse
        return Sym.LT
    b_pos, b_mp = (b.position.x, b.position.y), int(b.current_moves)
    c_pos, c_mp = (c.position.x, c.position.y), int(c.current_moves)
    cand_ge = (c.current_hp >= b.current_hp
               and pos_mp_dominates(cand_gs, c, b_pos, b_mp))
    base_ge = (b.current_hp >= c.current_hp
               and pos_mp_dominates(base_gs, b, c_pos, c_mp))
    if cand_ge and base_ge:
        return Sym.EQ
    if cand_ge:
        return Sym.GT
    if base_ge:
        return Sym.LT
    return Sym.INCOMP


def compare_states(base_gs: GameState, cand_gs: GameState,
                   side: int) -> Comparison:
    """Product-order dominance of two CONCRETE end-states from `side`'s
    view: per own-unit (matched by id) over existence/HP/(pos,MP) -- using
    the reachability (pos,MP) criterion -- plus side gold. The building
    block for the side-turn-level verifier (a distributional layer over
    reordered turns comes on top)."""
    ids = {u.id for u in base_gs.map.units if u.side == side}
    ids |= {u.id for u in cand_gs.map.units if u.side == side}
    vec: Dict[str, str] = {}
    syms = set()
    for uid in sorted(ids, key=str):
        s = _unit_state_sym(base_gs, cand_gs,
                            _unit_by_id(base_gs, uid),
                            _unit_by_id(cand_gs, uid))
        vec[f"unit:{uid}"] = s.value
        syms.add(s)
    # side gold (more is better).
    def _gold(gs):
        s = next((s for s in gs.sides if s.player == side), None)
        return getattr(s, "current_gold", 0) if s else 0
    gb, gc = _gold(base_gs), _gold(cand_gs)
    if gc != gb:
        gs_sym = Sym.GT if gc > gb else Sym.LT
        vec["gold"] = gs_sym.value
        syms.add(gs_sym)

    if Sym.INCOMP in syms:
        return Comparison(Verdict.INCOMPARABLE, vec)
    has_gt, has_lt = Sym.GT in syms, Sym.LT in syms
    if has_gt and has_lt:
        return Comparison(Verdict.INCOMPARABLE, vec)
    if has_gt:
        return Comparison(Verdict.STRICTLY_BETTER, vec)
    if has_lt:
        return Comparison(Verdict.WORSE, vec)
    return Comparison(Verdict.EQUAL, vec)


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
