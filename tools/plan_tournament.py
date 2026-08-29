"""Incumbent-anchored turn-plan tournament with certify-or-abstain
distillation (proposition 1, docs/procedure_propositions_20260826.md,
user-approved 2026-08-26; adversarial review rounds 1-3 same day).

Per side-turn, three stages:

1. SELECTION (salted, cheap, biased-is-ok): the policy's own sampled
   turn is the INCUMBENT; challengers are ranked against it by
   paired projection margins under sequential halving. Prefix
   perturbations are built by forking the incumbent's OWN recorded
   pre-states (same ':inc' stream), so challenger and incumbent
   share the realized prefix dice exactly -- partial coupling, the
   one form of shared randomness Q8 permits (round-3 C6: replaying
   the prefix on a different stream silently degenerated the arm).
   The incumbent is projected once per (round, redraw) and reused
   across candidates -- a common offset cancels under ranking
   (round-3 C4).
2. CERTIFICATION (salted, variance-aware, fixed odd depth): the
   selected challenger's command list and the incumbent's are each
   re-executed per replicate via record_spine(actions=...) on fresh
   streams -- the SAME divergence-repair semantics serving uses, so
   the certified object IS the played object (round-3 C3:
   materialize()'s truncate-on-stale grading certified a different
   object than the respine served). Accepted only by
   the n-aware accept_rule over the completed replicates (>=2; a
   budget death keeps completed evidence, round-3 C2): mean >
   max(_T_CRIT[n]*sd/sqrt(n), band), alpha held at the n=3 legacy
   level for every reachable n (round-8/9). An argmax over
   selection draws never certifies (round-2 C1/C6).
3. SERVING: certified -> the winning command list respun UNSALTED on
   the live stream, with certified one-hot targets LATCHED to the
   un-diverged prefix only (round-3 C1/C7/C9: a per-slot equality
   re-armed targets after divergence on states certification never
   graded). Abstained -> the policy plays its own turn sampled
   DIRECTLY on the live stream (round-3 C5: replaying the salted
   incumbent's commands served choices premised on dice that never
   happened).

Budget: per SIDE-TURN, held in the policy's ledger (survives
bounces); a side-turn already at cap skips the tournament entirely
(round-3 C8). Within a tournament, certification + respine costs
are RESERVED up front -- selection charges against (cap - reserve),
so it can never starve the evidence stage (round-3 C0). The
per-half-turn action estimate driving the cost model is an EMA of
measured projection half-turns; cold start falls back to the
incumbent's own length for SELECTION sizing (safe: the reserve is
priced separately, at the conservative bound on cold start and at a
CERT_RESERVE_PAD-padded EMA after), so an optimistic estimate
shrinks selection, never certification.

Learning: certified turns record one-hot targets at
policy_weight=beta; everything else is value-only. finalize_game
renormalizes certified states' policy mass per side toward beta/2
per game-side (exact below the factor clamp 1/beta_max; when the
clamp binds -- sparse certification -- the realized mass is
beta*n_targets*f_cap/(2*n_states), reported via
pt_renorm_factor_mean; round-3 C10 / round-6 C5).
Beta starvation tripwire DISARMED for the first iteration: log,
calibrate, then arm (user ruling 2026-08-26).

Known limitations (accepted, measured at step 1): forwards
un-batched (the pool server batches across actors); spool-worker
pt_* telemetry not aggregated (loud launch warning); selection
retains one-realization grading bias (certification is the
counter).

Pre-registered v1 constants (derivation path = the branch-audit
instrument): margin_band, beta_max, margin_ref, cert_depth,
cert_redraws.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wesnoth_ai.classes import GameState, state_key  # noqa: E402
from tools.mcts import MCTSConfig  # noqa: E402
from tools.mcts_policy import MCTSPolicy, _PendingMCTSState  # noqa: E402
from tools.turn_search import (  # noqa: E402
    SpineStep, _match_action, _sample_prior_idx, boundary_value,
    forward_state, record_spine,
)

log = logging.getLogger("plan_tournament")

# Bound on the per-drain beta log (review C13 round 2).
BETA_LOG_CAP = 4096
# EMA smoothing for the measured per-half-turn action estimate.
HALF_EST_EMA = 0.9
# Every key drain_tournament_stats can emit (round-12 C4: the CSV
# parity test scraped literals and was blind to f-string keys).
PT_DRAIN_KEYS = frozenset({
    "pt_tournaments", "pt_arm_prefix_frac", "pt_cert_rate",
    "pt_cert_attempt_rate", "pt_cert_replicates_mean",
    "pt_cert_starved_rate", "pt_renorm_factor_mean",
    "pt_abstain_events_per_turn", "pt_beta_mean", "pt_beta_p50",
    "pt_beta_p90", "pt_forwards_per_turn", "pt_challengers_per_tournament",
    "pt_replans_per_turn", "pt_grades_per_tournament", "pt_half_est",
    "pt_margin_mean", "pt_cert_margin_mean",
    "pt_cert_rate_f0", "pt_cert_rate_f1", "pt_cert_rate_f2p",
    "pt_cert_attempt_frac_f0", "pt_cert_attempt_frac_f1",
    "pt_cert_attempt_frac_f2p",
    "pt_cert_len_delta_mean", "pt_cert_shorten_rate",
    "pt_half_cap_hit_rate",
})

# n-aware critical factors for certification acceptance: t_{n-1}
# quantiles at 1 - 0.0918, holding alpha at the n=3 legacy level
# (mean > 2*sd/sqrt(3) <=> P(t_2 > 2) = 0.0918) for every replicate
# count. Derivation: docs/design_constants.md.
_T_CRIT = {2: 3.37, 3: 2.0, 4: 1.72, 5: 1.61}


@dataclass
class TournamentConfig:
    n_challengers:      int = 6      # ceiling; sized down to budget
    depths:             Tuple[int, ...] = (1, 3)   # selection rounds
    redraws:            int = 1      # selection draws/candidate/round
    # Hard per-SIDE-TURN cap. 900 funds the full (1,3) selection
    # schedule + the tight certification reserve at est<=12 through
    # L<=17, and keeps every longer/costlier regime on single-depth
    # selection with FULL certification (no abstention cliff through
    # L=36; verified round-20/21 -- at est=17 the reserve saturates
    # at the hard bound and depth-3 selection does not fund, an
    # accepted trade). The equal-compute step-1 match sets this knob
    # explicitly and pins compute by MEASURED forwards.
    budget_forwards:    int = 900
    # Certification band: pre-registered v1 value = 2 C51 atoms
    # (2 * 2/(51-1) = 0.08 on the [-1,1] value scale). Replaced by
    # the branch-audit shrinkage curve once measured. Applied via
    # accept_rule: mean > max(_T_CRIT[n]*sd/sqrt(n), band).
    margin_band:        float = 0.08
    beta_max:           float = 0.25
    margin_ref:         float = 0.32  # margin where beta saturates
    cert_depth:         int = 3      # FIXED certification depth (odd)
    cert_redraws:       int = 3      # certification replicates
    project_max_actions: int = 20    # per projected half-turn
    min_challengers:    int = 2      # sizing floor
    max_spine:          int = 40


@dataclass
class _TPlan:
    side:          int
    turn_no:       int
    decision_step: int
    commands:      List[Dict]
    pre_keys:      List[str]
    targets:       List[List[Tuple]]   # one-hot tuples ([] = no target)
    beta:          float               # 0.0 on abstained turns
    certified:     bool
    cursor:        int = 0

    @property
    def exhausted(self) -> bool:
        return self.cursor >= len(self.commands)


def _odd_depths(raw, fallback: Tuple[int, ...],
                what: str) -> Tuple[int, ...]:
    """Enforce the own-frame invariant (review C4 round 2): depths
    must be positive and ODD -- an even depth reads the boundary in
    the opponent's fogged frame (the leg-4 blindness) -- and are
    normalized ASCENDING: sequential halving escalates fidelity, so
    a descending list would grade the full pool at the deepest
    depth first (round-18 C2/C3)."""
    kept = tuple(sorted(d for d in raw if d > 0 and d % 2 == 1))
    dropped = tuple(d for d in raw if d not in kept)
    if dropped:
        log.error(f"{what}: dropping non-odd/non-positive depths "
                  f"{dropped} (own-frame invariant)")
    if not kept:
        log.error(f"{what}: no valid depths given; using {fallback}")
        return fallback
    return kept


# The knob surface, single source of truth (round-10 C2: eval's knob
# list and the pt_config provenance dict were hand-maintained
# parallel copies). CLI flag name = "--pt-" + key.replace("_", "-").
PT_KNOB_KEYS = ("challengers", "depths", "redraws", "cert_depth",
                "cert_redraws", "budget_forwards", "margin_band",
                "beta_max", "margin_ref")


def pt_knobs_dict(cfg: TournamentConfig) -> Dict:
    """Provenance dict for eval result files: the full effective
    knob set, JSON-round-trip stable."""
    return {
        "n_challengers": cfg.n_challengers,
        "depths": list(cfg.depths),
        "redraws": cfg.redraws,
        "cert_depth": cfg.cert_depth,
        "cert_redraws": cfg.cert_redraws,
        "budget_forwards": cfg.budget_forwards,
        "margin_band": cfg.margin_band,
        "beta_max": cfg.beta_max,
        "margin_ref": cfg.margin_ref,
    }


def config_from_args(args) -> Optional[TournamentConfig]:
    """TournamentConfig from CLI args, or None when --plan-tournament
    is off (mirrors turn_search.config_from_args)."""
    if not getattr(args, "plan_tournament", False):
        return None
    cfg = TournamentConfig()
    n_req = int(getattr(args, "pt_challengers", cfg.n_challengers))
    cfg.n_challengers = max(cfg.min_challengers, n_req)
    if n_req < cfg.min_challengers:
        log.error(f"--pt-challengers {n_req} raised to "
                  f"{cfg.min_challengers} (round-11 C1: a"
                  f" non-positive pool silently disables the "
                  f"procedure with healthy-looking telemetry).")
    d = getattr(args, "pt_depths", None)
    if d:
        cfg.depths = _odd_depths(
            tuple(int(x) for x in str(d).split(",")),
            TournamentConfig.depths, "--pt-depths")
    rd_req = int(getattr(args, "pt_redraws", cfg.redraws))
    cfg.redraws = max(1, rd_req)
    if rd_req < 1:
        log.error(f"--pt-redraws {rd_req} raised to 1 (round-11 "
                  f"C1: zero redraws grades nothing and certifies "
                  f"nothing while the launch banner looks healthy).")
    cfg.budget_forwards = int(getattr(args, "pt_budget_forwards",
                                      cfg.budget_forwards))
    cfg.margin_band = float(getattr(args, "pt_margin_band",
                                    cfg.margin_band))
    cfg.beta_max = float(getattr(args, "pt_beta_max", cfg.beta_max))
    cfg.margin_ref = float(getattr(args, "pt_margin_ref",
                                   cfg.margin_ref))
    cfg.cert_depth = _odd_depths(
        (int(getattr(args, "pt_cert_depth", cfg.cert_depth)),),
        (TournamentConfig.cert_depth,), "--pt-cert-depth")[0]
    reps_req = int(getattr(args, "pt_cert_redraws", cfg.cert_redraws))
    reps_max = max(_T_CRIT)
    cfg.cert_redraws = min(max(2, reps_req), reps_max)
    if reps_req < 2:
        log.error(f"--pt-cert-redraws {reps_req} raised to 2 (the "
                  f"accept_rule minimum; round-10 C3).")
    if reps_req > reps_max:
        log.error(f"--pt-cert-redraws {reps_req} clamped to "
                  f"{reps_max}: the n-aware critical-factor table "
                  f"(_T_CRIT) covers n<= {reps_max}; extend the "
                  f"table before raising the knob (round-9 C0).")
    return cfg


def beta_from_margin(margin: float, cfg: TournamentConfig) -> float:
    """Conservative step size from the CERTIFIED mean margin: 0 below
    the band, linear above, saturating at beta_max by margin_ref.
    Placeholder until the branch audit supplies E[delta | margin]."""
    if margin <= cfg.margin_band:
        return 0.0
    span = max(cfg.margin_ref - cfg.margin_band, 1e-9)
    return cfg.beta_max * min(1.0, (margin - cfg.margin_band) / span)


def one_hot_target(step: SpineStep) -> List[Tuple]:
    a = step.legal[step.action_idx]
    return [(a.actor_idx, a.target_idx, a.weapon_idx, 1.0, a.type_idx)]


def grade_cost(depth: int, per_half: int) -> int:
    """Predicted forwards for ONE paired projection grade at `depth`
    with an estimated `per_half` actions per projected half-turn."""
    return 2 * (depth * per_half + 1)


# Safety factor on the certification reserve's half-turn estimate: the
# reserve must survive a CERT_RESERVE_PAD-fold under-estimate by the
# EMA before certification can be starved by selection spend (the EMA
# itself corrects within a few tournaments at HALF_EST_EMA=0.9).
# Rationale + derivation note: docs/design_constants.md. The round-6
# hard bound (project_max_actions) over-reserved ~250 forwards at
# realistic half-turns and pushed fundable long side-turns (L>=30)
# into value-only abstention with most of the budget unspent
# (round-7 C0).
CERT_RESERVE_PAD = 2.0


def cert_ph(cfg: TournamentConfig,
            per_half: Optional[int] = None) -> int:
    """Per-half-turn action cap used by the certification reserve,
    certification's own projections, AND (round-25 C3) selection's
    projections + the _min_sel floor (round-20 C5: a reserve priced
    at a worst case the spend never hits is idle budget; pricing
    and capping with the same number makes the guarantee real by
    construction)."""
    if per_half is None:
        return cfg.project_max_actions
    return int(min(cfg.project_max_actions,
                   max(1, round(CERT_RESERVE_PAD * per_half))))


def cert_reserve(spine_len: int, cfg: TournamentConfig,
                 per_half: Optional[int] = None,
                 cert_redraws: Optional[int] = None) -> int:
    """Forwards reserved for certification + the serving respine:
    per replicate, both command lists re-executed via record_spine
    (~spine_len forwards each) and projected at cert_depth; plus one
    unsalted respine. Priced at a PADDED per-half estimate (never
    below the EMA times CERT_RESERVE_PAD, capped at the bound) so an
    optimistic estimate shrinks SELECTION, never the evidence stage
    (round-6 C3/C4) -- without the round-6 hard bound's long-turn
    abstention cliff (round-7 C0). per_half=None (cold start) keeps
    the conservative bound."""
    ph = cert_ph(cfg, per_half)
    reps = cfg.cert_redraws if cert_redraws is None else cert_redraws
    per_rep = 2 * (spine_len + 1) + grade_cost(cfg.cert_depth, ph)
    return reps * per_rep + (spine_len + 1)


def cert_reserve_excess(lc_max: int, spine_len: int,
                        cfg: TournamentConfig,
                        cert_reps: int) -> int:
    """Reserve top-up once the challenger pool exists: cert_reserve
    prices BOTH certification arms at the incumbent's spine length,
    but certify re-executes the CHALLENGER'S own command list
    (round-23 C2: the shortfall dropped replicates exactly on
    turn-LENGTHENING plans, and the n-aware accept_rule then taxed
    them with the harsher small-n critical factor -- a certification
    bias along the leg-3 passivity axis). One excess per replicate
    plus one for the serving respine; challenger re-execution is
    capped at max_spine by record_spine, so so is the price."""
    lc = min(lc_max, cfg.max_spine)
    return max(0, lc - (spine_len + 1)) * (cert_reps + 1)


def predicted_demand(spine_len: int, per_half: int, n_chal: int,
                     depths: Tuple[int, ...],
                     cfg: TournamentConfig,
                     cert_ph: Optional[int] = None,
                     cert_redraws: Optional[int] = None) -> int:
    """Forward demand of a full tournament at this schedule.
    Half-turn costs use `per_half` -- an EMA of MEASURED projection
    half-turns, not the incumbent's own length (round-3 C0: the
    incumbent draw is a bad proxy for reply length)."""
    d = n_chal * (spine_len + 1)                    # challenger spines
    pool = n_chal
    for dep in depths:                              # selection rounds
        # Candidates + ONE shared incumbent projection per redraw.
        d += (pool + 1) * cfg.redraws * (
            dep * per_half + 1)
        pool = max(1, pool // 2)
    d += cert_reserve(spine_len, cfg, per_half=cert_ph,
                      cert_redraws=cert_redraws)
    return d


def size_schedule(spine_len: int, per_half: int, budget_left: int,
                  cfg: TournamentConfig, cold_start: bool = False,
                  ) -> Tuple[int, Tuple[int, ...], int]:
    """Trim (n_challengers, depths) until predicted_demand fits
    `budget_left`. Preference: drop the deepest SELECTION round,
    then challengers to the floor. Returns (0, ()) when even the
    floor is unaffordable -- the caller abstains honestly instead of
    running a schedule it cannot fund (round-3 C13)."""
    cert_ph = None if cold_start else per_half
    n, deps = cfg.n_challengers, tuple(cfg.depths)
    while predicted_demand(spine_len, per_half, n, deps, cfg,
                           cert_ph=cert_ph) > budget_left:
        if len(deps) > 1:
            deps = deps[:-1]
        elif n > cfg.min_challengers:
            n -= 1
        else:
            # Floor unaffordable -> honest abstention. A planned
            # 2-replicate certification was tried (round 7) and
            # measured strictly worse on BOTH axes (round-8 C0:
            # n=2 alpha 0.148 vs n=3 0.092 band-free, power 0.66 vs
            # 0.73) -- degraded evidence is worse than none.
            return 0, (), 0
    return n, deps, cfg.cert_redraws


def launch_echo_schedule(cfg: TournamentConfig, k: int = 12,
                         ) -> Tuple[int, Tuple[int, ...], int, int]:
    """(n_challengers, depths, demand, per_half) the budget funds at
    spine length `k` under the COLD-START estimate -- the exact
    arithmetic run_tournament applies on its first side-turn. Called
    by the launch banner AND by tests (round-4 C0: an un-executed
    echo callsite rotted through a signature change)."""
    per_half = min(cfg.project_max_actions, k + 1)
    n, deps, _reps = size_schedule(k, per_half,
                                   max(0, cfg.budget_forwards - k),
                                   cfg, cold_start=True)
    demand = (predicted_demand(k, per_half, n, deps, cfg,
                               cert_redraws=_reps) if n else 0)
    return n, deps, demand, per_half


class _Budget:
    """Forward counter with a hard cap and a certification reserve:
    charges with `reserved=True` may use the full cap; ordinary
    (selection) charges stop at cap - reserve (round-3 C0)."""

    def __init__(self, cap: int, already_used: int = 0):
        self.cap = int(cap)
        self.used = int(already_used)
        self.reserve = 0

    def charge(self, n: int = 1, reserved: bool = False) -> bool:
        self.used += n
        limit = self.cap if reserved else self.cap - self.reserve
        return self.used <= limit

    @property
    def exhausted(self) -> bool:
        return self.used > self.cap


def _salted(sim, salt: str):
    """Fork `sim` onto a fresh dice stream -- GRADING forks only.
    Serving spines are deliberately unsalted."""
    f = sim.fork()
    f._seed_salt = salt
    return f


def _grading_boundary(sim, side: int):
    """Normalize a selection spine's boundary for grading: if the
    walk broke without end_turn, close the turn on the GRADING fork
    so every paired read sits at the same parity (round-2 C10)."""
    if sim.done or sim.gs.global_info.current_side != side:
        return sim
    f = sim.fork()
    try:
        f.step({"type": "end_turn"})
    except Exception:  # noqa: BLE001
        return sim
    return f


def _project_counted(policy, sim, side: int, decision_step: int,
                     half_turns: int, max_actions: int,
                     rng: np.random.Generator, budget: _Budget,
                     salt: str, reserved: bool = False,
                     half_obs: Optional[List[int]] = None,
                     ) -> Optional[float]:
    """turn_search.project_value with exact forward accounting and a
    salted fork. `half_obs` collects per-half-turn action counts for
    the cost-model EMA (round-3 C0)."""
    if sim.done:
        return boundary_value(policy, sim, side, decision_step)
    r = _salted(sim, salt)
    for _ in range(half_turns):
        if r.done:
            break
        mover = r.gs.global_info.current_side
        k = 0
        while (not r.done and r.gs.global_info.current_side == mover
               and k < max_actions):
            if not budget.charge(reserved=reserved):
                return None
            _, output, legal = forward_state(policy, r.gs,
                                             decision_step)
            if not legal:
                break
            try:
                r.step(legal[_sample_prior_idx(legal, rng)].action)
            except Exception:  # noqa: BLE001 -- search must not die
                break
            k += 1
        if half_obs is not None:
            # (count, censored-at-cap): the cap is DERIVED from the
            # EMA these observations feed, so an uncensored-mean
            # update is a self-reinforcing ratchet (round-26 C1:
            # measured absorbing at cap 4 from a true mean 3.05,
            # and at cap 2 in the K~2 passivity regime).
            half_obs.append((k, k >= max_actions))
        if not r.done and r.gs.global_info.current_side == mover:
            try:
                r.step({"type": "end_turn"})
            except Exception:  # noqa: BLE001
                break
    if not budget.charge(reserved=reserved):
        return None
    return boundary_value(policy, r, side, decision_step)


def _perturb_action(step: SpineStep,
                    rng: np.random.Generator) -> Optional[Dict]:
    """Sample an alternative action at a spine slot, prior-weighted
    over everything except the incumbent's choice."""
    if len(step.legal) < 2:
        return None
    p = np.array([max(a.prior, 0.0) for a in step.legal],
                 dtype=np.float64)
    p[step.action_idx] = 0.0
    s = p.sum()
    if s <= 0:
        idxs = [i for i in range(len(step.legal))
                if i != step.action_idx]
        return step.legal[int(rng.choice(idxs))].action
    return step.legal[int(rng.choice(len(p), p=p / s))].action


def accept_rule(deltas, band: float) -> Tuple[bool, float, float]:
    """Certification acceptance: mean > max(_T_CRIT[n]*sd/sqrt(n),
    band). The n-aware critical factor holds the band-free
    false-accept rate at the n=3 legacy level (P(t_2 > 2) = 0.0918)
    for every replicate count n in the table -- config_from_args
    clamps cert_redraws to the tabled range, so all reachable n are
    covered (round-8 C0, round-9 C0). Derivation:
    docs/design_constants.md."""
    d = np.array(deltas, dtype=np.float64)
    n_d = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    crit = _T_CRIT[min(n_d, max(_T_CRIT))]
    thr = max(crit * sd / np.sqrt(n_d), band)
    return mean > thr, mean, thr


def certify(policy, sim, side: int, decision_step: int,
            cand_cmds: List[Dict], inc_cmds: List[Dict],
            cfg: TournamentConfig, rng: np.random.Generator,
            budget: _Budget, salt_ns: str,
            half_obs: Optional[List[int]] = None,
            max_redraws: Optional[int] = None,
            max_actions: Optional[int] = None,
            ) -> Tuple[float, Optional[float], int]:
    """Stage-2 certification. Per replicate, BOTH command lists are
    re-executed on fresh streams via record_spine(actions=...) --
    the same divergence-repair semantics serving uses, so the graded
    object is the served object (round-3 C3) -- then projected at
    the fixed cert_depth. Accepts by the n-aware accept_rule over the
    COMPLETED replicates (>=2; a budget death mid-stage keeps the
    evidence in hand, round-3 C2)."""
    deltas = []
    ma = cfg.project_max_actions if max_actions is None else max_actions
    reps = cfg.cert_redraws if max_redraws is None else max_redraws
    for r in range(reps):
        s = f"{salt_ns}:cert{r}"
        pair = []
        for tag, cmds in (("c", cand_cmds), ("i", inc_cmds)):
            # SHARED salt for both arms of a replicate (round-6 C0:
            # per-arm salts made the "paired" test unpaired -- a
            # shared command prefix must replay on identical dice so
            # its variance cancels out of the delta; the pairing
            # turn_search's own stage 2 uses). Projection salts
            # below stay per-arm: post-boundary streams are not
            # couplable under the counter-keyed scheme.
            steps, bsim = record_spine(
                policy, _salted(sim, f"{s}:spine"), side,
                decision_step, rng, max_spine=cfg.max_spine,
                actions=list(cmds))
            if not budget.charge(len(steps), reserved=True):
                pair = None
                break
            v = _project_counted(
                policy, _grading_boundary(bsim, side), side,
                decision_step, cfg.cert_depth,
                ma, rng, budget,
                salt=f"{s}:p{tag}", reserved=True,
                half_obs=half_obs)
            if v is None:
                pair = None
                break
            pair.append(v)
        if pair is None:
            break
        deltas.append(pair[0] - pair[1])
    if len(deltas) < 2:
        return 0.0, (deltas[0] if deltas else None), len(deltas)
    ok, mean, _thr = accept_rule(deltas, cfg.margin_band)
    if not ok:
        return 0.0, mean, len(deltas)
    return beta_from_margin(mean, cfg), mean, len(deltas)


def run_tournament(policy, sim, side: int, decision_step: int,
                   cfg: TournamentConfig, rng: np.random.Generator,
                   salt_ns: str, budget_used: int = 0,
                   half_est: Optional[float] = None,
                   ) -> Tuple[_TPlan, Dict]:
    """Selection + certification + serving from `sim` (not mutated).
    `budget_used` = this side-turn's prior spend; `half_est` = the
    policy's EMA of projected half-turn lengths (None -> cold start
    on the incumbent's own length). Returns (plan, stats)."""
    budget = _Budget(cfg.budget_forwards, already_used=budget_used)
    stats = {"forwards": 0, "n_challengers": 0, "certified": 0,
             "abstained_budget": 0, "margin": None, "beta": 0.0,
             "graded": 0, "cert_attempts": 0, "half_obs": [],
             "arm_prefix": 0, "cert_replicates": 0,
             "cert_starved": 0, "fight_bucket": None,
             "cert_len_delta": None}
    turn_no = int(sim.gs.global_info.turn_number)

    def _live_spine_plan(cmds: Optional[List[Dict]], beta: float,
                         certified: bool) -> _TPlan:
        """Serve on the LIVE stream: replay `cmds` (certified) or
        sample the policy's own turn (abstained, cmds=None --
        round-3 C5). Certified targets LATCH off at the first
        divergence (round-3 C1/C7/C9)."""
        steps, _b = record_spine(
            policy, sim, side, decision_step, rng,
            max_spine=cfg.max_spine,
            actions=(list(cmds) if cmds is not None else None))
        budget.charge(len(steps), reserved=True)
        if not steps:
            return _TPlan(side=side, turn_no=turn_no,
                          decision_step=decision_step,
                          commands=[{"type": "end_turn"}],
                          pre_keys=[""], targets=[[]],
                          beta=0.0, certified=False)
        targets: List[List[Tuple]] = []
        on_plan = certified
        for i, st in enumerate(steps):
            if on_plan and not (cmds is not None and i < len(cmds)
                                and st.action == cmds[i]):
                on_plan = False          # latched: never re-arms
            targets.append(one_hot_target(st) if on_plan else [])
        return _TPlan(side=side, turn_no=turn_no,
                      decision_step=decision_step,
                      commands=[s.action for s in steps],
                      pre_keys=[state_key(s.pre_fork.gs)
                                for s in steps],
                      targets=targets,
                      beta=beta if certified else 0.0,
                      certified=certified)

    # Incumbent SELECTION spine (salted grading material).
    inc_steps, inc_boundary = record_spine(
        policy, _salted(sim, f"{salt_ns}:inc"), side, decision_step,
        rng, max_spine=cfg.max_spine)
    budget.charge(len(inc_steps))
    if not inc_steps:
        plan = _live_spine_plan(None, 0.0, certified=False)
        stats["forwards"] = budget.used - budget_used
        return plan, stats

    L = len(inc_steps)
    # The MEASURED EMA drives the cost model once it exists; the
    # incumbent's own length is only the cold-start fallback (an
    # unconditional max(L+1, ...) floor made the EMA inert on long
    # turns -- round-5 C1). The estimate prices SELECTION only; the
    # reserve is priced separately (conservative bound on cold
    # start, CERT_RESERVE_PAD-padded EMA after -- round-6 C3/C4,
    # round-7 C0), so an optimistic estimate shrinks selection,
    # never certification.
    per_half = int(min(cfg.project_max_actions,
                       max(1.0, half_est
                           if half_est is not None
                           else float(L + 1))))
    cold = half_est is None
    # Selection's per-half projection cap = certification's
    # (round-25 C3); cold start keeps the unconditional bound.
    sel_ph = cert_ph(cfg, None if cold else per_half)
    budget.reserve = cert_reserve(
        L, cfg, per_half=(None if cold else per_half))
    inc_cmds = [s.action for s in inc_steps]
    inc_grade = _grading_boundary(inc_boundary, side)

    n_chal, depths, cert_reps = size_schedule(
        L, per_half, max(0, budget.cap - budget.used), cfg,
        cold_start=cold)
    if n_chal == 0:
        # Even the floor schedule is unaffordable: abstain honestly
        # (round-3 C13) -- the turn still plays, on the live stream.
        stats["abstained_budget"] = 1
        plan = _live_spine_plan(None, 0.0, certified=False)
        stats["forwards"] = budget.used - budget_used
        return plan, stats

    # Challengers: ONE arm per tournament (round-4 C7: prefix-coupled
    # and fully-independent candidates have different margin variance;
    # an argmax over the mixed pool favors the noisier arm). Prefix
    # perturbations fork the incumbent's OWN pre-state (':inc'
    # stream) so the realized prefix is shared exactly (round-3 C6).
    prefix_arm = bool(L > 1 and rng.random() < 0.5)
    stats["arm_prefix"] = int(prefix_arm)
    challengers = []
    for c in range(n_chal):
        if budget.used > budget.cap - budget.reserve:
            break
        if prefix_arm:
            i = int(rng.integers(L))
            st = inc_steps[i]
            alt = _perturb_action(st, rng)
            if alt is None:
                continue
            f = st.pre_fork.fork()
            try:
                f.step(alt)
            except Exception:  # noqa: BLE001
                continue
            if getattr(f, "last_step_rejected", False):
                continue
            budget.charge(1)
            suffix, ch_boundary = record_spine(
                policy, f, side, decision_step, rng,
                max_spine=cfg.max_spine)
            budget.charge(len(suffix))
            cmds = (inc_cmds[:i] + [alt]
                    + [s.action for s in suffix])
        else:
            ch_steps, ch_boundary = record_spine(
                policy, _salted(sim, f"{salt_ns}:ch{c}"), side,
                decision_step, rng, max_spine=cfg.max_spine)
            budget.charge(len(ch_steps))
            cmds = [s.action for s in ch_steps]
        if cmds and cmds != inc_cmds:
            challengers.append(
                {"cmds": cmds,
                 "grade": _grading_boundary(ch_boundary, side),
                 "margin": -np.inf, "depth_graded": 0})
    stats["n_challengers"] = len(challengers)
    if challengers:
        # Priced BEFORE selection spends, so the cap-minus-reserve
        # checks below throttle selection instead of letting it eat
        # the certification funding of a long challenger -- but
        # clamped so selection keeps funding for at least ONE
        # complete graded comparison at the shallowest depth
        # (round-24 C5: unclamped, a large excess starved selection
        # entirely, best stayed None and certify never ran on
        # exactly the long-challenger turns the bump protects).
        # A clamped-away remainder surfaces as cert_starved.
        _want = cert_reserve_excess(
            max(len(c["cmds"]) for c in challengers), L, cfg,
            cert_reps)
        _min_sel = 2 * cfg.redraws * (depths[0] * sel_ph + 1)
        _room = max(0, budget.cap - budget.used - _min_sel
                    - budget.reserve)
        budget.reserve += min(_want, _room)

    # SELECTION halving: incumbent projected once per (round,
    # redraw) and shared across candidates (round-3 C4). Selection
    # projections are capped at sel_ph -- the SAME number the
    # _min_sel floor prices (round-25 C3: floor at per_half with
    # projections free to run to project_max_actions left the
    # round-24 guarantee unreal; "price and cap with the same
    # number" is the round-20 C5 pattern certification already
    # uses, and it keeps selection and certification margins on
    # the same projection cap).
    pool = challengers
    for di, d in enumerate(depths):
        if not pool or budget.used > budget.cap - budget.reserve:
            break
        inc_vals = []
        for rd in range(cfg.redraws):
            v = _project_counted(
                policy, inc_grade, side, decision_step, d,
                sel_ph, rng, budget,
                salt=f"{salt_ns}:ir{di}.{rd}",
                half_obs=stats["half_obs"])
            if v is None:
                break
            inc_vals.append(v)
        if len(inc_vals) < cfg.redraws:
            break
        cut = False
        for ci, cand in enumerate(pool):
            ms = []
            for rd in range(cfg.redraws):
                v = _project_counted(
                    policy, cand["grade"], side, decision_step, d,
                    sel_ph, rng, budget,
                    salt=f"{salt_ns}:g{di}.{ci}.{rd}",
                    half_obs=stats["half_obs"])
                if v is None:
                    cut = True
                    break
                ms.append(v - inc_vals[rd])
                stats["graded"] += 1
            if len(ms) == cfg.redraws:
                # Only a COMPLETE redraw set overwrites: a budget-
                # death cut must not erase a prior round's measured
                # margin (round-10 C0), and a PARTIAL set must not
                # compete in the deepest-graded argmax on fewer
                # draws than its peers (round-11 C0 -- the mixed-
                # variance bias; mirrors the incumbent guard).
                cand["margin"] = float(np.mean(ms))
                cand["depth_graded"] = d
            if cut:
                break
        pool.sort(key=lambda c: c["margin"], reverse=True)
        if cut:
            break
        pool = pool[:max(1, len(pool) // 2)]

    # Best = top margin among candidates graded at the DEEPEST
    # completed depth (margins from different depths are not
    # comparable evidence -- round-10 C0 follow-up).
    graded = [c for c in pool if np.isfinite(c["margin"])]
    best = None
    if graded:
        d_max = max(c["depth_graded"] for c in graded)
        best = max((c for c in graded
                    if c["depth_graded"] == d_max),
                   key=lambda c: c["margin"])

    # CERTIFICATION (the only path to beta > 0; runs on the reserve).
    beta, mean_margin = 0.0, None
    if best is not None and not budget.exhausted:
        # Fight bucket of the GRADED plan, recorded only when
        # certification is actually attempted, so n_f* sums to
        # cert_attempts and each rate is conditional-on-attempt
        # (round-12 C1: unconditional bucketing polluted the
        # denominators with never-attempted turns).
        n_atk = sum(1 for c in best["cmds"]
                    if c.get("type") == "attack")
        stats["fight_bucket"] = ("f0" if n_atk == 0
                                 else ("f1" if n_atk == 1 else "f2p"))
        stats["cert_attempts"] = 1
        beta, mean_margin, n_reps = certify(
            policy, sim, side, decision_step, best["cmds"], inc_cmds,
            cfg, rng, budget, salt_ns, half_obs=stats["half_obs"],
            max_redraws=cert_reps,
            max_actions=cert_ph(cfg, None if cold else per_half))
        stats["cert_replicates"] = n_reps
        if beta > 0.0:
            # Turn-shortening direction of accepted swaps -- the
            # leg-3 passivity leading indicator (round-12 C2).
            stats["cert_len_delta"] = (len(best["cmds"])
                                       - len(inc_cmds))
        # Any shortfall vs the planned count is reserve starvation
        # (the one failure CERT_RESERVE_PAD exists to prevent);
        # planned reduction was removed in round 8, so the signal is
        # unambiguous (round-8 C1).
        stats["cert_starved"] = int(n_reps < cert_reps)

    if budget.exhausted and beta <= 0.0:
        stats["abstained_budget"] = 1

    if beta > 0.0:
        plan = _live_spine_plan(best["cmds"], beta, certified=True)
        stats["certified"] = 1
    else:
        plan = _live_spine_plan(None, 0.0, certified=False)
    stats["margin"] = mean_margin
    stats["beta"] = plan.beta
    stats["forwards"] = budget.used - budget_used
    return plan, stats


class PlanTournamentPolicy(MCTSPolicy):
    """MCTSPolicy with the decision procedure replaced by the
    incumbent-anchored plan tournament. Same serving contract as
    TurnCommitPolicy (deepcopy snapshot + live sim required)."""

    def __init__(self, base, mcts_config: Optional[MCTSConfig] = None,
                 *args, tournament_config: Optional[TournamentConfig]
                 = None, **kwargs):
        super().__init__(base, mcts_config, *args, **kwargs)
        self._t_cfg = tournament_config or TournamentConfig()
        self._t_plans: Dict[str, _TPlan] = {}
        # Per-side-turn forward ledger, OUTSIDE the plan so it
        # survives drop_last_pending (round-2 C2/C8/C12).
        self._t_spend: Dict[str, Tuple[int, int, int]] = {}
        self._t_half_ema: Optional[float] = None
        self._t_seq = 0
        self._t_rng = np.random.default_rng()
        self._t_acc: Dict[str, float] = {}
        self._t_betas: List[float] = []

    # -- side-turn spend ledger (self-locking: plain Lock, callers
    # must NOT hold self._lock -- the round-2 fix deadlocked here) --

    def _turn_spent(self, label: str, side: int, turn_no: int) -> int:
        with self._lock:
            e = self._t_spend.get(label)
            if e is None or e[0] != side or e[1] != turn_no:
                return 0
            return e[2]

    def _turn_charge(self, label: str, side: int, turn_no: int,
                     n: int) -> None:
        with self._lock:
            e = self._t_spend.get(label)
            if e is None or e[0] != side or e[1] != turn_no:
                # New side-turn: count it (pt_forwards_per_turn
                # divides by SIDE-TURNS, not tournaments -- C8).
                self._t_acc["side_turns"] = (
                    self._t_acc.get("side_turns", 0) + 1)
                used = n
            else:
                used = e[2] + n
            self._t_spend[label] = (side, turn_no, used)
            self._t_acc["forwards"] = (
                self._t_acc.get("forwards", 0) + n)

    # -- serving -------------------------------------------------------

    def select_action(self, game_state: GameState, *,
                      game_label: str = "default", sim=None) -> Dict:
        if sim is None:
            raise RuntimeError(
                "PlanTournamentPolicy.select_action requires `sim=`.")
        if game_state is sim.gs:
            raise ValueError(
                "PlanTournamentPolicy.select_action was passed the "
                "LIVE sim.gs; pass a deepcopy snapshot.")
        with self._base._lock:
            ds_call = self._base._decision_step
            self._base._decision_step += 1
        side = sim.gs.global_info.current_side
        turn_no = int(sim.gs.global_info.turn_number)
        live_key = state_key(sim.gs)
        with self._lock:
            plan = self._t_plans.get(game_label)
            self._t_seq += 1
            seq = self._t_seq
            half_est = self._t_half_ema

        fresh = (plan is None or plan.side != side
                 or plan.turn_no != turn_no or plan.exhausted)
        on_branch = (not fresh
                     and plan.pre_keys[plan.cursor] == live_key)
        replan = False
        if not fresh and not on_branch:
            # Divergence: replan ONLY on the legality trigger; the
            # probe forward is charged + counted (round-2 C3).
            self._turn_charge(game_label, side, turn_no, 1)
            _, _out, legal = forward_state(self._base, sim.gs, ds_call)
            if _match_action(legal, plan.commands[plan.cursor]) is None:
                replan = True

        spent = self._turn_spent(game_label, side, turn_no)
        if fresh and spent >= self._t_cfg.budget_forwards:
            # Side-turn already at cap (bounce path): no fresh
            # tournament -- serve the policy's own live turn,
            # value-only (round-3 C8).
            fresh = False
            replan = True
            self._t_bump("abstained_budget", 1)

        if fresh:
            plan, st = run_tournament(
                self._base, sim, side, ds_call, self._t_cfg,
                self._t_rng, salt_ns=f"pt:{game_label}:{seq}",
                budget_used=spent, half_est=half_est)
            self._turn_charge(game_label, side, turn_no,
                              st["forwards"])
            self._t_record(st)
            on_branch = plan.pre_keys[plan.cursor] == live_key
        elif replan:
            # Mid-turn recovery / at-cap serve: fresh live policy
            # completion, value-only, tournament-free (round-2 C9).
            steps, _b = record_spine(
                self._base, sim, side, ds_call, self._t_rng,
                max_spine=self._t_cfg.max_spine)
            self._turn_charge(game_label, side, turn_no, len(steps))
            self._t_bump("replans", 1)
            if steps:
                plan = _TPlan(
                    side=side, turn_no=turn_no, decision_step=ds_call,
                    commands=[s.action for s in steps],
                    pre_keys=[state_key(s.pre_fork.gs) for s in steps],
                    targets=[[] for _ in steps], beta=0.0,
                    certified=False)
                on_branch = True
            else:
                plan = _TPlan(
                    side=side, turn_no=turn_no, decision_step=ds_call,
                    commands=[{"type": "end_turn"}], pre_keys=[""],
                    targets=[[]], beta=0.0, certified=False)
                on_branch = False

        cmd = plan.commands[plan.cursor]
        # Distill ON-BRANCH only (round-1 U0/U1).
        target = plan.targets[plan.cursor] if on_branch else []
        plan.cursor += 1
        with self._lock:
            self._t_plans[game_label] = plan
            self._pending.setdefault(game_label, []).append(
                _PendingMCTSState(
                    gs=game_state, visit_counts=target, side=side,
                    decision_step=plan.decision_step,
                    policy_weight=plan.beta if target else 0.0))
            self._last_recorded[game_label] = True
        return cmd

    # -- plan lifecycle (bounce contract) ------------------------------

    def drop_last_pending(self, game_label: str) -> bool:
        """Bounce contract: undo the last decision AND discard the
        cached plan (the spend ledger survives)."""
        handled = super().drop_last_pending(game_label)
        with self._lock:
            self._t_plans.pop(game_label, None)
        return handled

    def drop_pending(self, game_label: str) -> None:
        super().drop_pending(game_label)
        with self._lock:
            self._t_plans.pop(game_label, None)
            self._t_spend.pop(game_label, None)

    def finalize_game(self, game_label: str, winner: int,
                      final_gs=None, midgame: bool = False) -> None:
        with self._lock:
            # Per-side policy-mass renormalization (round-3 C10):
            # game_weight = 1/(2*n_side_states) divides EVERY
            # recorded state, so a certified turn's policy mass
            # would scale with the game's value-only state count.
            # Scaling each target state by n_states/n_targets makes
            # one certified turn carry beta/2 per game-side
            # regardless of how many value-only states surround it
            # (the CPI step the spec pre-registered).
            states = self._pending.get(game_label, [])
            n_states: Dict[int, int] = {}
            n_targets: Dict[int, int] = {}
            for s in states:
                n_states[s.side] = n_states.get(s.side, 0) + 1
                if s.visit_counts:
                    n_targets[s.side] = n_targets.get(s.side, 0) + 1
            # The FACTOR is clamped at 1/beta_max, not the product
            # (round-5 C2: a product cap of 1.0 binds in the
            # ordinary sparse-certification regime and erases beta
            # sensitivity). weight = beta * factor <= beta/beta_max
            # <= 1, so the per-state step never exceeds full
            # strength AND stays proportional to the certified
            # margin. Below the clamp, one certified turn carries
            # beta/2 per game-side.
            f_cap = 1.0 / max(self._t_cfg.beta_max, 1e-9)
            from tools.mcts_policy import side_weight_divisor
            for s in states:
                if s.visit_counts and n_targets.get(s.side):
                    # The numerator is the divisor game_weight will
                    # ACTUALLY use -- midgame floor included -- so
                    # certified mass is beta/2 regardless of stub
                    # length (round-27 C4).
                    factor = min(f_cap, side_weight_divisor(
                        n_states[s.side], midgame)
                        / n_targets[s.side])
                    s.policy_weight = s.policy_weight * factor
                    self._t_acc["renorm_factor_sum"] = (
                        self._t_acc.get("renorm_factor_sum", 0.0)
                        + factor)
                    self._t_acc["renorm_n"] = (
                        self._t_acc.get("renorm_n", 0) + 1)
            self._t_plans.pop(game_label, None)
            self._t_spend.pop(game_label, None)
        super().finalize_game(game_label, winner, final_gs=final_gs,
                              midgame=midgame)

    # -- telemetry -----------------------------------------------------

    def _t_record(self, st: Dict) -> None:
        with self._lock:
            a = self._t_acc
            a["tournaments"] = a.get("tournaments", 0) + 1
            for k in ("certified", "abstained_budget", "graded",
                      "n_challengers", "cert_attempts", "arm_prefix",
                      "cert_replicates", "cert_starved"):
                a[k] = a.get(k, 0) + st[k]
            a["beta_sum"] = a.get("beta_sum", 0.0) + st["beta"]
            if len(self._t_betas) < BETA_LOG_CAP:
                self._t_betas.append(float(st["beta"]))
            if st["margin"] is not None:
                a["margin_n"] = a.get("margin_n", 0) + 1
                a["margin_sum"] = (a.get("margin_sum", 0.0)
                                   + st["margin"])
                if st["certified"]:
                    a["cert_margin_sum"] = (
                        a.get("cert_margin_sum", 0.0) + st["margin"])
            fb = st.get("fight_bucket")
            if fb is not None:
                a[f"n_{fb}"] = a.get(f"n_{fb}", 0) + 1
                if st["certified"]:
                    a[f"cert_{fb}"] = a.get(f"cert_{fb}", 0) + 1
            ld = st.get("cert_len_delta")
            if ld is not None:
                a["len_delta_sum"] = a.get("len_delta_sum", 0) + ld
                a["len_delta_n"] = a.get("len_delta_n", 0) + 1
                if ld < 0:
                    a["shorten_n"] = a.get("shorten_n", 0) + 1
            if st["half_obs"]:
                ks = [k for k, _ in st["half_obs"]]
                n_cens = sum(1 for _, c in st["half_obs"] if c)
                a["half_obs_n"] = a.get("half_obs_n", 0) + len(ks)
                a["half_cap_hits"] = (a.get("half_cap_hits", 0)
                                      + n_cens)
                if n_cens < len(ks):
                    # Right-censored mean (exponential-tail MLE:
                    # total count over uncensored draws) -- moves
                    # UP when the cap binds, so a shrunken cap is
                    # not an absorbing state (round-26 C1).
                    obs = float(sum(ks)) / float(len(ks) - n_cens)
                else:
                    # Every draw capped: no tail estimate exists;
                    # push past the cap so the next tournament
                    # measures with a larger one.
                    obs = float(max(ks)) + 1.0
                self._t_half_ema = (
                    obs if self._t_half_ema is None
                    else HALF_EST_EMA * self._t_half_ema
                    + (1.0 - HALF_EST_EMA) * obs)

    def _t_bump(self, key: str, v) -> None:
        with self._lock:
            self._t_acc[key] = self._t_acc.get(key, 0) + v

    def drain_distill_stats(self) -> Optional[Dict[str, float]]:
        """Tournament telemetry rides the distill drain. None when
        idle (round-1 C11: an always-truthy dict shadowed the
        actor-pool fallback)."""
        base = super().drain_distill_stats() or {}
        t = self.drain_tournament_stats()
        if t:
            base.update(t)
        return base or None

    def drain_tournament_stats(self) -> Dict[str, float]:
        """Per-iteration telemetry; {} when idle. All keys are RATES
        or means so the actor pool's unweighted mean-across-actors
        aggregation stays meaningful (round-3 C12); pt_tournaments
        is per-actor-per-drain under the pool, documented as such."""
        with self._lock:
            acc, self._t_acc = self._t_acc, {}
            betas, self._t_betas = self._t_betas, []
            half_ema = self._t_half_ema
        if not acc:
            return {}
        n = max(1.0, float(acc.get("tournaments", 0)))
        turns = max(1.0, float(acc.get("side_turns", 0)))
        out = {
            "pt_tournaments": acc.get("tournaments", 0),
            "pt_arm_prefix_frac": acc.get("arm_prefix", 0) / n,
            "pt_cert_rate": acc.get("certified", 0) / n,
            "pt_cert_attempt_rate": acc.get("cert_attempts", 0) / n,

            "pt_abstain_events_per_turn":
                acc.get("abstained_budget", 0) / turns,
            "pt_beta_mean": acc.get("beta_sum", 0.0) / n,
            "pt_forwards_per_turn": acc.get("forwards", 0) / turns,
            "pt_challengers_per_tournament":
                acc.get("n_challengers", 0) / n,
            "pt_replans_per_turn": acc.get("replans", 0) / turns,
            "pt_grades_per_tournament": acc.get("graded", 0) / n,
        }
        if half_ema is not None:
            out["pt_half_est"] = float(half_ema)
        # Certification split by fight count of the graded plan
        # (spec repair 6): rate conditional on attempt, WITH its
        # denominator (round-12 C3: a rate without n is bare); keys
        # omitted when the bucket is empty.
        ca_all = float(acc.get("cert_attempts", 0))
        if ca_all:
            for fb in ("f0", "f1", "f2p"):
                nb = float(acc.get(f"n_{fb}", 0))
                # The frac's denominator is ca_all, so an EMPTY
                # bucket is a measured 0.0, not a missing value --
                # omitting it made the pool mean inflate rare
                # buckets (round-14 C0). The rate's denominator is
                # nb; it stays omitted when empty.
                out[f"pt_cert_attempt_frac_{fb}"] = nb / ca_all
                if nb:
                    out[f"pt_cert_rate_{fb}"] = (
                        acc.get(f"cert_{fb}", 0) / nb)
        ldn = float(acc.get("len_delta_n", 0))
        if ldn:
            out["pt_cert_len_delta_mean"] = (
                acc.get("len_delta_sum", 0) / ldn)
            out["pt_cert_shorten_rate"] = (
                acc.get("shorten_n", 0) / ldn)
        ca = float(acc.get("cert_attempts", 0))
        if ca:
            out["pt_cert_replicates_mean"] = (
                acc.get("cert_replicates", 0) / ca)
            out["pt_cert_starved_rate"] = (
                acc.get("cert_starved", 0) / ca)
        rn = float(acc.get("renorm_n", 0))
        if rn:
            out["pt_renorm_factor_mean"] = (
                acc.get("renorm_factor_sum", 0.0) / rn)
        hn = float(acc.get("half_obs_n", 0))
        if hn:
            # A binding projection cap must be VISIBLE (round-26
            # C1): silent truncation shortens every graded reply.
            out["pt_half_cap_hit_rate"] = (
                acc.get("half_cap_hits", 0) / hn)
        # Empty-denominator means are OMITTED, not zeroed: the
        # actor pool averages whatever keys each actor reports, and
        # a fabricated 0.0 is indistinguishable from a measured one
        # (round-9 C5; same class as the None-when-idle drain fix).
        mn = float(acc.get("margin_n", 0))
        if mn:
            out["pt_margin_mean"] = acc.get("margin_sum", 0.0) / mn
        c = float(acc.get("certified", 0))
        if c:
            out["pt_cert_margin_mean"] = (
                acc.get("cert_margin_sum", 0.0) / c)
        if betas:
            bs = np.array(betas, dtype=np.float64)
            out["pt_beta_p50"] = float(np.percentile(bs, 50))
            out["pt_beta_p90"] = float(np.percentile(bs, 90))
        return out
