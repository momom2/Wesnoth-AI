"""Turn-Commitment Search core (TCS, docs/tcs_spec.md).

The search object is a complete side-turn (a command sequence ending
at the turn boundary), refined by counterfactual coordinate
substitution and graded ONLY at turn boundaries by the value head.
Validated by the 2026-08-14 rung-1 probe (300 ladder states, two
checkpoints: revalidated accept 0.64/0.46, median accepted delta
0.070/0.106 = ~2 C51 atoms, placebo 0.13/0.18, rho(delta,survival)
~0.02 -- see BACKLOG 2026-08-13 item 0).

This module is the shared core: `tools/turn_counterfactual_probe.py`
(the offline measurement instrument) and `tools/turn_policy.py` (the
production `TurnCommitPolicy`) both import from here so measurement
and production provably cannot diverge -- the same rationale
`_completed_q`'s docstring gives for sharing search/target code.

Key contracts (user rulings 2026-08-13, docs/tcs_spec.md par.5):
  * grade-what-you-commit: acceptance is over MATERIALIZED turns
    (the commands that actually landed, bounces excluded);
  * two-stage acceptance: argmax at the selection salt, then paired
    re-evaluation at fresh salts (the argmax-over-N noise trap);
  * suffix survival is a logged covariate, NEVER a behavioral gate;
  * the live sim is never salted and never mutated -- all search
    work happens on forks (`WesnothSim.fork`).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from wesnoth_ai.action_sampler import (
    LegalActionPrior, enumerate_legal_actions_with_priors,
)
from wesnoth_ai.classes import state_key
from wesnoth_ai.visibility import units_visible_to
from tools.mcts import MCTSConfig, _gumbel_sigma, _terminal_value

log = logging.getLogger("turn_search")


@dataclass
class TurnSearchConfig:
    """TCS knobs (sigma/damping constants come from MCTSConfig so the
    target transform stays byte-shared with the Gumbel path)."""
    n_alt:          int = 4      # alternatives per coordinate/round
    rounds:         int = 3      # hill-climb rounds on full turns
    fast_rounds:    int = 1      # rounds on cheap (no-target) turns
    reval_salts:    int = 3      # fresh salts in acceptance stage 2
    min_delta:      float = 0.01  # accept floor (float-jitter guard)
    max_spine:      int = 40     # hard cap on spine length
    turn_full_prob: float = 0.25  # playout-cap analog, per TURN
    # Multi-turn projection at the boundary (docs/tcs_spec.md par.3;
    # user directive 2026-08-17, generalizing the opponent-reply arm).
    # Candidate turns are graded by the value `project_halfturns`
    # half-turns PAST our boundary, each half-turn played closed-loop
    # by the same policy -- one line, no branching, so cost is LINEAR
    # in depth. This is the guard against value-head tempo blindness
    # (the leg-3 turn-collapse mechanism): passing early stops looking
    # free once the evaluated state shows the opponent's free reply.
    # Placement:
    #   none  -- grade at our own boundary (status quo). DEFAULT OFF.
    #   reval -- projection gates stage-2 acceptance only: the climb
    #            proposes by the cheap boundary objective, the gate
    #            re-grades both sides of the pairing with projection.
    #   all   -- projection also drives stage-1 selection and the
    #            distill targets (the search and the training signal
    #            both optimize the projected objective; costlier).
    project:             str = "none"   # none | reval | all
    project_halfturns:   int = 1        # depth past our boundary
    project_max_actions: int = 40       # per-half-turn action cap
    # Target link function (user ruling 2026-08-17): "random draw
    # among the evaluated actions should not push their probability
    # up" -- evaluation EXPOSURE must carry no expected mass gain
    # under an uninformative grader.
    #   linear -- target = prior^lam * max(0, 1 + beta*(q - LOO
    #            mean of the other evaluated q)); linear in q, so
    #            symmetric judge error cancels to first order and
    #            E[target] ~ prior regardless of how often an action
    #            is evaluated. DEFAULT (leg-4 ruling: the grader is
    #            fresh/unproven; noise-robustness beats the exp
    #            link's concentration).
    #   exp    -- the AlphaZero/Gumbel mirror-descent tilt (sigma
    #            transform shared byte-for-byte with the MCTS path).
    #            Concentrates faster under a KNOWN-GOOD grader, but
    #            convex in q: under noise, evaluated actions gain
    #            expected mass in proportion to evaluation frequency
    #            (the leg-3 R2 end_turn exposure ratchet).
    target_link:         str = "linear"  # linear | exp
    target_beta:         float = 5.0     # linear-link advantage gain
    #   beta=5: an action 5 C51 atoms (0.20) below its evaluated
    #   peers' mean clips to zero mass; 2 atoms (0.08, the probe's
    #   median accepted delta) above gains +40% before renorm.
    #   Derivation in docs/design_constants.md.


def config_from_args(args) -> Optional["TurnSearchConfig"]:
    """Build a TurnSearchConfig from parsed CLI args (shared by
    sim_self_play and selfplay_worker so the flag surface stays
    symmetric -- the worker-side-targets contract). None when
    turn search is off."""
    if not getattr(args, "turn_search", False):
        return None
    project = str(getattr(args, "turn_project", "none"))
    halfturns = int(getattr(args, "turn_project_halfturns", 1))
    max_actions = int(getattr(args, "turn_project_max_actions", 40))
    # Legacy alias: --turn-reply was the depth-1 special case of
    # projection (the 2026-08-14 opponent-reply arm). Map it when the
    # new flag doesn't override.
    legacy_reply = str(getattr(args, "turn_reply", "none"))
    if project == "none" and legacy_reply != "none":
        project = legacy_reply
        halfturns = 1
        max_actions = int(getattr(args, "turn_reply_max_actions", 4))
        log.warning("--turn-reply is deprecated; using projection "
                    f"project={project} halfturns=1 "
                    f"max_actions={max_actions} (--turn-project)")
    return TurnSearchConfig(
        n_alt=int(getattr(args, "turn_alt", 4)),
        rounds=int(getattr(args, "turn_rounds", 3)),
        fast_rounds=int(getattr(args, "turn_fast_rounds", 1)),
        reval_salts=int(getattr(args, "turn_reval_salts", 3)),
        min_delta=float(getattr(args, "turn_min_delta", 0.01)),
        max_spine=int(getattr(args, "turn_max_spine", 40)),
        turn_full_prob=float(getattr(args, "turn_full_prob", 0.25)),
        project=project,
        project_halfturns=halfturns,
        project_max_actions=max_actions,
        target_link=str(getattr(args, "turn_target_link", "linear")),
        target_beta=float(getattr(args, "turn_target_beta", 5.0)),
    )


# ---------------------------------------------------------------------
# Model plumbing
# ---------------------------------------------------------------------

def forward_state(policy, gs, decision_step: int):
    """(encoded, output, legal) for one state, inference-only.
    `policy` is a TransformerPolicy; reading `_inference_*` follows
    the MCTSPolicy precedent (tools/mcts_policy.py:226-227)."""
    with torch.no_grad():
        encoded = policy._inference_encoder.encode(gs)
        output = policy._inference_model(encoded)
        legal = enumerate_legal_actions_with_priors(
            encoded, output, gs, decision_step=decision_step)
    return encoded, output, legal


def _value_for(output, gs, side: int) -> float:
    """output.value is side-to-move-perspective; flip to `side`'s."""
    v = float(output.value.squeeze().item())
    return v if gs.global_info.current_side == side else -v


def boundary_value(policy, sim, side: int, decision_step: int) -> float:
    """V(boundary state) from `side`'s perspective. Terminal states
    use the exact outcome (eval contract: no material tiebreak)."""
    if sim.done:
        return _terminal_value(sim, side, tiebreak=None)
    with torch.no_grad():
        encoded = policy._inference_encoder.encode(sim.gs)
        output = policy._inference_model(encoded)
    return _value_for(output, sim.gs, side)


def project_value(policy, sim, side: int, decision_step: int,
                  half_turns: int, max_actions: int,
                  rng: np.random.Generator) -> float:
    """Boundary value after `half_turns` closed-loop half-turns past
    `sim`'s state, from `side`'s perspective -- the multi-turn
    projection (anti-value-exploitation guard; sole guard per user
    ruling 2026-08-13, generalized to depth H 2026-08-17).

    Each half-turn: the side to move plays <=max_actions with the
    same policy (sampled from the enumerated priors, no branching);
    if the cap cuts the turn short, end_turn is forced so half-turns
    stay well-defined and the walk advances. A terminal state grades
    exactly (eval contract: no material tiebreak). Never mutates
    `sim`; all play happens on a fork."""
    if half_turns <= 0 or sim.done:
        return boundary_value(policy, sim, side, decision_step)
    r = sim.fork()
    for _ in range(half_turns):
        if r.done:
            break
        mover = r.gs.global_info.current_side
        k = 0
        while (not r.done and r.gs.global_info.current_side == mover
               and k < max_actions):
            _, output, legal = forward_state(policy, r.gs,
                                             decision_step)
            if not legal:
                break
            try:
                r.step(legal[_sample_prior_idx(legal, rng)].action)
            except Exception:  # noqa: BLE001 -- search must not die
                break
            k += 1
        if not r.done and r.gs.global_info.current_side == mover:
            try:
                r.step({"type": "end_turn"})
            except Exception:  # noqa: BLE001
                break
    return boundary_value(policy, r, side, decision_step)


# ---------------------------------------------------------------------
# Spine
# ---------------------------------------------------------------------

@dataclass
class SpineStep:
    pre_fork:      object                # WesnothSim, state BEFORE action
    pre_value:     float                 # V(pre_state), side persp.
    action:        Dict
    action_idx:    int                   # index into `legal`
    legal:         List[LegalActionPrior]
    decision_step: int


def record_spine(policy, sim0, side: int, decision_step: int,
                 rng: np.random.Generator, max_spine: int = 40,
                 actions: Optional[List[Dict]] = None,
                 ):
    """Walk one side-turn from a fork of `sim0`, recording per
    coordinate the pre-action fork, chosen action, and the full
    legal list with priors.

    With `actions=None` the policy plays (sampled from the enumerated
    joint priors). With `actions` given, replays that command list
    (the respine after an accepted improvement); on divergence
    (a command no longer in the legal list) falls back to policy
    sampling for the remainder. Unsalted -- the spine is the
    on-distribution reference; salts belong to variant evaluation.

    Returns (steps, boundary_sim)."""
    sim = sim0.fork()
    steps: List[SpineStep] = []
    k = 0
    while (not sim.done
           and sim.gs.global_info.current_side == side
           and k < max_spine):
        pre_fork = sim.fork()
        _, output, legal = forward_state(policy, sim.gs, decision_step)
        if not legal:
            log.warning("spine: empty legal list; ending turn walk")
            break
        v0 = _value_for(output, sim.gs, side)
        if actions is not None and k < len(actions):
            idx = _match_action(legal, actions[k])
            if idx is None:
                actions = None
                idx = _sample_prior_idx(legal, rng)
        else:
            idx = _sample_prior_idx(legal, rng)
        act = legal[idx].action
        steps.append(SpineStep(pre_fork=pre_fork, pre_value=v0,
                               action=act, action_idx=idx,
                               legal=legal, decision_step=decision_step))
        try:
            sim.step(act)
        except Exception as e:  # noqa: BLE001 -- search must not die
            log.warning(f"spine: step raised {e!r}; ending walk")
            break
        k += 1
        if act.get("type") == "end_turn":
            break
    return steps, sim


def _sample_prior_idx(legal: List[LegalActionPrior],
                      rng: np.random.Generator) -> int:
    p = np.array([max(a.prior, 0.0) for a in legal], dtype=np.float64)
    s = p.sum()
    if s <= 0:
        return int(rng.integers(len(legal)))
    return int(rng.choice(len(legal), p=p / s))


def _match_action(legal: List[LegalActionPrior],
                  action: Dict) -> Optional[int]:
    for i, a in enumerate(legal):
        if a.action == action:
            return i
    return None


# ---------------------------------------------------------------------
# Materialization (grade-what-you-commit)
# ---------------------------------------------------------------------

@dataclass
class Materialized:
    executed:   List[Dict]     # commands that actually landed
    attempted:  int
    accepted:   int
    value:      float          # boundary V, side perspective
    done:       bool           # game ended during the turn
    stochastic: bool           # any synced-RNG request consumed
    invalid:    bool           # a step RAISED (not a clean bounce)
    vis_ids:    frozenset      # visible enemy unit ids at boundary
    boundary_sim: object = None  # the boundary fork (for projection)

    @property
    def survival(self) -> float:
        return self.accepted / self.attempted if self.attempted else 1.0


def materialize(policy, start, side: int, commands: List[Dict],
                salt: str, decision_step: int,
                keep_boundary_sim: bool = False) -> Materialized:
    """Replay `commands` from a fork of `start` under `salt`; evaluate
    at the boundary. Clean bounces (`last_step_rejected`) are skipped
    and the replay continues; a raised exception marks the variant
    invalid (excluded from selection). `end_turn` is appended when the
    list doesn't end the turn on its own, so the result is always a
    complete turn."""
    sim = start.fork()
    sim._seed_salt = salt
    rng0 = sim._rng_requests
    executed: List[Dict] = []
    attempted = accepted = 0
    invalid = False
    cmds = list(commands)
    if not any(c.get("type") == "end_turn" for c in cmds):
        cmds.append({"type": "end_turn"})
    for cmd in cmds:
        if sim.done or sim.gs.global_info.current_side != side:
            break
        attempted += 1
        try:
            sim.step(cmd)
        except Exception as e:  # noqa: BLE001
            log.debug(f"materialize: step raised {e!r}; variant invalid")
            invalid = True
            break
        if getattr(sim, "last_step_rejected", False):
            continue
        accepted += 1
        executed.append(cmd)
    if (not sim.done and not invalid
            and sim.gs.global_info.current_side == side):
        try:
            sim.step({"type": "end_turn"})
            executed.append({"type": "end_turn"})
        except Exception:  # noqa: BLE001
            invalid = True
    value = float("nan") if invalid else boundary_value(
        policy, sim, side, decision_step)
    vis = frozenset(
        u.id for u in units_visible_to(sim.gs, side) if u.side != side
    ) if not invalid else frozenset()
    return Materialized(
        executed=executed, attempted=attempted, accepted=accepted,
        value=value, done=sim.done,
        stochastic=(sim._rng_requests > rng0), invalid=invalid,
        vis_ids=vis,
        boundary_sim=sim if keep_boundary_sim else None)


# ---------------------------------------------------------------------
# Pure decision helpers
# ---------------------------------------------------------------------

def gumbel_top_k_alternatives(priors: np.ndarray, exclude_idx: int,
                              end_turn_idx: Optional[int], k: int,
                              rng: np.random.Generator) -> List[int]:
    """Gumbel-top-k over log-priors, excluding the incumbent's choice,
    force-including end_turn (docs/tcs_spec.md par.3)."""
    n = len(priors)
    if n <= 1:
        return []
    logp = np.log(np.maximum(priors, 1e-12))
    scores = logp + rng.gumbel(size=n)
    scores[exclude_idx] = -np.inf
    order = [int(i) for i in np.argsort(-scores)]
    picks = order[:k]
    if (end_turn_idx is not None and end_turn_idx != exclude_idx
            and end_turn_idx not in picks and len(picks) == k and k > 0):
        picks[-1] = end_turn_idx
    return picks


def two_stage_accept(deltas: np.ndarray, min_delta: float,
                     ) -> Tuple[bool, float, float]:
    """Stage-2 rule on the revalidation deltas (paired, fresh salts):
    accept iff mean > max(2*sd/sqrt(n), min_delta)."""
    d = np.asarray(deltas, dtype=np.float64)
    n = len(d)
    mean = float(d.mean()) if n else float("-inf")
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    thr = max(2.0 * sd / math.sqrt(max(n, 1)), min_delta)
    return mean > thr, mean, thr


def tcs_target_distribution(
    priors: np.ndarray, values: np.ndarray, evaluated: np.ndarray,
    v_root: float, max_visits: float, mcts_config: MCTSConfig,
    lam: Optional[float] = None, temp: Optional[float] = None,
    link: str = "linear", beta: float = 5.0,
    stats_out: Optional[Dict] = None,
) -> np.ndarray:
    """The TCS target distribution over one coordinate's legal list.

    link="exp": the EXISTING transform, byte-shared with the Gumbel
    search path: completed-Q per `_completed_q` semantics (evaluated
    actions keep their paired boundary value; unevaluated fall back
    to v_mix with one visit per evaluated action), sigma via
    `_gumbel_sigma` verbatim (incl. the 0.04 rescale floor),
    lambda/temperature damping from MCTSConfig. Mirror descent:
    concentrates fastest, but convex in q -- evaluation exposure
    gains expected mass under a noisy grader (the leg-3 ratchet).

    link="linear" (default; user ruling 2026-08-17): exposure-
    invariant target. Evaluated actions get a multiplicative factor
    linear in their advantage over the LEAVE-ONE-OUT mean of the
    other evaluated actions; unevaluated actions keep factor 1.
    Linear in q, so symmetric grader error cancels to first order
    and E[target] ~ prior for every action REGARDLESS of evaluation
    frequency (residual: renormalization is second-order and biases
    evaluated actions slightly DOWN, never up -- pinned by
    test_turn_target_link's decoy invariant). Temperature scales
    beta down; lam still discounts the prior. `stats_out` (if
    given) receives link telemetry: clip_frac = fraction of
    evaluated actions whose factor clipped at zero."""
    p = np.maximum(np.asarray(priors, dtype=np.float64), 1e-12)
    p = p / p.sum()
    ev = np.asarray(evaluated, dtype=bool)
    q = np.asarray(values, dtype=np.float64)
    n_ev = float(ev.sum())
    if lam is None:
        lam = float(getattr(mcts_config, "distill_prior_discount", 1.0))
    if temp is None:
        temp = float(getattr(mcts_config, "distill_target_temp", 1.0))

    if link == "exp":
        if n_ev > 0:
            pv = p[ev]
            weighted = float((pv * q[ev]).sum() / pv.sum())
            v_mix = (v_root + n_ev * weighted) / (1.0 + n_ev)
        else:
            v_mix = v_root
        completed = np.where(ev, q, v_mix)
        t = lam * np.log(p) + _gumbel_sigma(completed, max_visits,
                                            mcts_config)
        if temp != 1.0:
            t = t / max(temp, 1e-6)
        t -= t.max()
        tgt = np.exp(t)
        tgt /= tgt.sum()
        return tgt

    if link != "linear":
        raise ValueError(f"unknown target link {link!r} "
                         f"(expected 'linear' or 'exp')")
    base = p ** lam
    base /= base.sum()
    factor = np.ones(len(p))
    idxs = np.flatnonzero(ev)
    clipped = 0
    if len(idxs) >= 2:
        # A single evaluation carries no comparative information
        # (LOO mean undefined) -> factor stays 1, target = prior.
        b = beta / max(temp, 1e-6)
        qs = q[idxs]
        s = float(qs.sum())
        for k, i in enumerate(idxs):
            loo = (s - qs[k]) / (len(idxs) - 1)
            f = 1.0 + b * (float(qs[k]) - loo)
            if f <= 0.0:
                f = 0.0
                clipped += 1
            factor[i] = f
    if stats_out is not None:
        stats_out["link_clip_frac"] = (clipped / len(idxs)
                                       if len(idxs) else 0.0)
    tgt = base * factor
    total = tgt.sum()
    if total <= 0.0:                # all evaluated AND all clipped
        return p.copy()
    return tgt / total


def build_coordinate_target(
    legal: List[LegalActionPrior], values: np.ndarray,
    evaluated: np.ndarray, v_root: float, max_visits: float,
    mcts_config: MCTSConfig,
    link: str = "linear", beta: float = 5.0,
) -> Tuple[List[Tuple], Dict[str, float]]:
    """One coordinate's policy target in the trainer's 5-tuple schema
    plus the distill-telemetry stats `extract_gumbel_policy_target`
    emits (kl_prior, sharpen_top, ...)."""
    priors = np.array([a.prior for a in legal], dtype=np.float64)
    stats: Dict[str, float] = {}
    tgt = tcs_target_distribution(priors, values, evaluated, v_root,
                                  max_visits, mcts_config,
                                  link=link, beta=beta,
                                  stats_out=stats)
    p = np.maximum(priors, 1e-12)
    p = p / p.sum()
    top = int(p.argmax())
    stats.update({
        "tgt_entropy": float(-(tgt * np.log(tgt + 1e-12)).sum()),
        "prior_entropy": float(-(p * np.log(p + 1e-12)).sum()),
        "sharpen_top": float(tgt[top] - p[top]),
        "prior_top": float(p[top]),
        "kl_prior": float((tgt * (np.log(tgt + 1e-12)
                                  - np.log(p))).sum()),
    })
    for i, a in enumerate(legal):
        if a.action.get("type") == "end_turn":
            stats["et_prior"] = float(p[i])
            stats["et_target"] = float(tgt[i])
            break
    tuples = [
        (a.actor_idx, a.target_idx, a.weapon_idx, float(w), a.type_idx)
        for a, w in zip(legal, tgt) if w > 1e-9
    ]
    return tuples, stats


# ---------------------------------------------------------------------
# The full planning pass
# ---------------------------------------------------------------------

@dataclass
class TurnPlan:
    """A refined turn commitment, ready to execute one command per
    select_action call. `pre_keys[i]` is the state_key the live state
    must match before serving command i (mismatch = a realized
    stochastic outcome diverged from the planning branch -> re-plan;
    this IS the plan-once-replan-at-chance rule, docs/tcs_spec.md
    par.3). `targets[i]` is the coordinate's 5-tuple policy target
    (None on cheap turns). `stats[i]` carries distill telemetry."""
    side:          int
    decision_step: int                     # enumeration ds (anneal)
    turn_no:       int = 0                 # game turn the plan is for
    full:          bool = False            # targets recorded?
    commands:      List[Dict] = field(default_factory=list)
    pre_keys:      List[int] = field(default_factory=list)
    targets:       List[Optional[List[Tuple]]] = field(
        default_factory=list)
    stats:         List[Optional[Dict]] = field(default_factory=list)
    cursor:        int = 0
    accepts:       int = 0
    projections:   int = 0                 # project_value calls made

    @property
    def exhausted(self) -> bool:
        return self.cursor >= len(self.commands)


def plan_turn(policy, sim, side: int, decision_step: int,
              cfg: TurnSearchConfig, mcts_config: MCTSConfig,
              rng: np.random.Generator, salt_ns: str,
              full: bool, incumbent: Optional[List[Dict]] = None,
              ) -> TurnPlan:
    """Spine -> hill-climb rounds (two-stage acceptance, materialized
    -turn semantics) -> per-coordinate targets (full turns only).
    Never mutates `sim`; all work on forks. `salt_ns` namespaces the
    search salts (never applied to a live sim). `incumbent` warm-
    starts the climb (execution-time re-plan after a realized
    stochastic outcome diverged)."""
    steps, _ = record_spine(policy, sim, side, decision_step, rng,
                            max_spine=cfg.max_spine, actions=incumbent)
    plan = TurnPlan(side=side, decision_step=decision_step,
                    turn_no=int(sim.gs.global_info.turn_number),
                    full=full)
    if not steps:
        plan.commands = [{"type": "end_turn"}]
        plan.pre_keys = [state_key(sim.gs)]
        plan.targets = [None]
        plan.stats = [None]
        return plan
    commands = [s.action for s in steps]
    n_rounds = cfg.rounds if full else cfg.fast_rounds
    accepts = 0
    # Projection placement (docs/tcs_spec.md par.3): `use_proj` grades
    # stage-2 pairings H half-turns out; `proj_all` extends that to
    # stage-1 selection and the distill targets.
    use_proj = (cfg.project in ("reval", "all")
                and cfg.project_halfturns > 0)
    proj_all = use_proj and cfg.project == "all"
    n_proj = 0

    def _projected(m: Materialized) -> float:
        nonlocal n_proj
        n_proj += 1
        return project_value(policy, m.boundary_sim, side,
                             decision_step, cfg.project_halfturns,
                             cfg.project_max_actions, rng)

    for rnd in range(n_rounds):
        salt = f"{salt_ns}:r{rnd}"
        inc = materialize(policy, sim, side, commands, salt,
                          decision_step, keep_boundary_sim=proj_all)
        if inc.invalid:
            log.warning("plan_turn: incumbent materialization invalid")
            break
        inc_val = _projected(inc) if proj_all else inc.value
        cands: List[Tuple[int, int, Materialized, float]] = []
        for j, st in enumerate(steps):
            priors = np.array([a.prior for a in st.legal])
            et_idx = next((i for i, a in enumerate(st.legal)
                           if a.action.get("type") == "end_turn"), None)
            for alt_i in gumbel_top_k_alternatives(
                    priors, st.action_idx, et_idx, cfg.n_alt, rng):
                cand_cmds = (commands[:j]
                             + [st.legal[alt_i].action]
                             + commands[j + 1:])
                m = materialize(policy, sim, side, cand_cmds, salt,
                                decision_step,
                                keep_boundary_sim=proj_all)
                if m.invalid or math.isnan(m.value):
                    continue
                cands.append((j, alt_i, m,
                              _projected(m) if proj_all else m.value))
        if not cands:
            break
        deltas = np.array([v - inc_val for _, _, _, v in cands])
        best = int(np.argmax(deltas))
        j, alt_i, best_m, _ = cands[best]
        best_cmds = (commands[:j] + [steps[j].legal[alt_i].action]
                     + commands[j + 1:])
        # Stage 2: paired re-evaluation at fresh salts. Deterministic
        # pairs replicate exactly; skip the redundant forwards -- but
        # only when projection is off: projection rollouts sample the
        # policy through `rng`, so their grades never replicate, and
        # under "reval" placement the gate MUST re-grade with
        # projection (stage 1 graded blind).
        if (not use_proj and not best_m.stochastic
                and not inc.stochastic):
            reval = np.array([float(deltas[best])])
        else:
            reval_l = []
            for v in range(cfg.reval_salts):
                s2 = f"{salt}:v{v}"
                inc2 = materialize(policy, sim, side, commands, s2,
                                   decision_step,
                                   keep_boundary_sim=use_proj)
                var2 = materialize(policy, sim, side, best_cmds, s2,
                                   decision_step,
                                   keep_boundary_sim=use_proj)
                if inc2.invalid or var2.invalid:
                    continue
                if use_proj:
                    reval_l.append(_projected(var2) - _projected(inc2))
                else:
                    reval_l.append(var2.value - inc2.value)
            reval = (np.array(reval_l) if reval_l
                     else np.array([float("-inf")]))
        accept, _, _ = two_stage_accept(reval, cfg.min_delta)
        if not accept:
            break
        accepts += 1
        # Grade-what-you-commit: the new incumbent is the MATERIALIZED
        # winner (commands that landed at the selection salt).
        commands = list(best_m.executed)
        steps, _ = record_spine(policy, sim, side, decision_step, rng,
                                max_spine=cfg.max_spine,
                                actions=commands)
        if not steps:
            break
        commands = [s.action for s in steps]

    # Emit the plan from the final respined steps: commands,
    # pre-state keys (divergence detection), and -- on full turns --
    # per-coordinate targets from a final evaluation pass.
    plan.accepts = accepts
    kl_salt = f"{salt_ns}:t"
    inc = materialize(policy, sim, side, commands, kl_salt,
                      decision_step,
                      keep_boundary_sim=proj_all) if full else None
    # Under "all" placement the distill targets rank actions by the
    # PROJECTED objective -- the training signal itself learns the
    # tempo-aware ordering. (v_root stays the blind pre-state value:
    # it only sets the unevaluated-mass fallback.)
    inc_target_val = (_projected(inc)
                      if (proj_all and inc is not None
                          and not inc.invalid)
                      else (inc.value if inc is not None else 0.0))
    for j, st in enumerate(steps):
        plan.commands.append(st.action)
        plan.pre_keys.append(state_key(st.pre_fork.gs))
        if not full or inc is None or inc.invalid:
            plan.targets.append(None)
            plan.stats.append(None)
            continue
        priors = np.array([a.prior for a in st.legal])
        et_idx = next((i for i, a in enumerate(st.legal)
                       if a.action.get("type") == "end_turn"), None)
        values = np.zeros(len(st.legal))
        evaluated = np.zeros(len(st.legal), dtype=bool)
        values[st.action_idx] = inc_target_val
        evaluated[st.action_idx] = True
        for alt_i in gumbel_top_k_alternatives(
                priors, st.action_idx, et_idx, cfg.n_alt, rng):
            cand = (commands[:j] + [st.legal[alt_i].action]
                    + commands[j + 1:])
            m = materialize(policy, sim, side, cand, kl_salt,
                            decision_step, keep_boundary_sim=proj_all)
            if m.invalid or math.isnan(m.value):
                continue
            values[alt_i] = _projected(m) if proj_all else m.value
            evaluated[alt_i] = True
        tuples, stats = build_coordinate_target(
            st.legal, values, evaluated, st.pre_value,
            float(evaluated.sum()), mcts_config,
            link=cfg.target_link, beta=cfg.target_beta)
        plan.targets.append(tuples)
        plan.stats.append(stats)
    plan.projections = n_proj
    return plan
