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
    # Boundary evaluation frame (2026-08-21 fog finding, leg-4
    # postmortem): the post-end_turn boundary state's acting side is
    # the OPPONENT, and the encoder is acting-side-framed -- so the
    # grader saw only the opponent's fogged view of the mover's
    # turn. On no-contact fogged turns EVERY candidate graded
    # bit-identically (measured: 4 different candidate turns, one
    # value to 16 digits; fogless control spread 0.24-0.63).
    #   opponent -- post-flip state, sign-flipped (status quo;
    #               assumes fog symmetry that does not exist).
    #   mover    -- the PRE-end_turn state, mover still acting: the
    #               mover's own information set. Terminal flips
    #               still grade by exact outcome.
    # Default stays "opponent" until the A/B probes re-baseline;
    # leg-5 config must assert this explicitly.
    boundary_frame:      str = "opponent"  # opponent | mover


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
        boundary_frame=str(getattr(args, "turn_boundary_frame",
                                   "opponent")),
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


# One inference request per chunk; matches the pool server's
# max_batch=16 convention (tools/actor_pool.py) so a single actor's
# candidate batch cannot monopolize a serve cycle.
BOUNDARY_BATCH_CHUNK = 16


def batch_boundary_values(policy, mats: List["Materialized"],
                          side: int, decision_step: int,
                          chunk: int = BOUNDARY_BATCH_CHUNK) -> None:
    """Fill `m.value` IN PLACE for skip_value materializations, in
    batched forwards. Terminal boundaries use the exact outcome
    (`_terminal_value`, no forward -- same contract as
    `boundary_value`); invalid entries are left NaN. Frees each
    entry's `boundary_sim` afterwards so candidate forks don't
    accumulate (up to K*n_alt live forks otherwise). Per-value
    results are identical to per-sim `boundary_value` calls -- the
    batched forward path is pinned per-sample-equal by the model
    tests; only the transport is batched."""
    live: List["Materialized"] = []
    for m in mats:
        if m.invalid or m.boundary_sim is None:
            continue
        if m.boundary_sim.done:
            m.value = _terminal_value(m.boundary_sim, side,
                                      tiebreak=None)
            m.boundary_sim = None
        else:
            live.append(m)
    for lo in range(0, len(live), max(1, chunk)):
        part = live[lo:lo + max(1, chunk)]
        with torch.no_grad():
            encs = [policy._inference_encoder.encode(m.boundary_sim.gs)
                    for m in part]
            outs = policy._inference_model.forward_batch(encs)
        for m, out in zip(part, outs):
            m.value = _value_for(out, m.boundary_sim.gs, side)
            m.boundary_sim = None


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
                keep_boundary_sim: bool = False,
                skip_value: bool = False,
                mover_frame: bool = False) -> Materialized:
    """Replay `commands` from a fork of `start` under `salt`; evaluate
    at the boundary. Clean bounces (`last_step_rejected`) are skipped
    and the replay continues; a raised exception marks the variant
    invalid (excluded from selection). `end_turn` is appended when the
    list doesn't end the turn on its own, so the result is always a
    complete turn.

    `skip_value=True` skips the boundary forward and keeps the
    boundary sim instead (value stays NaN): the caller either grades
    by projection rollout, or evaluates MANY candidates in one
    `batch_boundary_values` call -- the pool's inference server is
    the serial bottleneck (~54 fwd/s, leg-3 A6 postmortem), so 48
    one-at-a-time boundary forwards per turn plan was the single
    largest avoidable latency in the pipeline. NOTE: with skip_value
    the invalid check is `m.invalid` (value is NaN either way).

    `mover_frame=True` (TurnSearchConfig.boundary_frame="mover")
    evaluates the PRE-end_turn state -- the mover still acting --
    instead of the post-flip state. The encoder is acting-side-
    framed, so the post-flip boundary is the OPPONENT'S fogged view
    and is structurally blind to whatever the opponent cannot see
    of the mover's turn (2026-08-21 leg-4 postmortem finding).
    Terminal flips still grade by exact outcome. When the CALLER
    keeps the boundary sim (projection), the post-flip sim is kept
    regardless -- projection's deep boundaries are not yet
    frame-fixed."""
    explicit_keep = keep_boundary_sim
    sim = start.fork()
    sim._seed_salt = salt
    rng0 = sim._rng_requests
    executed: List[Dict] = []
    attempted = accepted = 0
    invalid = False
    pre_flip = None
    cmds = list(commands)
    if not any(c.get("type") == "end_turn" for c in cmds):
        cmds.append({"type": "end_turn"})
    for cmd in cmds:
        if sim.done or sim.gs.global_info.current_side != side:
            break
        attempted += 1
        if (mover_frame and not explicit_keep
                and cmd.get("type") == "end_turn"):
            pre_flip = sim.fork()   # re-fork on every attempt: a
            #                         bounced end_turn leaves a stale
            #                         snapshot otherwise
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
        if mover_frame and not explicit_keep:
            pre_flip = sim.fork()
        try:
            sim.step({"type": "end_turn"})
            executed.append({"type": "end_turn"})
        except Exception:  # noqa: BLE001
            invalid = True
    # Terminal flips (sim.done) keep the post-flip sim so the exact
    # outcome grades the boundary; only live boundaries switch to
    # the mover's pre-flip information set.
    eval_sim = sim
    if (mover_frame and not explicit_keep and not invalid
            and not sim.done and pre_flip is not None):
        eval_sim = pre_flip
    value = float("nan") if (invalid or skip_value) else boundary_value(
        policy, eval_sim, side, decision_step)
    if skip_value and not invalid:
        keep_boundary_sim = True
    vis = frozenset(
        u.id for u in units_visible_to(sim.gs, side) if u.side != side
    ) if not invalid else frozenset()
    return Materialized(
        executed=executed, attempted=attempted, accepted=accepted,
        value=value, done=sim.done,
        stochastic=(sim._rng_requests > rng0), invalid=invalid,
        vis_ids=vis,
        boundary_sim=(sim if explicit_keep else eval_sim)
        if keep_boundary_sim else None)


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
    # Fog-frame instrumentation (2026-08-21): a coordinate is BLIND
    # when every evaluated candidate graded identically -- the
    # grader contributed zero information and the target degenerates
    # to prior^lam. In-vivo counterpart of the E2 blind fraction.
    if int(evaluated.sum()) >= 2:
        ev_vals = values[evaluated]
        stats["blind_coord"] = (1.0 if float(np.ptp(ev_vals)) < 1e-9
                                else 0.0)
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
    # Gate-effectiveness telemetry (2026-08-21 user directive:
    # instrument frame/projection effectiveness). Only pairings that
    # were INDEPENDENTLY re-graded count (the deterministic
    # replicate-skip carries no new information).
    gate_n:        int = 0                 # re-graded pairings
    gate_flips:    int = 0                 # reval sign != stage-1 sign
    gate_delta_sum: float = 0.0            # sum(stage1 delta - reval mean)
    gate_shortens: int = 0                 # accepts that SHORTENED the turn

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
    mf = cfg.boundary_frame == "mover"

    def _projected(m: Materialized) -> float:
        nonlocal n_proj
        n_proj += 1
        return project_value(policy, m.boundary_sim, side,
                             decision_step, cfg.project_halfturns,
                             cfg.project_max_actions, rng)

    for rnd in range(n_rounds):
        salt = f"{salt_ns}:r{rnd}"
        # Sim work first (skip_value), boundary forwards BATCHED
        # after -- the pool's inference server is the serial
        # bottleneck (A6 postmortem: ~54 fwd/s shared by ~24 games),
        # so one batched request per round replaces up to
        # 1 + K*n_alt one-at-a-time round-trips. Values are
        # identical; only the transport changes. Under project="all"
        # grades come from rollouts and no boundary forward is
        # issued at all (the review's redundant-forward patch).
        inc = materialize(policy, sim, side, commands, salt,
                          decision_step, keep_boundary_sim=proj_all,
                          skip_value=True, mover_frame=mf)
        if inc.invalid:
            log.warning("plan_turn: incumbent materialization invalid")
            break
        raw: List[Tuple[int, int, Materialized]] = []
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
                                keep_boundary_sim=proj_all,
                                skip_value=True, mover_frame=mf)
                if m.invalid:
                    continue
                raw.append((j, alt_i, m))
        if proj_all:
            inc_val = _projected(inc)
            cands = [(j, a, m, _projected(m)) for j, a, m in raw]
        else:
            batch_boundary_values(policy, [inc] + [m for _, _, m in raw],
                                  side, decision_step)
            inc_val = inc.value
            cands = [(j, a, m, m.value) for j, a, m in raw
                     if not math.isnan(m.value)]
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
        regraded = False
        if (not use_proj and not best_m.stochastic
                and not inc.stochastic):
            reval = np.array([float(deltas[best])])
        else:
            regraded = True
            pairs = []
            for v in range(cfg.reval_salts):
                s2 = f"{salt}:v{v}"
                inc2 = materialize(policy, sim, side, commands, s2,
                                   decision_step,
                                   keep_boundary_sim=use_proj,
                                   skip_value=True, mover_frame=mf)
                var2 = materialize(policy, sim, side, best_cmds, s2,
                                   decision_step,
                                   keep_boundary_sim=use_proj,
                                   skip_value=True, mover_frame=mf)
                if inc2.invalid or var2.invalid:
                    continue
                pairs.append((inc2, var2))
            reval_l = []
            if use_proj:
                for inc2, var2 in pairs:
                    reval_l.append(_projected(var2) - _projected(inc2))
            elif pairs:
                flat = [m for pr in pairs for m in pr]
                batch_boundary_values(policy, flat, side,
                                      decision_step)
                reval_l = [var2.value - inc2.value
                           for inc2, var2 in pairs
                           if not (math.isnan(inc2.value)
                                   or math.isnan(var2.value))]
            reval = (np.array(reval_l) if reval_l
                     else np.array([float("-inf")]))
        accept, _dbar, _ = two_stage_accept(reval, cfg.min_delta)
        if regraded and math.isfinite(_dbar):
            # Stage-1 verdict vs the gate's independent re-grade:
            # under projection reval this is exactly the boundary-vs-
            # projected disagreement (the Q7 quantity, in vivo); with
            # projection off it is the salt-reval shift.
            plan.gate_n += 1
            plan.gate_delta_sum += float(deltas[best]) - float(_dbar)
            if (float(deltas[best]) > 0.0) != (float(_dbar) > 0.0):
                plan.gate_flips += 1
        if not accept:
            break
        accepts += 1
        if len(best_m.executed) < len(inc.executed):
            plan.gate_shortens += 1
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
                      decision_step, keep_boundary_sim=proj_all,
                      skip_value=True, mover_frame=mf) if full else None
    # Materialize every coordinate's alternatives first (sim work),
    # then grade ALL boundaries in one batched pass -- this is the
    # "48 serial forwards per turn plan" hot spot the A6 postmortem
    # flagged. NOTE the gumbel draws happen in coordinate order
    # before any evaluation, so the rng stream is identical to the
    # old interleaved loop.
    per_coord: List[List[Tuple[int, Materialized]]] = []
    if full and inc is not None and not inc.invalid:
        for j, st in enumerate(steps):
            priors = np.array([a.prior for a in st.legal])
            et_idx = next((i for i, a in enumerate(st.legal)
                           if a.action.get("type") == "end_turn"),
                          None)
            coord: List[Tuple[int, Materialized]] = []
            for alt_i in gumbel_top_k_alternatives(
                    priors, st.action_idx, et_idx, cfg.n_alt, rng):
                cand = (commands[:j] + [st.legal[alt_i].action]
                        + commands[j + 1:])
                m = materialize(policy, sim, side, cand, kl_salt,
                                decision_step,
                                keep_boundary_sim=proj_all,
                                skip_value=True, mover_frame=mf)
                if not m.invalid:
                    coord.append((alt_i, m))
            per_coord.append(coord)
        if not proj_all:
            batch_boundary_values(
                policy, [inc] + [m for c in per_coord
                                 for _, m in c],
                side, decision_step)
    # Under "all" placement the distill targets rank actions by the
    # PROJECTED objective -- the training signal itself learns the
    # tempo-aware ordering. (v_root stays the blind pre-state value:
    # it only sets the unevaluated-mass fallback.)
    inc_target_val = 0.0
    if inc is not None and not inc.invalid:
        inc_target_val = _projected(inc) if proj_all else inc.value
    for j, st in enumerate(steps):
        plan.commands.append(st.action)
        plan.pre_keys.append(state_key(st.pre_fork.gs))
        if not full or inc is None or inc.invalid:
            plan.targets.append(None)
            plan.stats.append(None)
            continue
        values = np.zeros(len(st.legal))
        evaluated = np.zeros(len(st.legal), dtype=bool)
        values[st.action_idx] = inc_target_val
        evaluated[st.action_idx] = True
        for alt_i, m in per_coord[j]:
            v = _projected(m) if proj_all else m.value
            if math.isnan(v):
                continue
            values[alt_i] = v
            evaluated[alt_i] = True
        tuples, stats = build_coordinate_target(
            st.legal, values, evaluated, st.pre_value,
            float(evaluated.sum()), mcts_config,
            link=cfg.target_link, beta=cfg.target_beta)
        plan.targets.append(tuples)
        plan.stats.append(stats)
    plan.projections = n_proj
    return plan
