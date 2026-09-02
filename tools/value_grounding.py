"""Value grounding on search-consulted states (arm VG, 2026-09-01).

The signal-profiler rounds 5-6 measured the erosion channel: the
value head is trained only on REAL game states, but search reads it
on IMAGINED candidate-turn boundaries, where one training step
moves its verdicts by more than the search's own accept threshold
(3-4 atoms vs ~2), unchecked by any loss. This module turns those
consulted states into training states, two ways:

  (2) ROLLOUT GROUNDING — play the captured boundary sim to the end
      with the raw policy (no search) and label the state with the
      terminal outcome. Truth from the sim, no circularity.
  (3) CONSISTENCY TARGETS — label the state with the projected
      (depth-H) value the search already computed for it. Free, but
      self-referential: biases shared by shallow and deep eval are
      confirmed, not fixed (the leg-3 lesson) — so these ship
      DOWN-WEIGHTED and always alongside (2), never alone.

Capture happens in plan_turn's stage-2 gate (the projection pairs);
labeling runs at finalize_game, worker-side, so the experiences
ride the existing worker->learner channel in both topologies.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("value_grounding")


@dataclass
class GroundingConfig:
    enabled: bool = False
    # Per projected stage-2 evaluation, probability a state is
    # captured (both members of a pair offer independently).
    capture_prob: float = 0.10
    # Consistency experiences per game (cheap: label already known).
    max_consist_per_game: int = 8
    # Rollout-labeled experiences per game (each costs a raw-policy
    # playout to the end of the game, worker-side at finalize).
    max_rollout_per_game: int = 4
    rollout_max_halfturns: int = 120
    rollout_max_actions: int = 30
    rollouts_per_state: int = 1
    # Consistency label = the gate's own stage-2 grade of the
    # captured pre-flip state: end the turn, project H half-turns
    # (mirrors TurnSearchConfig.project_halfturns / max_actions).
    project_halfturns: int = 1
    project_max_actions: int = 40
    # value_weight of the produced experiences. Rollout labels are
    # honest outcomes -> full weight; projected labels are
    # self-referential -> down-weighted (leg-3 ratchet guard).
    ground_value_weight: float = 1.0
    consist_value_weight: float = 0.25


def is_grounding_experience(e) -> bool:
    """Grounding/consistency experiences carry a non-"game"
    label_kind (legacy fallback: no policy target at all). Used to
    keep them OUT of the fresh probe and z-composition telemetry —
    fresh_value_ce must stay comparable across legs."""
    kind = getattr(e, "label_kind", None)
    if kind is not None:
        return kind != "game"
    return (not getattr(e, "visit_counts", None)
            and float(getattr(e, "policy_weight", 1.0)) == 0.0)


def config_from_args(args, turn_cfg) -> Optional[GroundingConfig]:
    """CLI -> config; None when --value-ground is off. Raises on an
    unusable combination (grounding captures ride the stage-2
    projection pairs, so TCS + projection are prerequisites)."""
    if not getattr(args, "value_ground", False):
        return None
    if turn_cfg is None or turn_cfg.project == "none":
        raise ValueError(
            "--value-ground requires TCS with --turn-project "
            "reval|all (capture rides the stage-2 projection pairs)")
    return GroundingConfig(
        enabled=True,
        capture_prob=args.value_ground_capture_prob,
        max_consist_per_game=args.value_ground_consist,
        max_rollout_per_game=args.value_ground_rollouts,
        ground_value_weight=args.value_ground_weight,
        consist_value_weight=args.value_consist_weight,
        project_halfturns=int(turn_cfg.project_halfturns),
        project_max_actions=int(turn_cfg.project_max_actions))


@dataclass
class GroundCapture:
    sim: object            # stage-1 pre-flip WesnothSim (mover frame; referenced)
    side: int              # the planning side (values are side-persp)
    decision_step: int
    projected: Optional[float] = None   # filled at finalize (depth-H grade)


def _handed_over(sim, side: int):
    """Fork and hand the turn over if `side` is still to move (the
    captured state is the mover's pre-flip boundary; every label is
    'end the turn here, then ...')."""
    r = sim.fork()
    if not r.done and r.gs.global_info.current_side == side:
        r.step({"type": "end_turn"})
    return r


def projected_value(policy, sim, side: int, decision_step: int,
                    cfg: GroundingConfig,
                    rng: np.random.Generator) -> float:
    """The gate's stage-2 grade of the captured state: hand over,
    then the production depth-H projection, side perspective."""
    from tools.turn_search import project_value
    r = _handed_over(sim, side)
    return project_value(policy, r, side, decision_step,
                         cfg.project_halfturns, cfg.project_max_actions,
                         rng)


def rollout_outcome(policy, sim, side: int, decision_step: int,
                    cfg: GroundingConfig,
                    rng: np.random.Generator) -> Optional[float]:
    """Terminal outcome from `side`'s perspective: hand the turn
    over, then closed-loop raw-policy play (forked). None when the
    cap cut the game (censored, per the 2026-08-17 truncation ruling
    — a capped rollout is not a draw)."""
    from tools.turn_search import _sample_prior_idx, forward_state
    r = _handed_over(sim, side)
    for _ in range(cfg.rollout_max_halfturns):
        if r.done:
            break
        mover = r.gs.global_info.current_side
        k = 0
        while (not r.done and r.gs.global_info.current_side == mover
               and k < cfg.rollout_max_actions):
            _, _out, legal = forward_state(policy, r.gs, decision_step)
            if not legal:
                break
            try:
                r.step(legal[_sample_prior_idx(legal, rng)].action)
            except Exception:  # noqa: BLE001 -- grounding must not die
                break
            k += 1
        if not r.done and r.gs.global_info.current_side == mover:
            try:
                r.step({"type": "end_turn"})
            except Exception:  # noqa: BLE001
                break
    if not r.done:
        return None
    winner = int(getattr(r, "winner", 0) or 0)
    if winner == side:
        return 1.0
    if winner != 0:
        return -1.0
    return 0.0


def build_grounding_experiences(
        policy, captures: List[GroundCapture], cfg: GroundingConfig,
        rng: np.random.Generator) -> Tuple[List, Dict[str, float]]:
    """Consistency experiences for up to max_consist_per_game
    captures, rollout-grounded experiences for up to
    max_rollout_per_game of them. z is stored in the side-to-move
    perspective of the captured state (trainer contract), flipping
    the side-perspective values when the mover differs."""
    from wesnoth_ai.trainer import MCTSExperience

    stats = {"ground_captures": float(len(captures)),
             "ground_rollouts": 0.0, "ground_censored": 0.0,
             "ground_win": 0.0, "ground_loss": 0.0, "ground_draw": 0.0,
             "consist_n": 0.0, "consist_abs_mean": 0.0}
    exps: List = []
    if not captures:
        return exps, stats

    def _z_stm(c: GroundCapture, val_side: float) -> float:
        stm = c.sim.gs.global_info.current_side
        return val_side if stm == c.side else -val_side

    order = list(rng.permutation(len(captures)))
    consist = order[:cfg.max_consist_per_game]
    roll = order[:cfg.max_rollout_per_game]

    # Rollouts first: the rollout subset is a PREFIX of the consist
    # subset, so every rolled state's consistency experience can
    # carry the rollout outcome as z_pair (the paired data the
    # learner's bias/variance estimates are built from).
    roll_z: Dict[int, float] = {}
    for i in roll:
        c = captures[i]
        outs = []
        for _ in range(max(1, cfg.rollouts_per_state)):
            try:
                o = rollout_outcome(policy, c.sim, c.side,
                                    c.decision_step, cfg, rng)
            except Exception as e:  # noqa: BLE001 -- a broken rollout
                # censors this label, never kills the game seal.
                log.warning(f"grounding rollout failed: {e!r}")
                o = None
            stats["ground_rollouts"] += 1
            if o is None:
                stats["ground_censored"] += 1
            else:
                outs.append(o)
        if not outs:
            continue
        mean_o = float(np.mean(outs))
        if mean_o > 0:
            stats["ground_win"] += 1
        elif mean_o < 0:
            stats["ground_loss"] += 1
        else:
            stats["ground_draw"] += 1
        z_roll = _z_stm(c, mean_o)
        roll_z[i] = z_roll
        exps.append(MCTSExperience(
            game_state=c.sim.gs, visit_counts=[], z=z_roll,
            decision_step=c.decision_step,
            value_weight=cfg.ground_value_weight,
            policy_weight=0.0, label_kind="roll"))

    for i in consist:
        c = captures[i]
        if c.projected is None:
            try:
                c.projected = projected_value(policy, c.sim, c.side,
                                              c.decision_step, cfg, rng)
            except Exception as e:  # noqa: BLE001 -- skip the label
                log.warning(f"grounding projection failed: {e!r}")
                continue
        z = float(np.clip(_z_stm(c, c.projected), -1.0, 1.0))
        exps.append(MCTSExperience(
            game_state=c.sim.gs, visit_counts=[], z=z,
            decision_step=c.decision_step,
            value_weight=cfg.consist_value_weight,
            policy_weight=0.0, label_kind="consist",
            z_pair=roll_z.get(i)))
        stats["consist_n"] += 1
        stats["consist_abs_mean"] += abs(z)
    if stats["consist_n"]:
        stats["consist_abs_mean"] /= stats["consist_n"]
    return exps, stats
