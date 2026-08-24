"""TurnCommitPolicy -- TCS as the self-play data generator.

Subclasses `MCTSPolicy` and swaps ONLY the decision procedure: instead
of a per-micro-action Gumbel root search, each side-turn is planned
once by turn-commitment search (`tools/turn_search.plan_turn`) and
executed one command per `select_action` call. Everything downstream
-- the `_PendingMCTSState` schema, terminal-z sealing, per-side game
weights, aux/moves-left targets, holdout diversion, boundary-pair
harvest, the replay buffer, `train_step`, checkpoint I/O -- is
INHERITED UNCHANGED, so TCS data trains through the exact same
pipeline as MCTS data.

Execution semantics (docs/tcs_spec.md par.3, "plan once, re-plan when
surprised"):
  * The plan stores each coordinate's expected pre-state key
    (`TurnPlan.pre_keys`). Before serving command i, the LIVE state's
    key must match; a mismatch means a realized stochastic outcome
    (combat, trait roll) diverged from the planning branch -> warm
    re-plan from the realized state with the remaining commands as
    the incumbent. This makes chance-node re-planning exact: it fires
    precisely when the realized outcome differs from the planned one.
  * A live recruit bounce invalidates the plan: `play_one_game` calls
    `drop_last_pending` (inherited: pops the pending target + rolls
    back decision_step), and our override also discards the plan so
    the retry re-plans against the updated rejection set -- serving
    the cached command again would loop on the same fogged hex.
  * Targets are recorded ONLY for coordinates that actually execute
    on-trajectory (the pre-state key matched), so every stored
    experience pairs a reached state with a target whose indices were
    enumerated on a state-key-identical state.

Boundary value experiences (spec par.4) are deliberately NOT emitted:
with TCS playing both sides, every turn-boundary state is already
recorded as the opponent's first-coordinate experience, so extra
value-only experiences would duplicate states while requiring the
trainer normalizer/harvest changes the spec warns about. Recorded as
a measured follow-up (upweighting coordinate-0 states) in the spec.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

from wesnoth_ai.classes import GameState, state_key
from tools.mcts import MCTSConfig
from tools.mcts_policy import MCTSPolicy, _PendingMCTSState
from tools.turn_search import TurnPlan, TurnSearchConfig, plan_turn

log = logging.getLogger("turn_policy")


class TurnCommitPolicy(MCTSPolicy):
    """MCTSPolicy with the decision procedure replaced by TCS."""

    def __init__(self, base, mcts_config: Optional[MCTSConfig] = None,
                 *args, turn_config: Optional[TurnSearchConfig] = None,
                 **kwargs):
        super().__init__(base, mcts_config, *args, **kwargs)
        self._turn_cfg = turn_config or TurnSearchConfig()
        self._plans: Dict[str, TurnPlan] = {}
        self._plan_seq = 0
        # Telemetry: planning passes / warm re-plans / accepted
        # improvements, drained alongside the distill stats.
        self._tcs_plans = 0
        self._tcs_replans = 0
        self._tcs_accepts = 0
        self._tcs_projections = 0
        self._tcs_gate_n = 0
        self._tcs_gate_flips = 0
        self._tcs_gate_delta = 0.0
        self._tcs_gate_shortens = 0

    # -- decision procedure -------------------------------------------

    def select_action(self, game_state: GameState, *,
                      game_label: str = "default", sim=None) -> Dict:
        if sim is None:
            raise RuntimeError(
                "TurnCommitPolicy.select_action requires `sim=` to "
                "fork the turn search from.")
        if game_state is sim.gs:
            raise ValueError(
                "TurnCommitPolicy.select_action was passed the LIVE "
                "sim.gs; it must be a deepcopy snapshot (sim.step "
                "would mutate the recorded training target). See "
                "play_one_game's `copy.deepcopy(sim.gs)`.")
        # Per-decision progress counter (combat-oracle anneal +
        # drop_last_pending rollback contract): advance one per call.
        # Stored experiences carry the PLAN's enumeration counter so
        # the distillation loss rebuilds reference logits at the same
        # alpha the plan's priors used.
        with self._base._lock:
            ds_call = self._base._decision_step
            self._base._decision_step += 1
        side = sim.gs.global_info.current_side
        turn_no = int(sim.gs.global_info.turn_number)
        live_key = state_key(sim.gs)
        with self._lock:
            plan = self._plans.get(game_label)

        fresh_turn = (plan is None or plan.side != side
                      or plan.turn_no != turn_no or plan.exhausted)
        diverged = (not fresh_turn
                    and plan.pre_keys[plan.cursor] != live_key)
        if fresh_turn or diverged:
            warm = None
            if diverged:
                # Realized stochastic outcome differs from the
                # planning branch: warm re-plan, remaining commands
                # as incumbent, keep the turn's full/fast draw.
                warm = plan.commands[plan.cursor:]
                full = plan.full
                self._tcs_replans += 1
            else:
                full = bool(self._rng.random()
                            < self._turn_cfg.turn_full_prob)
            with self._lock:
                self._plan_seq += 1
                salt_ns = f"tcs:{game_label}:{self._plan_seq}"
            plan = plan_turn(self._base, sim, side, ds_call,
                             self._turn_cfg, self._mcts_config,
                             self._rng, salt_ns, full,
                             incumbent=warm)
            self._tcs_plans += 1
            self._tcs_accepts += plan.accepts
            self._tcs_projections += plan.projections
            self._tcs_gate_n += plan.gate_n
            self._tcs_gate_flips += plan.gate_flips
            self._tcs_gate_delta += plan.gate_delta_sum
            self._tcs_gate_shortens += plan.gate_shortens

        cmd = plan.commands[plan.cursor]
        target = plan.targets[plan.cursor]
        stats = plan.stats[plan.cursor]
        plan.cursor += 1
        with self._lock:
            self._plans[game_label] = plan
            recorded = bool(target)
            if recorded:
                self._pending.setdefault(game_label, []).append(
                    _PendingMCTSState(
                        gs=game_state, visit_counts=target, side=side,
                        decision_step=plan.decision_step))
                if stats:
                    a = self._distill_acc
                    a["n"] = a.get("n", 0) + 1
                    for k in ("tgt_entropy", "prior_entropy",
                              "sharpen_top"):
                        a[k] = a.get(k, 0.0) + stats[k]
                    a["kl_prior"] = (a.get("kl_prior", 0.0)
                                     + stats.get("kl_prior", 0.0))
                    if "link_clip_frac" in stats:
                        a["link_n"] = a.get("link_n", 0) + 1
                        a["link_clip"] = (a.get("link_clip", 0.0)
                                          + stats["link_clip_frac"])
                    if "blind_coord" in stats:
                        a["blind_n"] = a.get("blind_n", 0) + 1
                        a["blind"] = (a.get("blind", 0.0)
                                      + stats["blind_coord"])
                    a["top80"] = a.get("top80", 0) + (
                        1 if stats["prior_top"] > 0.8 else 0)
                    if "et_prior" in stats:
                        a["et_n"] = a.get("et_n", 0) + 1
                        a["et_prior"] = (a.get("et_prior", 0.0)
                                         + stats["et_prior"])
                        a["et_target"] = (a.get("et_target", 0.0)
                                          + stats["et_target"])
            self._last_recorded[game_label] = recorded
        return cmd

    # -- plan lifecycle -----------------------------------------------

    def drop_last_pending(self, game_label: str) -> bool:
        """Bounce contract: undo the last decision AND discard the
        cached plan -- the retry must re-plan against the updated
        rejection set (re-serving the cached command would loop)."""
        handled = super().drop_last_pending(game_label)
        with self._lock:
            self._plans.pop(game_label, None)
        return handled

    def drop_pending(self, game_label: str) -> None:
        super().drop_pending(game_label)
        with self._lock:
            self._plans.pop(game_label, None)

    def finalize_game(self, game_label: str, winner: int,
                      final_gs=None, midgame: bool = False) -> None:
        with self._lock:
            self._plans.pop(game_label, None)
        super().finalize_game(game_label, winner, final_gs=final_gs,
                              midgame=midgame)

    # -- telemetry ----------------------------------------------------

    def drain_distill_stats(self) -> Optional[Dict[str, float]]:
        """Merge the TCS planning counters into the distill drain so
        they ride the EXISTING telemetry transport (actor drain ->
        pool mean-of-actor-means -> learner log + snapshot_sink CSV).
        Leg 3 ran with `drain_tcs_stats` defined but never called on
        any path -- the 2026-08-17 collapse postmortem's telemetry
        gap. Rates (per-plan) are shipped because the pool AVERAGES
        across actors; `tcs_plans` is therefore per-actor under the
        pool and absolute in-process."""
        out = super().drain_distill_stats() or {}
        tcs = self.drain_tcs_stats()
        if tcs["tcs_plans"]:
            plans = tcs["tcs_plans"]
            out["tcs_plans"] = float(plans)
            out["tcs_replans_per_plan"] = tcs["tcs_replans"] / plans
            out["tcs_accepts_per_plan"] = tcs["tcs_accepts_per_plan"]
            out["tcs_projections_per_plan"] = (
                tcs["tcs_projections"] / plans)
            # Gate effectiveness (2026-08-21): flip rate + mean
            # stage1-vs-regrade shift (= boundary-vs-projected
            # disagreement when projection reval is on -- the Q7
            # quantity in vivo), and the share of accepts that
            # SHORTEN turns (passivity direction).
            gn = tcs["tcs_gate_n"]
            out["tcs_gate_flip_frac"] = (
                tcs["tcs_gate_flips"] / gn if gn else None)
            out["tcs_gate_delta_reval"] = (
                tcs["tcs_gate_delta"] / gn if gn else None)
            out["tcs_gate_shorten_per_plan"] = (
                tcs["tcs_gate_shortens"] / plans)
        return out or None

    def drain_tcs_stats(self) -> Dict[str, float]:
        """Planning-pass counters since the last drain."""
        out = {"tcs_plans": self._tcs_plans,
               "tcs_replans": self._tcs_replans,
               "tcs_accepts": self._tcs_accepts,
               "tcs_accepts_per_plan": (
                   self._tcs_accepts / self._tcs_plans
                   if self._tcs_plans else 0.0),
               "tcs_projections": self._tcs_projections,
               "tcs_gate_n": self._tcs_gate_n,
               "tcs_gate_flips": self._tcs_gate_flips,
               "tcs_gate_delta": self._tcs_gate_delta,
               "tcs_gate_shortens": self._tcs_gate_shortens}
        self._tcs_plans = self._tcs_replans = self._tcs_accepts = 0
        self._tcs_projections = 0
        self._tcs_gate_n = self._tcs_gate_flips = 0
        self._tcs_gate_delta = 0.0
        self._tcs_gate_shortens = 0
        return out
