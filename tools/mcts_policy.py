"""MCTS+visit-count distillation policy wrapper.

Wraps a `TransformerPolicy` so that the rollout loop in
`tools/sim_self_play.py` can opt into AlphaZero-style training
without touching the loop's interface beyond the new `sim=` kwarg
on `select_action`. Same five-method duck type as
`TransformerPolicy`: `select_action`, `observe`, `finalize_game`,
`train_step`, `save_checkpoint` / `load_checkpoint`.

What MCTS mode changes vs REINFORCE:

  - **Action picking**: an MCTS search produces a visit-count
    distribution over legal actions; we sample (or argmax) from
    that instead of the raw policy logits.
  - **Per-step shaping reward**: ignored. AlphaZero distills the
    final game outcome (z = ±1 for win/loss, 0 for draws/timeouts)
    onto every visited state — no Bellman-style reward shaping.
    The user's `--reward-config` is silently unused in MCTS mode;
    the iter log still reports per-step shaping totals so REINFORCE
    runs and MCTS runs are easy to compare visually.
  - **Trainer step**: `trainer.step_mcts(experiences)` instead of
    `trainer.step(transitions)`. Cross-entropy against the
    visit-count distribution + value loss against z.

Threading: `select_action` and `observe` may be called from
worker threads in parallel rollouts (one game per thread). The
`_pending` and `_queue` accesses are guarded by `_lock` to mirror
TransformerPolicy's thread-safety contract.

Dependencies: tools.mcts, transformer_policy, trainer
Dependents:   tools.sim_self_play (when --mcts is passed)
"""

from __future__ import annotations

import logging
import random
import threading
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from wesnoth_ai.classes import GameState, state_key
from wesnoth_ai.trainer import MCTSExperience, TrainStats
from tools.draw_tiebreak import draw_tiebreak_z, material_margin
from tools.mcts import (
    MCTSConfig, mcts_search, extract_visit_counts, best_action,
    sample_action, extract_gumbel_policy_target,
)


log = logging.getLogger("mcts_policy")

# Normalizer for moves-left targets: remaining_turns / THIS, clipped
# to [0, 1]. Fixed (not per-game max_turns) so the predicted fraction
# means the same wall-clock distance on every map; 200 = the training
# turn cap in use since the 2026-07 campaigns.
MOVES_LEFT_NORM_TURNS = 200.0
# Game-weight floor for HUMAN-DERIVED (midgame-start) games only:
# caps any single state at 1/8 of a game's weight (Fable review M-1;
# user pick K=8, 2026-07-12 -- preserves most of the equal-per-game
# boost for short decisive continuations while bounding variance).
MIDGAME_GW_FLOOR = 8


@dataclass
class ReplayConfig:
    """Experience-replay + multi-epoch training for the MCTS path.

    Diagnosis (2026-06-15): the default one-gradient-step-per-fresh-
    batch-then-discard schedule is severely sample-inefficient. Overfit
    probes showed the value head needs ~80-100 gradient steps to
    converge on a batch, but live training gave it ONE step per
    (shifting, high-variance) batch -> the value head never left its
    ~uniform floor (val loss ~3.56 vs ln(51)=3.93), MCTS produced
    near-uniform visit targets, and the policy plateaued. A replay
    buffer + several minibatch updates per iteration is the AlphaZero-
    standard fix: it decouples gradient steps from data generation so
    the value head can actually converge.

    `enabled=False` reproduces the exact legacy one-pass behavior, so
    this is a safe, A/B-able default-off addition.
    """
    enabled:          bool = False
    capacity:         int  = 4000   # max experiences retained (memory!)
    updates_per_iter: int  = 8      # gradient steps per train_step
    minibatch:        int  = 128    # experiences per gradient step
    min_size:         int  = 512    # warm up (legacy one-pass) until this


@dataclass
class _PendingMCTSState:
    """One mid-game record awaiting a terminal z. We stash the
    `gs` reference (NOT a deepcopy — `_run_one_game` already passes
    in a deepcopy as `pre_state`), the MCTS visit counts emitted
    at this state, and the side-to-move so finalize_game can
    compute z = +1 / -1 / 0 from the perspective of that side.

    `decision_step` is the training-progress counter at the time this
    state's search ran -- carried through to the MCTSExperience so the
    distillation loss rebuilds reference logits at the SAME combat-oracle
    alpha the search used (`combat_alphas_at`)."""
    gs:            GameState
    visit_counts:  List[Tuple]
    side:          int
    decision_step: int = 0


class MCTSPolicy:
    """Wraps a `TransformerPolicy` to do MCTS+distillation."""

    trainable = True
    # Optimization #6 (2026-06-14): MCTS distills terminal z, not
    # per-step shaping -- `observe` is a no-op. Signals the rollout
    # loop to skip the (expensive, Dijkstra-bearing) compute_delta +
    # reward_fn calls entirely.
    uses_step_rewards = False

    def __init__(self, base, mcts_config: Optional[MCTSConfig] = None,
                 replay_config: Optional[ReplayConfig] = None,
                 holdout_size: int = 0,
                 holdout_per_game_cap: int = 64,
                 train_draw_tiebreak: bool = False):
        self._base = base
        self._mcts_config = mcts_config or MCTSConfig()
        self._replay_config = replay_config or ReplayConfig()
        # Draw-label policy for TRAINING targets (2026-07-10). The
        # material tiebreak remains a SEARCH preference (mcts.py
        # _terminal_value) and the aux head still learns material --
        # but by default draws now train the value head toward the
        # honest z=0. Labeling draws with material-z made "predict
        # material" the dominant lesson (~93% of ladder games are
        # draws) and measurably eroded win/loss discrimination
        # (human-corpus late-game AUC 0.88 -> 0.60 in ~80 iters;
        # r_material/r_outcome rose 1.28 -> 2.18).
        self._train_draw_tiebreak = bool(train_draw_tiebreak)
        # Optional diagnostic hook: called with the search ROOT after
        # every mcts_search (see tools/ladder_anatomy.py -- root
        # child-Q spread is the value signal PUCT actually compares).
        # None (default) = zero overhead.
        self.search_stats_sink = None
        # Per-game search diagnostics (engagement telemetry,
        # 2026-07-12): root child-Q spread + how often the search
        # OVERTURNED the prior's argmax. Keyed by game_label; drained
        # by play_one_game via pop_search_diag(). Cheap (a max + a
        # pstdev over <=B root edges per decision).
        self._search_diag = {}
        # Held-out generalization probe: while the holdout has fewer
        # than `holdout_size` experiences, finalize_game diverts WHOLE
        # games here instead of the training queue (whole games, so no
        # sibling state from the same trajectory leaks into training).
        # Each diverted game contributes at most `holdout_per_game_cap`
        # RANDOMLY SAMPLED states (the rest of the game is discarded,
        # NOT trained on) so the probe spans many games: the original
        # whole-game fill made a 512-state holdout out of ~2 games
        # (~275 experiences each, 2026-07-07 diagnosis) and the "CE"
        # measured those 2 games' idiosyncrasies, not generalization.
        # Frozen once full; `holdout_metrics()` evaluates the current
        # net's value CE on it. 0 = off. NOT persisted in checkpoints:
        # a resumed run re-collects from its first games, restarting
        # the curve's baseline (documented in the runbook).
        self._holdout_target = max(0, int(holdout_size))
        self._holdout_per_game_cap = max(1, int(holdout_per_game_cap))
        self._holdout: List[MCTSExperience] = []
        self._holdout_games = 0
        self._holdout_rng = random.Random(0x5EED)
        # Per-game pending: game_label -> List[_PendingMCTSState].
        # Mid-game states accumulate here; `finalize_game` drains
        # them into `_queue` with their terminal z attached.
        self._pending: Dict[str, List[_PendingMCTSState]] = {}
        # Completed: flat list of MCTSExperience, drained by
        # `train_step` and handed to `trainer.step_mcts`.
        self._queue: List[MCTSExperience] = []
        # Experience-replay buffer (only used when replay_config is
        # enabled). Bounded FIFO of the most recent experiences; each
        # holds a per-decision deepcopy of the game state (see
        # _PendingMCTSState), so retaining it across iterations is
        # safe -- no aliasing with the live sim.
        self._replay: "deque[MCTSExperience]" = deque(
            maxlen=self._replay_config.capacity)
        # Dedicated seeded RNG for minibatch sampling (separate from
        # the search RNG) so replay runs are reproducible given a seed.
        self._replay_rng = random.Random(0)
        self._lock = threading.Lock()
        # RNG for temperature sampling at the root (AlphaZero's
        # tau=1 phase). Unseeded like mcts_search's noise RNG --
        # self-play data generation wants diversity, not
        # reproducibility; tests construct their own Generator.
        self._rng = np.random.default_rng()
        # Tree-reuse stash: game_label -> (child_node, child_state_key)
        # for the action played at the previous decision. The next
        # select_action reuses the subtree iff the LIVE state's key
        # matches (deterministic actions match; combat RNG diverges
        # and rebuilds). See MCTSConfig.tree_reuse.
        self._reuse: Dict[str, Tuple] = {}
        # game_label -> whether the LAST select_action recorded a pending
        # training target (True only for full-budget moves). Lets the
        # fog-bounce retry loop drop exactly the rejected decision's tail
        # via drop_last_pending (see select_action / drop_last_pending).
        self._last_recorded: Dict[str, bool] = {}
        # `_inference_*` already exist on base — under the lock used
        # by base's snapshot. We just borrow them. mcts_search runs
        # `torch.no_grad()` internally so concurrent gradient steps
        # via `train_step` won't corrupt these forwards.
        self._inference_model = base._inference_model
        self._inference_encoder = base._inference_encoder

    def _note_search_diag(self, game_label, root, action,
                          reused: bool = False) -> None:
        edges = getattr(root, "edges", None) or []
        if action is None:
            return
        # Tree shape + reuse: recorded for EVERY decision (the
        # spread/overturn pair below needs >= 2 visited edges).
        # Depth is in ACTIONS; see tree_depth_stats.
        from tools.mcts import tree_depth_stats
        dmax, dw, nodes = tree_depth_stats(root)
        # end_turn decision context (premature-end investigation,
        # 2026-07-21): when the search CHOSE end_turn, how many own
        # units still had MP, and how much prior/visit mass the
        # end_turn edge held. gs is the root fork's decision state.
        et_chosen = action.get("type") == "end_turn"
        et_ctx = None
        if et_chosen:
            gs = root.sim.gs
            side = gs.global_info.current_side
            movable = sum(1 for u in gs.map.units
                          if u.side == side and u.current_moves > 0)
            unstruck = sum(1 for u in gs.map.units
                           if u.side == side and not u.has_attacked)
            et_edge = next((e for e in edges
                            if e.action.get("type") == "end_turn"),
                           None)
            tot_v = sum(e.n_visits for e in edges) or 1
            et_ctx = (movable, unstruck,
                      et_edge.prior if et_edge else 0.0,
                      (et_edge.n_visits / tot_v) if et_edge else 0.0)
        qs = [e.q_value for e in edges if e.n_visits > 0]
        spread = overturned = None
        if len(qs) >= 2:
            import statistics as _st
            spread = _st.pstdev(qs)
            prior_best = max(edges, key=lambda e: e.prior)
            # Actions are the edge dicts by identity (gumbel_action
            # and best_action both return edge.action).
            overturned = prior_best.action is not action
        with self._lock:
            d = self._search_diag.setdefault(
                game_label,
                {"n": 0, "spread_sum": 0.0, "overturns": 0,
                 "n_all": 0, "reuse_hits": 0, "depth_max": 0,
                 "depth_w_sum": 0.0, "nodes_sum": 0,
                 "et_n": 0, "et_movable_sum": 0, "et_unstruck_sum": 0,
                 "et_with_movable": 0, "et_prior_sum": 0.0,
                 "et_visit_frac_sum": 0.0})
            d["n_all"] += 1
            d["reuse_hits"] += 1 if reused else 0
            d["depth_max"] = max(d["depth_max"], dmax)
            d["depth_w_sum"] += dw
            d["nodes_sum"] += nodes
            if et_ctx is not None:
                movable, unstruck, et_p, et_vf = et_ctx
                d["et_n"] += 1
                d["et_movable_sum"] += movable
                d["et_unstruck_sum"] += unstruck
                d["et_with_movable"] += 1 if movable > 0 else 0
                d["et_prior_sum"] += et_p
                d["et_visit_frac_sum"] += et_vf
            if spread is not None:
                d["n"] += 1
                d["spread_sum"] += spread
                d["overturns"] += 1 if overturned else 0

    def pop_search_diag(self, game_label: str):
        """Per-game search diagnostics dict (games.jsonl
        engagement.search); None if nothing recorded. Depth in
        actions, et_* = end_turn decision context."""
        with self._lock:
            d = self._search_diag.pop(game_label, None)
        if not d or not d["n_all"]:
            return None
        out = {"n_searches": d["n_all"],
               "reuse_frac": d["reuse_hits"] / d["n_all"],
               "depth_max": d["depth_max"],
               "depth_w_mean": d["depth_w_sum"] / d["n_all"],
               "nodes_mean": d["nodes_sum"] / d["n_all"]}
        if d["n"]:
            out["q_spread_mean"] = d["spread_sum"] / d["n"]
            out["overturn_frac"] = d["overturns"] / d["n"]
        if d["et_n"]:
            out["et_n"] = d["et_n"]
            out["et_movable_mean"] = d["et_movable_sum"] / d["et_n"]
            out["et_unstruck_mean"] = d["et_unstruck_sum"] / d["et_n"]
            out["et_with_movable_frac"] = (
                d["et_with_movable"] / d["et_n"])
            out["et_prior_mean"] = d["et_prior_sum"] / d["et_n"]
            out["et_visit_frac_mean"] = (
                d["et_visit_frac_sum"] / d["et_n"])
        return out

    # ------------------------------------------------------------------
    # Policy duck-type
    # ------------------------------------------------------------------

    def select_action(
        self,
        game_state: GameState,
        *,
        game_label: str = "default",
        sim=None,
    ) -> Dict:
        if sim is None:
            raise RuntimeError(
                "MCTSPolicy.select_action requires `sim=` to fork "
                "the search tree. Update the rollout loop to pass "
                "the live sim through."
            )
        # Enforce the snapshot contract: `game_state` must be a
        # DEEPCOPY of the pre-action state, not the live `sim.gs`. We
        # stash `game_state` in `_pending` as the AlphaZero training
        # target; if it's the live object, subsequent `sim.step` calls
        # mutate it and the recorded action indices stop matching the
        # state's re-encoding at train time (actor-slot drift -> the
        # policy loss IndexErrors or trains on a corrupted target).
        # Every real rollout/eval caller already deepcopies (see
        # play_one_game). Fail loudly here so a future caller/test that
        # forgets can't silently corrupt training data.
        if game_state is sim.gs:
            raise ValueError(
                "MCTSPolicy.select_action was passed the LIVE sim.gs; it "
                "must be a deepcopy snapshot of the pre-action state "
                "(sim.step would otherwise mutate the recorded training "
                "target). See play_one_game's `copy.deepcopy(sim.gs)`."
            )
        # Combat-oracle anneal: capture the policy's current training-
        # progress counter for THIS decision and advance it. The MCTS
        # path otherwise never increments decision_step (only
        # TransformerPolicy.select_action does, on the REINFORCE path),
        # so without this the oracle bias would never anneal in MCTS
        # mode. The captured value drives the search's leaf-prior alpha
        # AND is recorded on the training target so the distillation loss
        # rebuilds reference logits at the same alpha. Atomic under the
        # base lock (worker threads may call select_action concurrently).
        with self._base._lock:
            decision_step = self._base._decision_step
            self._base._decision_step += 1
        reuse_root = None
        if self._mcts_config.tree_reuse:
            with self._lock:
                stash = self._reuse.pop(game_label, None)
            if stash is not None:
                # The stash maps outcome state_key -> searched child
                # node. Deterministic actions have one entry; combat
                # has one per sampled outcome -- if the live RNG
                # produced an outcome the search visited, its subtree
                # is reusable.
                cand = stash.get(state_key(sim.gs))
                if (cand is not None and cand.expanded
                        and not cand.is_terminal):
                    reuse_root = cand
        # Playout-cap randomization (KataGo): most moves are "fast"
        # (cheap search, NOT recorded as a training target); a random
        # fraction `playout_cap_prob` are "full" (full budget AND
        # recorded). Decoupling game-advancing moves from data-
        # generating moves yields ~3-10x more self-play games per
        # GPU-hour while targets still come from full-strength searches.
        cfg = self._mcts_config
        full_move = True
        n_override = None
        if cfg.playout_cap_randomization:
            full_move = bool(self._rng.random() < cfg.playout_cap_prob)
            if not full_move:
                n_override = (cfg.playout_cap_fast_sims
                              or max(1, cfg.n_simulations // 4))
        root = mcts_search(
            sim,
            self._inference_model,
            self._inference_encoder,
            self._mcts_config,
            reuse_root=reuse_root,
            n_sims_override=n_override,
            decision_step=decision_step,
        )
        if self.search_stats_sink is not None:
            self.search_stats_sink(root)
        if self._mcts_config.gumbel_root:
            # Gumbel mode: the search already chose
            # argmax(g + logits + sigma(q̂)) -- stochastic via the
            # Gumbel draws, provably an improved policy. No
            # temperature schedule needed.
            action = root.gumbel_action or best_action(root)
        else:
            # Classic AlphaZero exploration schedule: sample
            # proportional to visit counts for the first
            # `temperature_decisions` decisions of each game, argmax
            # afterwards. The decision index is exactly how many
            # states this game has already recorded.
            with self._lock:
                decision_idx = len(self._pending.get(game_label, ()))
            if decision_idx < self._mcts_config.temperature_decisions:
                action = sample_action(
                    root, self._mcts_config.temperature, self._rng)
            else:
                action = best_action(root)
        self._note_search_diag(game_label, root, action,
                               reused=reuse_root is not None)
        if action is None:
            # No legal action and no terminal — pathological state.
            # The base policy's select_action would have raised; we
            # raise too rather than let the rollout loop wedge.
            raise RuntimeError(
                f"MCTS produced no action at game_label={game_label!r}; "
                f"root.is_terminal={root.is_terminal}, "
                f"n_edges={len(root.edges)}"
            )
        # Record a training target ONLY for full-budget moves. Fast
        # moves advance the game but generate no policy target
        # (playout-cap randomization); their search ran a cheap budget
        # purely to pick a reasonable action.
        if full_move:
            if self._mcts_config.gumbel_root:
                # Completed-Q target: every legal action gets a weight
                # (search Q if visited, mixed value if not) -- denser
                # signal than raw visit counts at small sim budgets.
                visits = extract_gumbel_policy_target(
                    root, self._mcts_config)
            else:
                visits = extract_visit_counts(root)
            side = game_state.global_info.current_side
            with self._lock:
                self._pending.setdefault(game_label, []).append(
                    _PendingMCTSState(gs=game_state, visit_counts=visits,
                                      side=side, decision_step=decision_step)
                )
        # Stash the played edge's outcome children for
        # state-key-checked reuse at the next decision. Action dicts
        # are returned by identity from the edge, so `is` finds the
        # edge; `==` is the fallback for wrappers that copy.
        if self._mcts_config.tree_reuse:
            edge = next(
                (e for e in root.edges
                 if e.action is action or e.action == action), None)
            if edge is not None and edge.children:
                stash = {k: n for k, n in edge.children.items()
                         if isinstance(k, int)}   # skip error sentinel
                if stash:
                    with self._lock:
                        self._reuse[game_label] = stash
        # Record whether this decision left a pending training target, so a
        # fog-bounce retry can undo exactly this decision (full moves append
        # a pending state; fast playout-cap moves don't). decision_step was
        # advanced above regardless.
        with self._lock:
            self._last_recorded[game_label] = full_move
        return action

    def observe(self, game_label: str, side: int, reward: float,
                done: bool = False) -> None:
        """MCTS ignores per-step shaping rewards. We only act on
        the rollout loop's terminal-observe pulse to know when the
        game is over for *this side* — but the actual MCTSExperience
        building happens in `finalize_game`, which needs the winner
        (not just per-side done flags). Treat this as a no-op."""
        # Intentionally empty.

    def finalize_game(self, game_label: str, winner: int,
                      final_gs: Optional[GameState] = None,
                      midgame: bool = False) -> None:
        """Drain `_pending[game_label]` into `_queue` with one
        `MCTSExperience` per recorded state. `winner == 0` means
        draw/timeout: z is 0, or the material tiebreaker score of
        the FINAL state when `MCTSConfig.draw_tiebreak` is set --
        the same function the search applies at turn-cap terminals,
        so targets and search horizon agree. `final_gs` is the
        game's actual end state (passed by the rollout loop); if
        absent we fall back to the last recorded pre-action state,
        which trails the true final state by one action."""
        with self._lock:
            states = self._pending.pop(game_label, [])
            self._reuse.pop(game_label, None)
        tiebreak = self._mcts_config.draw_tiebreak
        if winner == 0 and tiebreak is not None and final_gs is None \
                and states:
            final_gs = states[-1].gs
            log.debug(
                f"finalize_game({game_label!r}): no final_gs passed; "
                f"tiebreak scored on the last pre-action state")
        # Auxiliary margin target (KataGo §3.5): the final MATERIAL
        # margin from each state's side, computed from the game's end
        # state for EVERY outcome (denser than z). Needs the tiebreak
        # weights + a final state; falls back to the last recorded
        # state if the rollout didn't pass `final_gs`. `None` when no
        # tiebreak config is set (trainer then skips the aux loss).
        aux_gs = final_gs if final_gs is not None else (
            states[-1].gs if states else None)
        # Moves-left targets (Lc0-style, 2026-07-04): fraction of the
        # turn budget still to be played from each recorded state,
        # derived from the game's actual end turn. Normalized by a
        # fixed constant (not per-game max_turns) so "0.1 left" means
        # the same wall distance on every map.
        end_turn_no = (aux_gs.global_info.turn_number
                       if aux_gs is not None else None)
        exps: List[MCTSExperience] = []
        # Per-game gradient normalization (2026-07-12): each game
        # contributes equal total weight regardless of length.
        # HUMAN-DERIVED games (midgame starts) get a weight FLOOR
        # (user decision 2026-07-12, Fable review M-1): a continuation
        # cut near the end may record 1-3 states whose decisive label
        # mostly credits the HUMAN's play, not ours -- without a floor
        # a single such state can transiently own most of a minibatch.
        # gw = 1/max(n, 8): games shorter than 8 states contribute
        # n/8 of a full game; pure self-play keeps exact equal-per-
        # game weighting (floor 1).
        floor = MIDGAME_GW_FLOOR if midgame else 1
        gw = 1.0 / max(floor, len(states))
        for i, s in enumerate(states):
            if winner == 0:
                if (self._train_draw_tiebreak and tiebreak is not None
                        and final_gs is not None):
                    z = draw_tiebreak_z(final_gs, s.side, tiebreak)
                else:
                    z = 0.0     # honest draw label (see __init__)
            elif winner == s.side:
                z = +1.0
            else:
                z = -1.0
            # Aux target = NEXT recorded state's margin (2026-07-12,
            # was: the game's FINAL margin). One-step material
            # prediction directly credits captures/kills the moment
            # they happen -- and since the next recorded state may
            # follow the opponent's reply, it teaches captures that
            # HOLD. The last state falls back to the true final state.
            nxt_gs = (states[i + 1].gs if i + 1 < len(states)
                      else (aux_gs if aux_gs is not None else s.gs))
            aux = (material_margin(nxt_gs, s.side, tiebreak)
                   if tiebreak is not None else None)
            ml = None
            if end_turn_no is not None:
                remaining = max(0, end_turn_no
                                - s.gs.global_info.turn_number)
                ml = min(1.0, remaining / MOVES_LEFT_NORM_TURNS)
            exps.append(MCTSExperience(
                game_state=s.gs,
                visit_counts=s.visit_counts,
                z=z,
                aux_target=aux,
                moves_left_target=ml,
                decision_step=s.decision_step,
                game_weight=gw,
            ))
        # Holdout diversion: while the probe set is below target, the
        # WHOLE game goes there instead of training (states within one
        # game are correlated; splitting a game between train and
        # holdout would leak). Same code path the actor-pool drain
        # uses for its per-game _R_EXPS payloads.
        diverted = self.offer_holdout_game(exps)
        if not diverted:
            with self._lock:
                self._queue.extend(exps)
        if states:
            log.debug(
                f"finalize_game({game_label!r}, winner={winner}): "
                f"{'HELD OUT' if diverted else 'queued'} "
                f"{len(states)} MCTSExperiences"
            )

    def drop_pending(self, game_label: str) -> None:
        """Match TransformerPolicy's drop_pending API for error
        recovery: the rollout loop calls it when a game errors mid-
        rollout. We just drop pending without queuing."""
        with self._lock:
            self._pending.pop(game_label, None)
            self._reuse.pop(game_label, None)
            self._last_recorded.pop(game_label, None)

    def drop_last_pending(self, game_label: str) -> bool:
        """Undo the most recent select_action for `game_label` after a
        fog-bounce rejection: pop the pending training target it recorded
        (if it was a full move) and roll back the decision_step increment
        it made, so the rejected pick is neither trained on nor counted
        toward the combat-oracle anneal. Also drop the now-stale tree-reuse
        stash for the bounced action. Returns True (the rejection was
        handled) so the rollout loop knows not to fall back to observe().

        MUST be called BEFORE the re-decision so the rolled-back
        decision_step is re-consumed by the retry (no double-count).
        """
        with self._lock:
            recorded = self._last_recorded.pop(game_label, False)
            if recorded:
                pend = self._pending.get(game_label)
                if pend:
                    pend.pop()
            self._reuse.pop(game_label, None)
        # Roll back the per-decision counter the bounced call advanced.
        with self._base._lock:
            if self._base._decision_step > 0:
                self._base._decision_step -= 1
        return True

    def offer_holdout_game(self, exps: List[MCTSExperience]) -> bool:
        """Offer ONE GAME's experiences to the holdout probe. Returns
        True if the game was diverted (caller must NOT train on it),
        False if the probe is off or already full. Used by the
        actor-pool drain loop, whose per-game _R_EXPS messages
        preserve exactly the game granularity the probe needs."""
        if not exps:
            return False
        with self._lock:
            if (self._holdout_target <= 0
                    or len(self._holdout) >= self._holdout_target):
                return False
            # Cap this game's contribution (random sample, not a
            # prefix -- a prefix would over-sample openings) so the
            # probe spans >= holdout_target/cap different games. The
            # game's remaining states are dropped, not trained: the
            # game-level train/holdout split stays leak-free. The
            # freeze check stays per-game (may overshoot by < cap),
            # preserving the pool-drain contract.
            take = min(self._holdout_per_game_cap, len(exps))
            self._holdout.extend(self._holdout_rng.sample(exps, take))
            self._holdout_games += 1
            full = len(self._holdout) >= self._holdout_target
        if full:
            log.info(
                f"holdout set full: {len(self._holdout)} experiences "
                f"sampled from {self._holdout_games} games "
                f"(cap {self._holdout_per_game_cap}/game); "
                f"all further games train.")
        return True

    def save_holdout(self, path) -> bool:
        """Persist the holdout probe to `path` (atomic tmp+replace).

        Why (2026-07-18): the probe used to be resampled at every
        supervisor relaunch, so each session's holdout CE sat on a
        DIFFERENT 512-state set -- levels jumped 0.44<->0.88 across
        restarts and the capacity question ("is the value head
        saturating?") was unanswerable from the logs. Persisting the
        set beside the campaign checkpoint makes holdout CE one
        continuous comparable curve across restarts.

        Saves whatever has been collected (a partial set resumes
        collecting after load). Returns True if written.
        """
        import os
        import pickle
        from pathlib import Path as _P
        with self._lock:
            if self._holdout_target <= 0 or not self._holdout:
                return False
            payload = {
                "experiences": list(self._holdout),
                "games": self._holdout_games,
                "target": self._holdout_target,
            }
        path = _P(path)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        log.info(f"holdout probe saved: {len(payload['experiences'])} "
                 f"experiences from {payload['games']} games -> {path}")
        return True

    def maybe_persist_holdout(self) -> None:
        """Save the probe to the path stashed by sim_self_play
        whenever it has grown since the last save (a few ~25MB writes
        early in a campaign, then none once frozen). Crash-safe: the
        historical failure mode is sessions dying within 1-3
        iterations, so partial sets are saved too and resume
        collecting after relaunch."""
        path = getattr(self, "_holdout_persist_path", None)
        if path is None or self._holdout_target <= 0:
            return
        with self._lock:
            n = len(self._holdout)
        if n > getattr(self, "_holdout_saved_n", 0):
            if self.save_holdout(path):
                self._holdout_saved_n = n

    def load_holdout(self, path) -> bool:
        """Restore a persisted holdout probe. Returns True on success;
        on any failure (missing/corrupt/foreign file) logs and leaves
        the fresh-sampling behavior untouched. The stored TARGET does
        not override the configured one: a loaded partial set keeps
        collecting up to the current --holdout-size."""
        import pickle
        from pathlib import Path as _P
        path = _P(path)
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            exps = list(payload["experiences"])
            games = int(payload.get("games", 0))
        except Exception as e:  # noqa: BLE001 -- any corrupt file
            log.warning(f"holdout probe load failed ({path}): "
                        f"{type(e).__name__}: {e}; sampling fresh")
            return False
        with self._lock:
            self._holdout = exps
            self._holdout_games = games
        log.info(f"holdout probe restored: {len(exps)} experiences "
                 f"from {games} games ({path}); "
                 f"{'FROZEN' if len(exps) >= self._holdout_target else 'still collecting'}")
        return True

    def holdout_metrics(self) -> Optional[Tuple[float, int]]:
        """(value CE on the frozen holdout set, holdout size), or None
        when the probe is off or still collecting. Evaluated with the
        CURRENT training net, no gradients; comparable to the logged
        train value loss (game-weighted mean; the train term
        additionally applies draw_value_weight -- see
        Trainer.eval_value_loss)."""
        with self._lock:
            if (self._holdout_target <= 0
                    or len(self._holdout) < self._holdout_target):
                return None
            probe = list(self._holdout)
        loss = self._base._trainer.eval_value_loss(probe)
        # Cached for the main loop's holdout-stall tripwire, which
        # runs after run_iteration() has already consumed the metric
        # for its log line / CSV row (avoids a second full eval).
        self.last_holdout_loss = loss
        return (loss, len(probe))

    def train_step(self) -> TrainStats:
        with self._lock:
            batch = self._queue
            self._queue = []
        rc = self._replay_config
        if not rc.enabled:
            # Legacy: one gradient step over this iteration's fresh
            # experiences, then discard. Byte-for-byte the old path.
            if not batch:
                return TrainStats()
            result = self._base._trainer.step_mcts(batch)
            self._sync_inference_weights()
            return result

        # --- Experience replay + multi-epoch (default-off) -----------
        # Pre-update generalization probe: value CE on (a sample of)
        # THIS iteration's fresh games before any gradient step
        # touches them. Distribution-matched, no training data lost
        # (unlike the frozen holdout, which anchors to relaunch-era
        # games and drifts off-distribution). ~1 forward pass over
        # <=256 states.
        nan = float("nan")
        fresh = {"ce": nan, "ce_std": nan, "pred_entropy": nan,
                 "marginal_ce_floor": nan}
        # getattr-guarded: instrumentation only -- trainer test stubs
        # (and any custom trainer) without eval_value_metrics just
        # skip the probe rather than break training.
        _eval = getattr(self._base._trainer, "eval_value_metrics", None)
        if batch and _eval is not None:
            probe = (batch if len(batch) <= 256
                     else self._replay_rng.sample(batch, 256))
            fresh = _eval(probe)
            # Decisive-only variant: with draw_value_weight=0 the head
            # is deliberately not trained to predict z=0 states, so
            # the pooled CE is structurally inflated by them
            # (2026-07-10). CE on the +-1 subset is the gate metric
            # under that recipe.
            dec = [e for e in probe if abs(e.z) >= 0.999]
            fresh["decisive_ce"] = (_eval(dec)["ce"] if len(dec) >= 16
                                    else float("nan"))

        # Add the fresh experiences to the bounded buffer, then take
        # several minibatch gradient steps sampled from it. This gives
        # the value head the many steps it needs to converge (see
        # ReplayConfig) instead of one-and-discard.
        self._replay.extend(batch)
        if len(self._replay) < rc.min_size:
            # Warm-up: not enough history yet -- mirror legacy (one
            # pass over the fresh batch) so early iters aren't wasted.
            if not batch:
                return TrainStats()
            result = self._base._trainer.step_mcts(batch)
            self._sync_inference_weights()
            self._attach_fresh_metrics(result, fresh)
            self._attach_z_composition(result, batch)
            return result

        pool = list(self._replay)
        n = len(pool)
        mb = min(rc.minibatch, n)
        stats: List[TrainStats] = []
        for _ in range(rc.updates_per_iter):
            sample = self._replay_rng.sample(pool, mb)
            stats.append(self._base._trainer.step_mcts(sample))
        self._sync_inference_weights()
        combined = self._combine_stats(stats, buffer_size=n)
        self._attach_fresh_metrics(combined, fresh)
        self._attach_z_composition(combined, batch)
        return combined

    @staticmethod
    def _attach_fresh_metrics(stats: TrainStats, fresh: Dict) -> None:
        stats.fresh_value_ce = fresh["ce"]
        stats.fresh_ce_std = fresh.get("ce_std", float("nan"))
        stats.fresh_pred_entropy = fresh["pred_entropy"]
        stats.fresh_ce_floor = fresh["marginal_ce_floor"]
        stats.fresh_decisive_ce = fresh.get("decisive_ce", float("nan"))

    @staticmethod
    def _attach_z_composition(stats: TrainStats, batch) -> None:
        """Target-composition of this iteration's incoming
        experiences, in TWO normalizations:

        - z_*_frac: raw transition census (the 2026-07-10
          'draw-spike' diagnosis). NOT a gradient metric: long games
          inflate it in proportion to their length.
        - z_*_frac_w: game_weight-weighted -- each game contributes
          its total gw (=1 for non-midgame), matching the gradient
          contribution step_mcts's per-game normalization actually
          produces. THIS is the column to read for 'what is the
          value head training on' (2026-07-22: the unweighted
          census at 0.19 draws was misread as 20% of the gradient
          when the weighted share was ~5%)."""
        n = len(batch)
        if not n:
            return
        stats.z_win_frac = sum(1 for e in batch if e.z >= 0.999) / n
        stats.z_loss_frac = sum(1 for e in batch if e.z <= -0.999) / n
        stats.z_draw_frac = 1.0 - stats.z_win_frac - stats.z_loss_frac
        tw = ww = lw = 0.0
        for e in batch:
            gw = float(getattr(e, "game_weight", 1.0))
            tw += gw
            if e.z >= 0.999:
                ww += gw
            elif e.z <= -0.999:
                lw += gw
        if tw > 0:
            stats.z_win_frac_w = ww / tw
            stats.z_loss_frac_w = lw / tw
            stats.z_draw_frac_w = 1.0 - (ww + lw) / tw

    def _sync_inference_weights(self) -> None:
        """Propagate the freshly-updated `_model` weights into the
        inference copy that MCTS actually searches with.

        CRITICAL: `step_mcts` lands gradients on `self._base._model`,
        but `select_action` / leaf expansion run on the SEPARATE
        `self._base._inference_model` (see `__init__`: `_inference_model
        = base._inference_model`). `TransformerPolicy.train_step` snapshots
        after every gradient step for exactly this reason, but the MCTS
        path calls `_trainer.step_mcts` directly and bypasses it -- so
        without this call the self-play / search network would stay
        frozen at warm-start weights for the entire run while only the
        saved checkpoint drifts (the AlphaZero loop would never close).
        `_snapshot_inference_weights` briefly takes `_base._lock`, so a
        concurrent worker `select_action` either waits ~ms on the
        load_state_dict swap or runs on the previous snapshot."""
        self._base._snapshot_inference_weights()

    @staticmethod
    def _combine_stats(stats: List[TrainStats],
                       buffer_size: int) -> TrainStats:
        """Aggregate the multi-epoch minibatch steps into one TrainStats
        for the iter log: losses/grad-norm are MEANED over the updates
        (representative of the pass), transitions SUMMED (total samples
        touched), grad_norm uses the LAST step (post-update magnitude)."""
        if not stats:
            return TrainStats()
        k = len(stats)
        return TrainStats(
            value_signal_states=sum(
                getattr(s, "value_signal_states", 0) for s in stats),
            policy_loss=sum(s.policy_loss for s in stats) / k,
            value_loss=sum(s.value_loss for s in stats) / k,
            entropy=sum(s.entropy for s in stats) / k,
            total_loss=sum(s.total_loss for s in stats) / k,
            grad_norm=stats[-1].grad_norm,
            mean_return=sum(s.mean_return for s in stats) / k,
            n_transitions=sum(s.n_transitions for s in stats),
            n_trajectories=buffer_size,
            aux_loss=sum(s.aux_loss for s in stats) / k,
            moves_left_loss=sum(s.moves_left_loss for s in stats) / k,
        )

    # ------------------------------------------------------------------
    # Checkpoint forwarding
    # ------------------------------------------------------------------

    def save_checkpoint(self, path) -> None:
        return self._base.save_checkpoint(path)

    def load_checkpoint(self, path, *, strict: bool = False) -> None:
        return self._base.load_checkpoint(path, strict=strict)
