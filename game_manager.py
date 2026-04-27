"""Coordinates Wesnoth subprocesses + policy + per-step reward feeding.

Per game (per side's turn-local view):

    [read state_t] → [attribute reward to prev action] → [policy picks
     action_t (= internally records a pending Transition for side_t)]
     → [send action_t to Lua] → loop

At end of game we finalize: both sides' last pending transitions get a
terminal reward (win/loss/draw) with `done=True`, which flushes them
into the policy's training queue.

Reward plumbing is only active when the policy is trainable (exposes
an ``observe`` method). Scripted policies (DummyPolicy) don't care.

After a batch of games completes, if the policy is trainable and has
`train_step`, we call it. It drains the queue and runs one gradient
update. Checkpoints are saved on a schedule (CHECKPOINT_FREQUENCY) and
delegated to the policy's own save_checkpoint method.
"""

import asyncio
import logging
import statistics
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

from classes import GameState
from constants import (
    CHECKPOINT_FREQUENCY,
    CHECKPOINTS_PATH,
    LOG_FREQUENCY,
    LOGS_PATH,
    MAX_ACTIONS_PER_GAME,
    NUM_PARALLEL_GAMES,
    REPLAYS_PATH,
    SCENARIOS_PATH,
)
from policy import Policy, get_policy
from profiling import Profiler
from rewards import (
    OUTCOME_DRAW,
    OUTCOME_LOSS,
    OUTCOME_ONGOING,
    OUTCOME_TIMEOUT,
    OUTCOME_WIN,
    StepDelta,
    WeightedReward,
    compute_delta,
)
from state_converter import StateConverter
from wesnoth_interface import WesnothGame


# Per-label state we carry between turns to compute StepDeltas. Tuple:
#   (prev_state, prev_action_type, prev_acting_side, prev_recruit_cost)
_PrevEntry = Tuple[GameState, str, int, int]


# Fields accumulated per-side per-game from StepDelta. Every name here
# also appears in _log_stats' rolling aggregate, so keep the order stable
# (report order follows this list). Adding a field = bump it from the
# delta in _record_delta_stats and widen the format strings below.
_PER_SIDE_FIELDS = (
    'actions',           # count of Python-side decisions this side made
    'moves', 'attacks', 'recruits', 'end_turns',
    'invalid_actions',   # subset of actions that produced no state change
    'enemy_hp_lost',     # HP we took off enemies
    'our_hp_lost',       # HP enemies took off us
    'enemy_gold_lost',   # gold-cost of enemy units we killed
    'our_gold_lost',     # gold-cost of our units that died
    'recruit_gold',      # gold we spent recruiting (successful only)
    'villages_gained', 'villages_lost',
    'leader_moves',
    # Sum of min_enemy_distance across the side's actions. The per-game
    # summary divides by `actions` to report mean; lower = better.
    'enemy_distance_sum',
)


def _new_side_counter() -> Dict[str, int]:
    return {k: 0 for k in _PER_SIDE_FIELDS}


class GameManager:
    """Coordinates a batch of Wesnoth subprocesses sharing a single policy."""

    def __init__(
        self,
        num_games: int = NUM_PARALLEL_GAMES,
        policy: Optional[Policy] = None,
        reward_fn=None,
    ):
        self.num_games = num_games
        self.games: Dict[str, WesnothGame] = {}

        # Shared state converter — one mapping from unit-type-name to an
        # integer ID, used consistently across games.
        self.global_converter = StateConverter()

        # Serializes Wesnoth launches so each WesnothGame gets a clean
        # pre-launch-logs snapshot; otherwise concurrent launches all
        # see the same "new" log files and adopt the wrong one. Held
        # briefly — just long enough for Wesnoth to start writing its
        # log so the next launch's snapshot can see it as "existing".
        self._launch_lock = asyncio.Lock()

        self.logger = logging.getLogger("game_manager")
        self._setup_logging()

        self.policy: Policy = policy if policy is not None else get_policy("dummy")
        self.logger.info(f"Policy: {type(self.policy).__name__}")

        # Default reward fn is WeightedReward with its sensible defaults
        # (gold_killed_delta at 0.01, village_delta at 0.05, small
        # damage/recruit bonuses, ±1 terminal). Callers can pass any
        # RewardFn-compatible callable to override — the Phase 3 user
        # ask for "incentivize unorthodox strategies" flows through
        # construction with different weights, no other code touched.
        self._reward_fn = reward_fn if reward_fn is not None else WeightedReward()

        # Per-(game_label) state needed to build StepDeltas between
        # consecutive frames. See _PrevEntry type alias.
        self._prev: Dict[str, _PrevEntry] = {}

        # Per-game, per-side stat accumulators. Populated from each
        # StepDelta via _record_delta_stats, consumed when the game
        # ends. Shape: {label: {'turn': int, 1: {...}, 2: {...}}}.
        self._game_counters: Dict[str, Dict[Any, Any]] = {}

        # Rolling window of finished-game summaries for the periodic
        # Stats block. Each entry:
        #   {'turn': int, 'outcome': str, 1: {...}, 2: {...}}
        # maxlen chosen so averages are stable over ~10× LOG_FREQUENCY.
        self._recent_games: Deque[Dict[str, Any]] = deque(maxlen=100)

        # Rolling-pool bookkeeping. These track when we last fired a
        # train_step / checkpoint so the pool's event loop can fire
        # them at a schedule independent of batch boundaries.
        self._last_trained_at = 0
        self._last_checkpoint_at = 0

        # Handle on the currently-running train_step (if any). train_step
        # takes ~90s; running it on the event-loop thread blocks every
        # rollout for those 90s (no new Wesnoth launches, queued games
        # stalled). Instead we launch it on a worker via asyncio.to_thread
        # and let the event loop keep pumping. See _run_train_step_async
        # and TransformerPolicy.train_step for the concurrency contract.
        self._train_task: Optional[asyncio.Task] = None

        # Profiling. handle_game_turn wraps its stages in
        # `with self._profiler.time("..."): ...` and the periodic
        # stats log emits self._profiler.report().
        self._profiler = Profiler()

        self.stats = {
            'games_completed': 0,
            'total_actions': 0,
            'wins_side1': 0,
            'wins_side2': 0,
            'draws': 0,
            'timeouts': 0,
            'state_conversion_errors': 0,
            'action_send_errors': 0,
        }

        for p in (CHECKPOINTS_PATH, REPLAYS_PATH, LOGS_PATH):
            p.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> None:
        log_file = LOGS_PATH / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

    # ------------------------------------------------------------------
    # Game lifecycle
    # ------------------------------------------------------------------

    async def create_game(self, label: str) -> WesnothGame:
        """Launch a Wesnoth process for this game.

        Takes self._launch_lock so the pre-launch-logs snapshot taken
        inside start_wesnoth doesn't overlap with another launch. We
        also wait briefly for the new .out.log to appear before
        releasing the lock, so the NEXT game's snapshot sees our log
        as "existing".
        """
        self.logger.info(f"Creating game {label}")
        scenario_path = SCENARIOS_PATH / "training_scenario.cfg"
        game = WesnothGame(label, scenario_path)

        async with self._launch_lock:
            game.start_wesnoth()
            # Wait up to ~5s for Wesnoth to create its .out.log file,
            # then lock in the path so later reads don't re-probe
            # (re-probing would include OTHER games' subsequently-
            # created logs in the "new" set and assign the wrong one).
            for _ in range(50):
                candidate = game._find_out_log()
                if candidate is not None:
                    game._log_path = candidate
                    self.logger.debug(
                        f"Game {label}: .out.log = {candidate.name}")
                    break
                await asyncio.sleep(0.1)
        return game

    async def handle_game_turn(self, game: WesnothGame) -> bool:
        """One state→action exchange. Returns False when the game ends."""
        label = game.label
        try:
            # read_state spins in a polling loop with time.sleep, which
            # blocks the asyncio event loop. Push it to a thread so
            # parallel games can actually run in parallel — otherwise
            # asyncio.gather over N game tasks just serializes them.
            with self._profiler.time("read_state"):
                state_payload = await asyncio.to_thread(game.read_state)
            if not state_payload:
                self.logger.warning(f"Game {label}: no state, ending game")
                return False

            try:
                with self._profiler.time("parse_state"):
                    game_state = self.global_converter.convert_payload_to_game_state(state_payload)
            except Exception as e:
                self.logger.error(
                    f"Game {label}: state parse failed: {e}", exc_info=True)
                self.stats['state_conversion_errors'] += 1
                preview = state_payload[:400] if state_payload else "<empty>"
                self.logger.debug(f"Payload preview: {preview}")
                return False

            if game.game_id is None:
                game.adopt_game_id(game_state.game_id)

            # Attribute an intermediate reward to the previous action,
            # if any. Terminal rewards are applied in _finalize_game.
            with self._profiler.time("reward_delta"):
                self._reward_previous_step(label, game_state)

            if game_state.game_over:
                winner = game_state.winner or 0
                self.logger.info(f"Game {label}: over, winner={winner}")
                self._finalize_game(label, game_state, winner, OUTCOME_WIN)
                self._record_game_result(game, winner)
                game.game_over = True
                game.winner = winner
                return False

            # Policy forward + action send.
            try:
                with self._profiler.time("policy_select"):
                    internal_action = self.policy.select_action(
                        game_state, game_label=label
                    )
            except Exception as e:
                self.logger.error(f"Policy error: {e}", exc_info=True)
                internal_action = {'type': 'end_turn'}

            wire_action = self.global_converter.convert_action_to_json(internal_action)

            # Stash what we need to compute this step's delta next time
            # we see a state frame.
            self._prev[label] = (
                game_state,
                internal_action.get('type', ''),
                game_state.global_info.current_side,
                _recruit_cost(internal_action, game_state),
            )

            with self._profiler.time("send_action"):
                sent = game.send_action(wire_action)
            if not sent:
                # Non-fatal: Lua's 30s action timeout will end the side's
                # turn and the game proceeds. We lose one action of
                # learning signal but not the whole game.
                self.logger.warning(f"Game {label}: action send failed (turn will time out)")
                self.stats['action_send_errors'] += 1
            else:
                self.stats['total_actions'] += 1
            return True

        except Exception as e:
            self.logger.error(f"Game {label}: turn error: {e}", exc_info=True)
            return False

    async def run_game(self, label: str) -> None:
        game: Optional[WesnothGame] = None
        natural_over = False
        outcome_label = 'unknown'
        # Initialize per-game stat counters up front; _record_delta_stats
        # writes into these as each turn closes, _summarize_game drains.
        self._game_counters[label] = {
            'turn': 0, 1: _new_side_counter(), 2: _new_side_counter(),
        }
        try:
            game = await self.create_game(label)
            self.games[label] = game
            self.logger.info(f"Game {label}: started")

            actions = 0
            while actions < MAX_ACTIONS_PER_GAME:
                if not await self.handle_game_turn(game):
                    # Natural game-over was already counted by
                    # _record_game_result in handle_game_turn and set
                    # game.game_over. Anything else (no state, send
                    # error, parse error) needs counting here.
                    natural_over = bool(game and game.game_over)
                    break
                actions += 1

            if natural_over:
                # Already counted; derive an outcome label from winner.
                winner = game.winner if game is not None else 0
                outcome_label = (
                    f'win_s{winner}' if winner in (1, 2) else 'draw'
                )
            elif actions >= MAX_ACTIONS_PER_GAME:
                self.logger.warning(
                    f"Game {label}: timeout after {actions} actions")
                prev = self._prev.get(label)
                if prev is not None:
                    self._finalize_game(
                        label, prev[0], winner=0,
                        default_outcome=OUTCOME_TIMEOUT,
                    )
                else:
                    self._drop_trajectories(label)
                self._count_terminated('timeouts')
                outcome_label = 'timeout'
            else:
                # Mid-episode exit (no-state / send fail / ...).
                # Outcome unknown; drop trajectories rather than seal
                # with a fake reward. Still bump the counter so
                # train_step fires on schedule.
                self._drop_trajectories(label)
                self._count_terminated('interrupted')
                outcome_label = 'interrupted'

        except Exception as e:
            self.logger.error(f"Game {label}: error: {e}", exc_info=True)
            self._drop_trajectories(label)
            self._count_terminated('errored')
            outcome_label = 'errored'
        finally:
            if label in self.games:
                # Drop the state-converter's cached static map for this
                # game_id so long runs don't accumulate dead caches.
                gid = self.games[label].game_id
                if gid:
                    self.global_converter.forget_game(gid)
                self.games[label].terminate()
                del self.games[label]
            self._prev.pop(label, None)
            self._summarize_game(label, outcome_label)
            self.logger.info(f"Game {label}: finished")

    def _count_terminated(self, bucket: str) -> None:
        """Bump games_completed + the relevant stats bucket, and trigger
        a periodic stats log. Used for all non-natural game endings
        (timeouts, mid-episode exits, errors). Natural game-overs
        continue to go through _record_game_result."""
        self.stats['games_completed'] += 1
        self.stats.setdefault(bucket, 0)
        self.stats[bucket] += 1
        if self.stats['games_completed'] % LOG_FREQUENCY == 0:
            self._log_stats()

    # ------------------------------------------------------------------
    # Reward plumbing
    # ------------------------------------------------------------------

    def _reward_previous_step(
        self, label: str, game_state: GameState,
    ) -> None:
        """If there is a pending previous action for this game, compute
        its intermediate (non-terminal) delta and hand the reward to the
        policy's observe hook (if any)."""
        prev = self._prev.get(label)
        if prev is None:
            return
        prev_state, prev_action_type, prev_side, prev_recruit_cost = prev
        delta = compute_delta(
            prev_state, game_state, prev_action_type,
            recruit_cost=prev_recruit_cost, outcome=OUTCOME_ONGOING,
        )
        reward = self._reward_fn(delta)
        self._observe_policy(label, prev_side, reward, done=False)
        self._record_delta_stats(label, delta)

    def _record_delta_stats(self, label: str, delta: StepDelta) -> None:
        """Fold one per-step delta into the game's running stat counters.

        Drives both the per-game end-of-game log line and the rolling
        per-side averages in the periodic Stats block. The goal is
        diagnostic: when mean_return plateaus we want to see WHICH
        reward component is paying the bill — combat, villages, or
        recruits — and whether it's symmetric across sides.
        """
        counters = self._game_counters.get(label)
        if counters is None:
            return
        if delta.turn > counters['turn']:
            counters['turn'] = delta.turn
        s = counters.get(delta.side)
        if s is None:
            return
        s['actions'] += 1
        atype = delta.action_type
        if atype == 'move':
            s['moves'] += 1
        elif atype == 'attack':
            s['attacks'] += 1
        elif atype == 'recruit':
            s['recruits'] += 1
        elif atype == 'end_turn':
            s['end_turns'] += 1
        s['enemy_hp_lost']   += delta.enemy_hp_lost
        s['our_hp_lost']     += delta.our_hp_lost
        s['enemy_gold_lost'] += delta.enemy_gold_lost
        s['our_gold_lost']   += delta.our_gold_lost
        s['recruit_gold']    += delta.unit_recruited_cost
        s['villages_gained'] += delta.villages_gained
        s['villages_lost']   += delta.villages_lost
        if delta.leader_moved:
            s['leader_moves'] += 1
        if delta.invalid_action:
            s['invalid_actions'] += 1
        s['enemy_distance_sum'] += delta.min_enemy_distance

    def _summarize_game(self, label: str, outcome: str) -> None:
        """Emit a one-line-per-game summary and archive it for the
        rolling stats window. Called from run_game's finally block
        regardless of how the game ended."""
        counters = self._game_counters.pop(label, None)
        if counters is None:
            return
        s1, s2 = counters[1], counters[2]
        def fmt_side(s):
            avg_dist = (s['enemy_distance_sum'] / s['actions']) if s['actions'] else 0.0
            return (
                f"acts={s['actions']}(inv={s['invalid_actions']}) "
                f"mv={s['moves']} atk={s['attacks']} "
                f"rec={s['recruits']}({s['recruit_gold']}g) "
                f"dmg={s['enemy_hp_lost']}HP kill={s['enemy_gold_lost']}g "
                f"took={s['our_hp_lost']}HP lost={s['our_gold_lost']}g "
                f"vil+{s['villages_gained']}/-{s['villages_lost']} "
                f"ldr={s['leader_moves']} dist={avg_dist:.1f}"
            )
        self.logger.info(
            f"Game {label} done: turn={counters['turn']} outcome={outcome} "
            f"| s1 {fmt_side(s1)} "
            f"| s2 {fmt_side(s2)}"
        )
        self._recent_games.append({
            'turn': counters['turn'],
            'outcome': outcome,
            1: dict(s1),
            2: dict(s2),
        })

    def _finalize_game(
        self,
        label: str,
        game_state: GameState,
        winner: int,
        default_outcome: str = OUTCOME_WIN,  # unused — see body
    ) -> None:
        """Seal both sides' trajectories with the appropriate terminal
        outcome. `winner` is 0 for draw/timeout/endless, 1/2 otherwise.
        `default_outcome` kept for signature symmetry; outcome per side
        is always derived from `winner` here (OUTCOME_TIMEOUT if
        default_outcome == OUTCOME_TIMEOUT)."""
        for side in (1, 2):
            if default_outcome == OUTCOME_TIMEOUT:
                outcome = OUTCOME_TIMEOUT
            elif winner == 0:
                outcome = OUTCOME_DRAW
            elif winner == side:
                outcome = OUTCOME_WIN
            else:
                outcome = OUTCOME_LOSS
            terminal = StepDelta(
                side=side,
                turn=game_state.global_info.turn_number,
                action_type='terminal',
                outcome=outcome,
            )
            reward = self._reward_fn(terminal)
            self._observe_policy(label, side, reward, done=True)

    def _drop_trajectories(self, label: str) -> None:
        """Error path: discard pending transitions for this game."""
        dropper = getattr(self.policy, 'drop_pending', None)
        if dropper is not None:
            dropper(label)

    def _observe_policy(
        self, label: str, side: int, reward: float, done: bool,
    ) -> None:
        observer = getattr(self.policy, 'observe', None)
        if observer is None:
            return
        try:
            observer(label, side, reward, done=done)
        except Exception as e:
            self.logger.error(f"policy.observe error: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Stats + training loop
    # ------------------------------------------------------------------

    def _record_game_result(self, game: WesnothGame, winner: Optional[int]) -> None:
        self.stats['games_completed'] += 1
        if winner == 1:
            self.stats['wins_side1'] += 1
        elif winner == 2:
            self.stats['wins_side2'] += 1
        else:
            self.stats['draws'] += 1

        if self.stats['games_completed'] % LOG_FREQUENCY == 0:
            self._log_stats()

    def _log_stats(self) -> None:
        total = self.stats['games_completed']
        if total == 0:
            return
        self.logger.info("=" * 70)
        self.logger.info(f"Stats @ {total} games:")
        self.logger.info(
            f"  wins S1={self.stats['wins_side1']} "
            f"S2={self.stats['wins_side2']} "
            f"draws={self.stats['draws']} "
            f"timeouts={self.stats['timeouts']}")
        self.logger.info(f"  total actions: {self.stats['total_actions']}")
        self.logger.info(
            f"  errors: parse={self.stats['state_conversion_errors']} "
            f"send={self.stats['action_send_errors']}")
        self.logger.info("  " + self._profiler.throughput(total))
        self.logger.info("  per-stage:\n" + self._profiler.report())
        self._log_recent_game_stats()
        self.logger.info("=" * 70)

    def _log_recent_game_stats(self) -> None:
        """Rolling per-game averages over the last _recent_games window.

        Diagnostic: answers "is the policy fighting, recruiting, or
        ignoring combat?". Each field in _PER_SIDE_FIELDS gets a mean
        across the window; turn counts additionally get min/max so
        we can spot whether games are converging in length.
        """
        if not self._recent_games:
            return
        n = len(self._recent_games)
        turns = [g['turn'] for g in self._recent_games]
        self.logger.info(
            f"  per-game avg over last {n} games: "
            f"turn mean={statistics.mean(turns):.1f} "
            f"min={min(turns)} max={max(turns)}"
        )
        for side in (1, 2):
            agg: Dict[str, int] = {k: 0 for k in _PER_SIDE_FIELDS}
            for g in self._recent_games:
                for k in _PER_SIDE_FIELDS:
                    agg[k] += g[side][k]
            # Weighted avg distance: total enemy-distance summed across
            # games, divided by total actions — an approximation of the
            # typical min-enemy-distance observed per action.
            avg_dist = (agg['enemy_distance_sum'] / agg['actions']) if agg['actions'] else 0.0
            self.logger.info(
                f"    s{side}: "
                f"acts={agg['actions']/n:.1f}(inv={agg['invalid_actions']/n:.1f}) "
                f"mv={agg['moves']/n:.1f} "
                f"atk={agg['attacks']/n:.1f} "
                f"rec={agg['recruits']/n:.1f}({agg['recruit_gold']/n:.1f}g) "
                f"dmg={agg['enemy_hp_lost']/n:.1f}HP "
                f"kill={agg['enemy_gold_lost']/n:.1f}g "
                f"took={agg['our_hp_lost']/n:.1f}HP "
                f"lost={agg['our_gold_lost']/n:.1f}g "
                f"vil+{agg['villages_gained']/n:.2f}/-{agg['villages_lost']/n:.2f} "
                f"ldr={agg['leader_moves']/n:.1f} "
                f"dist={avg_dist:.1f}"
            )

    async def run_training(self) -> None:
        """Rolling pool of N in-flight games.

        Keeps self.num_games tasks running at all times. Whenever one
        finishes, immediately spawn a replacement — so the slowest game
        in a set doesn't pause the others.

        train_step and checkpoints fire on game-count schedules rather
        than at batch boundaries (there are no batches anymore).
        """
        trainable = bool(getattr(self.policy, 'trainable', False))
        train_every = self.num_games        # same cadence as old batched version
        ckpt_every = CHECKPOINT_FREQUENCY

        self.logger.info(
            f"Rolling pool: {self.num_games} in-flight games. "
            f"train_step every {train_every} games. "
            f"checkpoint every {ckpt_every} games."
        )

        # Map of label → Task, for cancellation on shutdown.
        active: Dict[str, asyncio.Task] = {}
        counter = 0

        def _spawn() -> None:
            nonlocal counter
            label = f"game_{counter}"
            counter += 1
            active[label] = asyncio.create_task(
                self.run_game(label), name=label,
            )

        try:
            for _ in range(self.num_games):
                _spawn()

            while True:
                done, _ = await asyncio.wait(
                    list(active.values()),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                finished_labels = [
                    label for label, task in active.items() if task in done
                ]
                # Drain exceptions so they're logged, not silently swallowed.
                for label in finished_labels:
                    task = active.pop(label)
                    if task.exception() is not None:
                        self.logger.error(
                            f"Game {label} raised: {task.exception()}",
                            exc_info=task.exception(),
                        )
                    _spawn()

                # Fire training and checkpointing on game-count schedules.
                self._maybe_train(train_every, trainable)
                self._maybe_checkpoint(ckpt_every, trainable)

        except asyncio.CancelledError:
            raise
        except KeyboardInterrupt:
            self.logger.info("Interrupted — cancelling in-flight games")
            for task in active.values():
                task.cancel()
            await asyncio.gather(*active.values(), return_exceptions=True)
            # Wait for any in-flight train_step to complete so we
            # checkpoint the newest weights, not the previous ones.
            # The task runs in a worker thread we can't cancel cleanly;
            # await its completion instead.
            if self._train_task is not None and not self._train_task.done():
                self.logger.info("Waiting for in-flight train_step to finish…")
                try:
                    await self._train_task
                except Exception as e:
                    self.logger.error(f"final train_step failed: {e}",
                                      exc_info=True)
            if trainable:
                self._save_checkpoint()
        finally:
            for game in list(self.games.values()):
                game.terminate()

    def _maybe_train(self, every: int, trainable: bool) -> None:
        """Fire-and-forget: schedule train_step on a worker thread if
        enough games have completed and no previous train_step is still
        running. Rollouts continue on the event-loop thread while the
        worker trains.
        """
        n = self.stats['games_completed']
        if not trainable or n == 0 or n < self._last_trained_at + every:
            return
        # Reap the previous task if it's finished — re-raise any
        # exception so we notice training failures instead of silently
        # dropping them.
        if self._train_task is not None and self._train_task.done():
            exc = self._train_task.exception()
            if exc is not None:
                self.logger.error(
                    f"train_step failed: {exc}", exc_info=exc)
            self._train_task = None

        # Still in flight? Skip this trigger; the next one (N more games
        # from now) will try again. Rollout has outrun training; that's
        # fine — the queue just keeps accumulating trajectories.
        if self._train_task is not None and not self._train_task.done():
            self.logger.info(
                f"train_step still in flight (prev started at "
                f"game {self._last_trained_at}); skipping this trigger "
                f"at game {n}. Queued trajectories will go in next step."
            )
            self._last_trained_at = n  # don't fire-spam every game
            return

        self._last_trained_at = n
        self._train_task = asyncio.create_task(self._run_train_step_async())

    async def _run_train_step_async(self) -> None:
        """Coroutine wrapper that pushes policy.train_step into a worker
        thread. Returns whatever the policy returns; exceptions propagate
        to the task and are reaped by the next _maybe_train call."""
        train_step = getattr(self.policy, 'train_step', None)
        if train_step is None:
            return
        await asyncio.to_thread(train_step)

    def _maybe_checkpoint(self, every: int, trainable: bool) -> None:
        n = self.stats['games_completed']
        if not trainable or n == 0 or n < self._last_checkpoint_at + every:
            return
        self._save_checkpoint()
        self._last_checkpoint_at = n

    def _save_checkpoint(self) -> None:
        saver = getattr(self.policy, 'save_checkpoint', None)
        if saver is None:
            return
        path = CHECKPOINTS_PATH / f"checkpoint_{self.stats['games_completed']}.pt"
        try:
            saver(path)
        except Exception as e:
            self.logger.error(f"Checkpoint save failed: {e}", exc_info=True)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _recruit_cost(action: Dict, game_state: GameState) -> int:
    """Best-effort cost lookup for a recruit action. We look up a unit
    of the same name in the current game state; if none exists, return
    0 (the reward-shaping term contributes 0 then, which is fine)."""
    if action.get('type') != 'recruit':
        return 0
    name = action.get('unit_type', '')
    for u in game_state.map.units:
        if u.name == name:
            return u.cost
    return 0


async def main():
    manager = GameManager(num_games=NUM_PARALLEL_GAMES)
    await manager.run_training()


if __name__ == "__main__":
    asyncio.run(main())
