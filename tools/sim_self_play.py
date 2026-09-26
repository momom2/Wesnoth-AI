"""Self-play training using the in-process Wesnoth simulator.

Why this exists: subprocess-based self-play (the old `game_manager.py`
plan) requires Wesnoth + IPC + Lua, none of which work reliably on
Windows or on the cluster. The Python simulator (`tools/wesnoth_sim.py`)
runs the same game logic in-process at ~1000x the speed and is
trivially cluster-portable. This module is the glue that connects
the existing TransformerPolicy (model + encoder + trainer) to
WesnothSim, drives self-play games, and applies one gradient update
per iteration: search distillation by default (`--mcts`, with turn
search), or REINFORCE with a value baseline under `--reinforce`.

The loop, per iteration:

  1. Draw N game setups: `scenario_pool.random_setup` (a random map,
     factions and leaders), or a mid-game start cut from the replay
     corpus when the category mix rolls one (`--midgame-ratio`).
  2. For each, build a WesnothSim and roll out the model against
     itself, calling `policy.select_action` for every action and
     `policy.observe` for the per-step shaping reward.
  3. On terminal, emit per-side terminal rewards (WIN/LOSS/TIMEOUT)
     so the trajectory closes cleanly.
  4. Call `policy.train_step()` to apply one gradient update across
     all queued trajectories (one per side per game).

Steps 2 and 3, one game, are `tools/selfplay_game.py`, which the actor
pool's actors play through too.

Reward function (`--reinforce` only; `--mcts` distills the terminal
outcome): the existing `WeightedReward` from rewards.py. It
diffs pre/post step GameStates -- we deepcopy the GameState before
each step so the diff is well-defined (the sim mutates in place,
swapping the units set, the sides list, and global_info attributes).
deepcopy is ~0.5ms per call; an 80-turn game with ~200 actions/side
adds ~200ms of overhead, negligible against any backward pass.

Usage:
    python tools/sim_self_play.py
        --checkpoint-in training/checkpoints/supervised_epoch3.pt
        --iterations    50
        --games-per-iter 8
        --max-turns     40
        --save-every    10
        --checkpoint-out training/checkpoints/sim_selfplay.pt

A "game" produces TWO trajectories (one per side); --games-per-iter
controls game count, not trajectory count. Set --max-turns
conservatively early on -- the dummy gets 20 turns done in <1s but a
trained policy might play full-length games.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Make project root importable when run as a script.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

from tools.scenario_pool import LADDER_SCENARIO_IDS
from wesnoth_ai.rewards import WeightedReward, load_reward_config
from wesnoth_ai.transformer_policy import TransformerPolicy
from tools.wesnoth_sim import PvPDefaults
from tools import selfplay_game
from tools.selfplay_game import (
    GameOutcome, _play_one_game_safe, _recruit_cost_lookup, _roll_max_turns, _worker_loop,
    k_median_of,
)


log = logging.getLogger("sim_self_play")


# ---------------------------------------------------------------------
# Initial-state pool
# ---------------------------------------------------------------------

# Self-play seed pool: union of the Ladder Era's three official map
# packs (Competitive + Classic + Adventurous, 21 scenarios total).
# Pulled from the single source-of-truth list in scenario_pool so
# the same canonical set drives both scenario sampling
# (`random_setup`) and replay-corpus filtering (`_is_ladder_map`).
# Wesnoth-side canonical IDs are verified case-exact against
# `wesnoth_src/data/multiplayer/scenarios/2p_*.cfg`.
_LADDER_MAP_SCENARIO_IDS: frozenset = frozenset(LADDER_SCENARIO_IDS)


def _is_ladder_map(scenario_id: str) -> bool:
    """True if `scenario_id` matches a Ladder Era map (any pack).
    Tolerates `_Ladder_Random` and `_Ladder` suffixes the Ladder Era
    add-on appends when a game is launched via its random-pool
    picker (e.g. `multiplayer_Basilisk_Ladder_Random` ->
    `multiplayer_Basilisk`)."""
    s = scenario_id
    for suffix in ("_Ladder_Random", "_Ladder"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s in _LADDER_MAP_SCENARIO_IDS


def _records_run_tag() -> str:
    from tools.validation_exports import run_tag
    return run_tag()


_GAME_LOG_RUN_ID: Optional[int] = None


def _game_log_run_id(game_log_dir: Path) -> int:
    """Stable per-PROCESS run number for game-log dirs
    (iter_MMM_NNNNNN). First call scans existing run prefixes and
    claims max+1; cached for the process lifetime. Old-format
    unprefixed dirs (iter_NNNNNN) are ignored -- they never collide
    with the new names."""
    global _GAME_LOG_RUN_ID
    if _GAME_LOG_RUN_ID is None:
        import re
        seen = [-1]
        try:
            for d in Path(game_log_dir).iterdir():
                m = re.match(r"iter_(\d{3})_\d{6}$", d.name)
                if m:
                    seen.append(int(m.group(1)))
        except OSError:
            pass
        _GAME_LOG_RUN_ID = max(seen) + 1
    return _GAME_LOG_RUN_ID




def _bundle_validation_exports(export_dir: Optional[Path],
                               iter_idx: int) -> Optional[Path]:
    """Tar all LOOSE validation-export replays into one
    per-iteration bundle under <export_dir>/bundles/ and remove the
    loose files. Full replay recording (--validate-export-every 1,
    user 2026-07-21) writes ~2k replays/day; as loose files that
    blows HF's <10k-per-folder / <100k-per-repo guidance within
    weeks, while one tar per iteration is ~40 files/day. Loose
    .bz2s are complete on arrival (atomic tmp+os.replace in the
    exporter), so tarring never sees partial files; replays landing
    mid-tar simply ride the next bundle -- bundle names are a
    sequence, not a strict game-set claim. Bundle run ids are
    claimed like game-log run ids (max existing + 1 per process)."""
    if export_dir is None:
        return None
    export_dir = Path(export_dir)
    if not export_dir.is_dir():
        return None
    loose = sorted(f for f in export_dir.rglob("*.bz2")
                   if "bundles" not in f.parts)
    if not loose:
        return None
    bdir = export_dir / "bundles"
    bdir.mkdir(parents=True, exist_ok=True)
    # Run identity = the shared UTC launch tag (2026-08-05; the old
    # claimed 3-digit counter carried no provenance and interleaved
    # unrelated runs' bundles as r000..rNNN in one directory).
    from tools.validation_exports import run_tag
    out = bdir / f"replays_r{run_tag()}_i{iter_idx:06d}.tar"
    tmp = bdir / (out.name + ".tmp")
    import tarfile
    try:
        with tarfile.open(tmp, "w") as tf:
            for f in loose:
                tf.add(f, arcname=f.relative_to(export_dir).as_posix())
        os.replace(tmp, out)
    except OSError as e:
        log.warning(f"replay bundling failed (iter {iter_idx}): {e}")
        tmp.unlink(missing_ok=True)
        return None
    for f in loose:
        f.unlink(missing_ok=True)
    log.info(f"bundled {len(loose)} replays -> {out.name}")
    return out


def _gpu_mem_mb():
    """(allocated_MB, reserved_MB) of the trainer process, or
    (None, None) off-CUDA. Reserved > allocated = allocator cache;
    reserved growth with flat allocated = fragmentation/ratchet."""
    import torch
    if not torch.cuda.is_available():
        return (None, None)
    return (round(torch.cuda.memory_allocated() / 2**20),
            round(torch.cuda.memory_reserved() / 2**20))


def _gpu_mem_peak_mb(reset: bool = False):
    """Max CUDA bytes allocated since the last reset (MB), or None
    off-CUDA. With `reset=True` the counter restarts -- called at
    each iteration start so the logged value is THIS iteration's
    true backward peak (2026-07-18 OOM: steady-state alloc looked
    fine at ~7GB while the backward peaked past 12GB)."""
    import torch
    if not torch.cuda.is_available():
        return None
    peak = round(torch.cuda.max_memory_allocated() / 2**20)
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return peak


def run_iteration(
    policy:        TransformerPolicy,
    pool_files:    Optional[List[Path]],   # legacy; ignored post-pivot
    reward_fn,
    cost_lookup:   Dict[str, int],
    *,
    iter_idx:      int,
    games_per_iter: int,
    max_turns:     int,
    rng:           random.Random,
    pvp_defaults:  Optional[PvPDefaults] = None,
    workers:       int = 0,
    train_at_end:  bool = True,
    forced_faction: Optional[str] = ...,
    mini_maps:     bool = False,
    mini_ratio:    float = 0.0,
    fogless_ratio: float = 0.0,
    midgame_ratio: float = 0.0,
    ladder_ratio:  float = 1.0,
    max_turns_min: Optional[int] = None,
    no_progress_turns: int = 0,
    midgame_dataset: Optional[Path] = None,
    game_log_dir: Optional[Path] = None,
    snapshot_sink: Optional[Callable[[Dict], None]] = None,
    actor_pool=None,
    human_anchor=None,   # (pool, updates_per_iter, batch, rng) or None
    human_anchor_policy=None,  # (pairs, updates_per_iter, batch, rng) or None
) -> List[GameOutcome]:
    """Roll out `games_per_iter` games and call `train_step` once at
    the end. Returns the per-game outcomes for logging.

    `train_at_end`: when False, skip the post-rollout `train_step`
    call and leave queued trajectories on `policy._queue` for the
    caller to inspect or drain. Tests use this to assert queue/
    pending invariants directly; production callers leave it True.

    `pvp_defaults`: forwarded to `WesnothSim.from_replay`. When set,
    each game starts with standard 2p ladder economy/experience
    rather than whatever the source replay's host had configured.
    Self-play wants this -- it ensures the policy learns a single
    consistent ruleset rather than per-host quirks.

    `workers`: 0 = serial (existing behavior; runs on the main
    thread, simplest path). >= 1 = spawn N worker threads each
    pulling games from a shared counter and feeding trajectories
    to the policy concurrently. Builds on the snapshot+lock design
    in TransformerPolicy: workers' select_action / observe calls
    are thread-safe; the main thread fires train_step after all
    workers finish.

    Throughput: on CPU, the GIL serializes most of the forward
    work, so workers buy mostly the encode-while-other-thread-is-
    forwarding overlap (~30% speedup at workers=4). On GPU,
    multiple workers dispatching forwards in parallel keep the
    GPU saturated while the main thread also runs gradient compute
    (the snapshot design makes this safe).
    """
    outcomes: List[GameOutcome] = []
    t0 = time.perf_counter()

    # Reset the per-component reward accumulator at the start of the
    # iter. WeightedReward (if that's the reward_fn) accumulates per
    # component into _component_acc; we read it out at iter end and
    # log to trainer_history. No-op for non-WeightedReward callers
    # (custom reward fns are free to not expose this attribute).
    if hasattr(reward_fn, "_component_acc"):
        reward_fn._component_acc = {}

    if actor_pool is not None:
        # Actor-pool path (MCTS only): self-play runs in weightless
        # actor PROCESSES feeding this process's central batched-
        # inference server (see tools/actor_pool). The actors ship back
        # completed MCTSExperiences; we drain them into the learner's
        # queue here, and the shared train_step below applies the
        # gradient update exactly as in the in-process paths.
        base_seed = rng.randint(0, 2**31 - 1)
        pool_outcomes, pool_exps = actor_pool.run_iteration(
            iter_idx, games_per_iter, base_seed)
        outcomes.extend(pool_outcomes)
        with policy._lock:
            policy._queue.extend(pool_exps)
    elif workers <= 0:
        # Serial path -- simplest, used for tests and smoke runs.
        from tools.scenario_pool import random_setup, roll_mix
        for g_idx in range(games_per_iter):
            cat = roll_mix(rng, midgame=midgame_ratio,
                           mini=mini_ratio,
                           fogless=fogless_ratio, ladder=ladder_ratio)
            setup = None
            if cat == "midgame":
                from tools.midgame_starts import sample_midgame_start
                mg = sample_midgame_start(
                    rng, midgame_dataset or Path("replays_dataset"))
                if mg is not None:
                    setup = ("__midgame__",) + mg
                else:
                    cat = "ladder"  # degraded sample -> regular game
            if setup is None:
                setup = random_setup(rng, forced_faction=forced_faction,
                                     mini_maps=mini_maps, category=cat)
            game_label = f"iter{iter_idx}_g{g_idx}"
            outcome = _play_one_game_safe(
                setup=setup,
                max_turns=_roll_max_turns(rng, max_turns,
                                          max_turns_min),
                pvp_defaults=pvp_defaults, policy=policy,
                reward_fn=reward_fn, cost_lookup=cost_lookup,
                game_label=game_label, seed_salt=f"pool:{game_label}",
                no_progress_turns=no_progress_turns,
            )
            if outcome is not None:
                outcomes.append(outcome)
    else:
        # Parallel path -- N worker threads share the policy +
        # reward_fn + replay pool. Each worker has its own RNG
        # (seeded from the master rng) so games stay deterministic
        # given the master seed even with worker scheduling jitter.
        import threading
        shared = {
            "lock":           threading.Lock(),
            "next_game":      0,
            "target_games":   games_per_iter,
            "iter_idx":       iter_idx,
            "outcomes":       outcomes,
            "game_seed":      rng.randint(0, 2**32 - 1),
        }
        threads = []
        for w in range(workers):
            t = threading.Thread(
                target=_worker_loop,
                kwargs=dict(
                    worker_id=w, policy=policy,
                    reward_fn=reward_fn, cost_lookup=cost_lookup,
                    max_turns=max_turns, pvp_defaults=pvp_defaults,
                    shared=shared,
                    forced_faction=forced_faction,
                    mini_maps=mini_maps,
                    mini_ratio=mini_ratio,
                    fogless_ratio=fogless_ratio,
                    midgame_ratio=midgame_ratio,
                    ladder_ratio=ladder_ratio,
                    midgame_dataset=midgame_dataset,
                    max_turns_min=max_turns_min,
                    no_progress_turns=no_progress_turns,
                ),
                daemon=True,
                name=f"selfplay-w{w}",
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    rollout_dt = time.perf_counter() - t0
    n_actions = sum(o.side1_actions + o.side2_actions for o in outcomes)
    n_turns = sum(o.turns for o in outcomes)
    log.info(
        f"iter {iter_idx}: rolled {len(outcomes)} games in {rollout_dt:.1f}s "
        f"({n_actions} actions, {n_turns} turns; "
        f"{n_actions/max(rollout_dt, 1e-9):.0f} actions/s)"
    )

    # Aggregate-side win-rate. winner==1 means side 1 won; we flip per
    # side so both contribute to a single "did the better policy win"
    # statistic. Currently both sides share the same policy, so we
    # just report side-1 wins / draws / losses.
    s1_wins = sum(1 for o in outcomes if o.winner == 1)
    s2_wins = sum(1 for o in outcomes if o.winner == 2)
    draws   = sum(1 for o in outcomes if o.winner == 0)
    avg_r1  = (sum(o.side1_reward for o in outcomes) / len(outcomes)) if outcomes else 0.0
    avg_r2  = (sum(o.side2_reward for o in outcomes) / len(outcomes)) if outcomes else 0.0
    # ended_by demix (user ruling 2026-08-17, A5(iii)/A6 telemetry):
    # a turn-cap game, a stalemate, and a mutual elimination are
    # different events and must never share one aggregate number.
    # `abandoned` counts actors whose in-flight games the pool's hard
    # deadline discarded (0 when the drain grace was enough).
    ended: Dict[str, int] = {}
    for o in outcomes:
        ended[o.ended_by] = ended.get(o.ended_by, 0) + 1
    ended_leader = ended.pop("leader_killed", 0)
    ended_cap = ended.pop("max_turns", 0)
    ended_actions = ended.pop("max_actions", 0)
    ended_noprog = ended.pop("no_progress", 0)
    ended_other = sum(ended.values())
    abandoned = int(getattr(actor_pool, "_last_abandoned", 0) or 0) \
        if actor_pool is not None else 0
    log.info(
        f"iter {iter_idx}: outcomes s1_wins={s1_wins} s2_wins={s2_wins} "
        f"draws/timeouts={draws}; ended_by[leader={ended_leader} "
        f"cap={ended_cap} actions={ended_actions} "
        f"noprog={ended_noprog} other={ended_other}] "
        f"abandoned_actors={abandoned}; "
        f"mean_reward s1={avg_r1:+.3f} s2={avg_r2:+.3f}"
    )

    # Behavioral diagnostics: action histogram + mean turns/game +
    # mean living units per side at game end. These three numbers
    # are the headline indicator of the no-kills problem: if the
    # histogram shows recruit + end_turn dominate and units_end is
    # high for both sides, the policy is "build forever, never
    # engage". A healthy training trajectory should see attack% rise
    # over iterations, mean_turns drop (decisive games end sooner),
    # and units_end skew toward the winner.
    # Computed below; declared up front so the train_at_end block can
    # write a CSV row regardless of whether outcomes were produced.
    action_pcts: Dict[str, float] = {k: 0.0 for k in
                                     ("recruit", "move", "attack",
                                      "end_turn", "other")}
    mean_turns: float = 0.0
    mean_u1: float = 0.0
    mean_u2: float = 0.0
    mean_ca1: Optional[float] = None
    mean_ca2: Optional[float] = None
    if outcomes:
        # Sum per-type tallies across games.
        totals: Dict[str, int] = {}
        for o in outcomes:
            for k, v in o.action_counts.items():
                totals[k] = totals.get(k, 0) + v
        total_actions = max(sum(totals.values()), 1)
        # Stable order so it's grep-able across runs; "other" sweeps
        # up anything that isn't one of the four known categories.
        known = ["recruit", "move", "attack", "end_turn"]
        for k in known:
            action_pcts[k] = 100.0 * totals.get(k, 0) / total_actions
        other = sum(v for k, v in totals.items() if k not in known)
        action_pcts["other"] = 100.0 * other / total_actions
        parts = [f"{k}={action_pcts[k]:.0f}%" for k in known]
        if other:
            parts.append(f"other={action_pcts['other']:.0f}%")
        mean_turns = n_turns / len(outcomes)
        mean_u1 = sum(o.side1_units_end for o in outcomes) / len(outcomes)
        mean_u2 = sum(o.side2_units_end for o in outcomes) / len(outcomes)
        # Closest-approach: average over games that have a value
        # (a game where this side's leader never met any opposing
        # unit returns None and is excluded). Smaller is more
        # threatening; high values across iterations = no-kills.
        ca1_vals = [o.side1_closest_approach for o in outcomes
                    if o.side1_closest_approach is not None]
        ca2_vals = [o.side2_closest_approach for o in outcomes
                    if o.side2_closest_approach is not None]
        if ca1_vals:
            mean_ca1 = sum(ca1_vals) / len(ca1_vals)
        if ca2_vals:
            mean_ca2 = sum(ca2_vals) / len(ca2_vals)
        ca_str = (f"  closest_approach s1="
                  f"{f'{mean_ca1:.1f}' if mean_ca1 is not None else 'n/a'}"
                  f" s2="
                  f"{f'{mean_ca2:.1f}' if mean_ca2 is not None else 'n/a'}")
        log.info(
            f"iter {iter_idx}: actions[{', '.join(parts)}] "
            f"mean_turns={mean_turns:.1f} "
            f"mean_units_end s1={mean_u1:.1f} s2={mean_u2:.1f}"
            f"{ca_str}"
        )

    # Per-map-class decisive split. The AGGREGATE decisive rate is
    # misleading over a mixed curriculum (2026-07-03: ~50% aggregate,
    # 0/8 decisive on ladder maps — all kills came from minis, and
    # the pool deadline abandoning slow ladder draws inflated it
    # further). Ladder decisiveness is the number that matters for
    # full-game strength; log both.
    # Midgame continuations are ladder-class but inherit village-rich
    # human positions; pooling them into the fresh buckets made both
    # the decisive split and the fog split misleading (2026-07-14).
    # They get their own segment; ladder/mini buckets are FRESH only.
    _fresh = [o for o in outcomes if not getattr(o, "midgame", False)]
    ladder_n = sum(1 for o in _fresh if o.map_class == "ladder")
    ladder_dec = sum(1 for o in _fresh
                     if o.map_class == "ladder" and o.winner != 0)
    other_n = len(_fresh) - ladder_n
    other_dec = sum(1 for o in _fresh
                    if o.map_class != "ladder" and o.winner != 0)
    # Per-class SIDE split (2026-07-07: a 1-7 side-2 iteration was
    # only visible as an anecdote; global s1/s2 hides which map class
    # carries an asymmetry).
    ladder_s1 = sum(1 for o in _fresh
                    if o.map_class == "ladder" and o.winner == 1)
    ladder_s2 = ladder_dec - ladder_s1
    other_s1 = sum(1 for o in _fresh
                   if o.map_class != "ladder" and o.winner == 1)
    other_s2 = other_dec - other_s1
    if outcomes:
        _mg_log = [o for o in outcomes if getattr(o, "midgame", False)]
        log.info(
            f"iter {iter_idx}: decisive split -- ladder "
            f"{ladder_dec}/{ladder_n} (s1 {ladder_s1}, s2 {ladder_s2}), "
            f"mini {other_dec}/{other_n} "
            f"(s1 {other_s1}, s2 {other_s2}), midgame "
            f"{sum(1 for o in _mg_log if o.winner != 0)}/{len(_mg_log)}")
    # Per-game JSONL (2026-07-12 user spec): one directory PER
    # ITERATION so records stay browsable, one JSON line per game
    # with the full engagement telemetry. Directory names carry a
    # per-PROCESS run number (iter_MMM_NNNNNN, user 2026-07-21):
    # the in-process iteration counter restarts at 0 on every
    # relaunch, so unprefixed dirs interleaved epochs and offline
    # analysis had to segment appends by block size.
    if game_log_dir is not None and outcomes:
        _dir = game_log_dir / f"iter_{_game_log_run_id(game_log_dir):03d}_{iter_idx:06d}"
        _dir.mkdir(parents=True, exist_ok=True)
        # NB json.dumps stringifies the engagement dicts' int side
        # keys ("1"/"2"); offline readers must not use int keys.
        with (_dir / "games.jsonl").open("a", encoding="utf-8") as _f:
            for o in outcomes:
                _f.write(json.dumps({
                    "game_label": o.game_label,
                    "winner": o.winner,
                    "ended_by": o.ended_by,
                    "turns": o.turns,
                    "map_class": o.map_class,
                    "fogless": getattr(o, "fogless", False),
                    "midgame": getattr(o, "midgame", False),
                    "action_counts": o.action_counts,
                    "noprogress": getattr(o, "noprogress", None),
                    "max_turns": getattr(o, "max_turns", None),
                    "units_end": [o.side1_units_end, o.side2_units_end],
                    "closest_approach": [o.side1_closest_approach,
                                         o.side2_closest_approach],
                    "end_gold": [o.side1_end_gold, o.side2_end_gold],
                    "recruits": [o.n_recruits_s1, o.n_recruits_s2],
                    "recruit_attempts": [o.n_recruit_attempts_s1,
                                         o.n_recruit_attempts_s2],
                    "villages_mean": [
                        getattr(o, "villages_mean_s1", 0.0),
                        getattr(o, "villages_mean_s2", 0.0)],
                    "villages_end": [getattr(o, "villages_end_s1", 0),
                                     getattr(o, "villages_end_s2", 0)],
                    "engagement": getattr(o, "engagement", None),
                }) + "\n")

    # Iteration-level aggregates of the engagement telemetry (the
    # curve-worthy scalars; full detail lives in the JSONL).
    _engs = [getattr(o, "engagement", None) for o in outcomes]
    _engs = [e for e in _engs if e]

    def _emean(fn):
        vals = [v for v in (fn(e) for e in _engs) if v is not None]
        return (sum(vals) / len(vals)) if vals else None

    def _e2(e, key):
        d = e.get(key) or {}
        return (d.get(1, 0) or 0) + (d.get(2, 0) or 0)

    def _e2mean(e, key):
        d = e.get(key) or {}
        vals = [v for v in (d.get(1), d.get(2)) if v is not None]
        return (sum(vals) / len(vals)) if vals else None

    _mg = [o for o in outcomes if getattr(o, "midgame", False)]
    eng_agg = {
        "midgame_games": len(_mg),
        "midgame_decisive": sum(1 for o in _mg if o.winner != 0),
        "eng_games": len(_engs),
        "eng_attacks_pg": _emean(lambda e: _e2(e, "attacks_attempted")),
        # Tripwires: SUMS across the iteration; expect exactly 0.
        "eng_invalid_attacks": (sum(_e2(e, "attacks_invalid_wesnoth")
                                    for e in _engs) if _engs else None),
        "eng_rejected_attacks": (sum(_e2(e, "attacks_rejected_sim")
                                     for e in _engs) if _engs else None),
        "eng_damage_pg": _emean(lambda e: _e2(e, "damage_dealt")),
        "eng_kills_pg": _emean(
            lambda e: sum(sum(v.values()) for v in
                          (e.get("kills") or {}).values())),
        "eng_kills_value_pg": _emean(lambda e: _e2(e, "kills_value")),
        "eng_heal_village_pg": _emean(lambda e: _e2(e, "heal_village")),
        "eng_heal_rest_pg": _emean(lambda e: _e2(e, "heal_rest")),
        "eng_heal_ability_pg": _emean(lambda e: _e2(e, "heal_ability")),
        "eng_advancements_pg": _emean(lambda e: _e2(e, "advancements")),
        "eng_poison_cured_pg": _emean(lambda e: _e2(e, "poison_cured")),
        "eng_poison_damage_pg": _emean(
            lambda e: _e2(e, "poison_damage_taken")),
        "eng_contact_rate": (sum(1 for e in _engs
                                 if e.get("first_contact_turn")
                                 is not None) / len(_engs)
                             if _engs else None),
        "eng_first_contact_turn": _emean(
            lambda e: e.get("first_contact_turn")),
        "eng_scouted_frac": _emean(lambda e: _e2mean(e, "scouted_frac")),
        # Per-SIDE watch metrics (user 2026-07-20). Kept per side,
        # unlike the side-averaged fractions: asymmetry between the
        # winner and loser is part of the signal.
        "eng_unused_mp_s1": _emean(
            lambda e: (e.get("unused_mp_frac") or {}).get(1)),
        "eng_unused_mp_s2": _emean(
            lambda e: (e.get("unused_mp_frac") or {}).get(2)),
        "eng_gold_bank_s1": _emean(
            lambda e: (e.get("gold_bank_mean") or {}).get(1)),
        "eng_gold_bank_s2": _emean(
            lambda e: (e.get("gold_bank_mean") or {}).get(2)),
        "eng_villages_frac": _emean(
            lambda e: _e2mean(e, "villages_frac_avg")),
        "eng_material_end_pg": _emean(lambda e: _e2(e, "material_end")),
        "search_q_spread": _emean(
            lambda e: (e.get("search") or {}).get("q_spread_mean")),
        "search_overturn_frac": _emean(
            lambda e: (e.get("search") or {}).get("overturn_frac")),
    }

    # Fogless-mixing observability (2026-07-11 user request): the
    # ladder pool mixes fogged and fogless games; whether fogless is
    # doing work (more captures, closer approach, decisive games)
    # must be visible per CONDITION, not pooled. villages/turn is
    # the time-averaged count of villages owned (both sides summed);
    # `end` is final ownership. getattr() defaults tolerate outcomes
    # pickled by an older code version.
    def _fog_cond_stats(want_fogless: bool) -> Dict:
        # FRESH ladder games only: midgame continuations are fog-on
        # and village-rich by inheritance, and pooling them here made
        # the fogged bucket read ~5 villages/turn vs 0.7 fogless
        # (2026-07-14 confusion).
        games = [o for o in outcomes
                 if o.map_class == "ladder"
                 and not getattr(o, "midgame", False)
                 and getattr(o, "fogless", False) == want_fogless]
        if not games:
            return {"n": 0, "dec": 0, "vpt": None, "vend": None,
                    "appr": None}
        vpt = sum(getattr(o, "villages_mean_s1", 0.0)
                  + getattr(o, "villages_mean_s2", 0.0)
                  for o in games) / len(games)
        vend = sum(getattr(o, "villages_end_s1", 0)
                   + getattr(o, "villages_end_s2", 0)
                   for o in games) / len(games)
        appr = [x for o in games
                for x in (o.side1_closest_approach,
                          o.side2_closest_approach) if x is not None]
        return {"n": len(games),
                "dec": sum(1 for o in games if o.winner != 0),
                "vpt": vpt, "vend": vend,
                "appr": (sum(appr) / len(appr)) if appr else None}
    fog_stats = _fog_cond_stats(False)
    fogless_stats = _fog_cond_stats(True)
    if fog_stats["n"] or fogless_stats["n"]:
        def _fmt_cond(c: Dict) -> str:
            if not c["n"]:
                return "0 games"
            appr = f"{c['appr']:.1f}" if c["appr"] is not None else "n/a"
            return (f"{c['dec']}/{c['n']} dec, vil/turn {c['vpt']:.2f} "
                    f"(end {c['vend']:.1f}), approach {appr}")
        log.info(
            f"iter {iter_idx}: FRESH ladder fog split -- fogged "
            f"{_fmt_cond(fog_stats)} | fogless "
            f"{_fmt_cond(fogless_stats)}")

    # Actions-per-side-turn distribution, pooled across the iter's
    # games (MCTS depth calibration: S sims / A actions-per-side-turn
    # ≈ how much of one turn plan the search can look ahead).
    pooled_apt = sorted(
        c for o in outcomes for c in (o.turn_action_counts or []))
    # (k_median_of below recomputes the same statistic for the
    # K-collapse tripwire in the main loop.)
    apt_mean = (sum(pooled_apt) / len(pooled_apt)) if pooled_apt else None
    apt_median = pooled_apt[len(pooled_apt) // 2] if pooled_apt else None
    if pooled_apt:
        log.info(
            f"iter {iter_idx}: actions/side-turn mean={apt_mean:.1f} "
            f"median={apt_median} max={pooled_apt[-1]} "
            f"(n={len(pooled_apt)} side-turns)")

    train_stats = None
    human_anchor_loss = None
    if train_at_end and human_anchor is not None:
        # Human-corpus rehearsal (2026-07-10): value-only gradient
        # steps on pre-encoded human states with clean +-1 labels,
        # BEFORE train_step so its inference-weight sync captures
        # them. The anti-forgetting anchor: self-play alone eroded
        # human late-game AUC 0.88 -> 0.60 in ~80 iterations.
        a_pool, a_updates, a_batch, a_rng = human_anchor
        trainer = getattr(policy, "_base", policy)._trainer
        losses = []
        for _ in range(a_updates):
            sample = a_rng.sample(a_pool, min(a_batch, len(a_pool)))
            st = trainer.step_value_from_raw(
                [t[0] for t in sample], [t[1] for t in sample],
                [t[2] for t in sample])
            losses.append(st["value_loss"])
        human_anchor_loss = sum(losses) / max(1, len(losses))
        # Return the anchor batches' cached blocks to the driver:
        # human ladder states are much larger than self-play batches
        # and ratchet the allocator cache (measured 2026-07-10:
        # 162MB allocated vs 11.5GB reserved after one iteration).
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:                       # noqa: BLE001
            pass
        log.info(f"iter {iter_idx}: human anchor -- {a_updates} "
                 f"updates x {a_batch}, value_loss="
                 f"{human_anchor_loss:.4f}")
    human_anchor_policy_ce = None
    if train_at_end and human_anchor_policy is not None:
        # Policy-head rehearsal (F1, 2026-08-10): the imitation
        # objective (four-head CE on winner-side human pairs) mixed
        # into every iteration -- the RLPD-shaped prior protection.
        # Default OFF; one protection per leg (vs A1 / piKL) for
        # attribution.
        from tools.policy_anchor import (
            anchor_policy_step, sample_pairs_game_normalized,
        )
        p_games, p_updates, p_batch, p_rng = human_anchor_policy
        trainer = getattr(policy, "_base", policy)._trainer
        ces = []
        for _ in range(p_updates):
            # v2 draw (2026-08-16): game-first, one pair per chosen
            # game -- equal per-game rehearsal weight, matching the
            # trainer's own per-game normalization principle.
            sample = sample_pairs_game_normalized(p_games, p_batch,
                                                  p_rng)
            st = anchor_policy_step(trainer, sample)
            ces.append(st["policy_ce"])
        human_anchor_policy_ce = sum(ces) / max(1, len(ces))
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:                       # noqa: BLE001
            pass
        log.info(f"iter {iter_idx}: policy anchor -- {p_updates} "
                 f"updates x {p_batch}, policy_ce="
                 f"{human_anchor_policy_ce:.4f}")
    if train_at_end:
        # One gradient step over all queued trajectories.
        train_t0 = time.perf_counter()
        try:
            train_stats = policy.train_step()
        except Exception as e:
            # OOM emergency path (2026-07-20): free the allocator
            # cache and retry ONCE. The step is retryable (replay-
            # buffer sampling); a second OOM re-raises to the
            # supervisor.
            import torch
            if not isinstance(e, torch.cuda.OutOfMemoryError):
                raise
            log.error("CUDA OOM in train_step; emptying cache, "
                      "retrying once")
            torch.cuda.empty_cache()
            train_stats = policy.train_step()
        # Value-memory step (user ruling 2026-08-30): one value-only
        # gradient step per iteration over game-uniform samples from
        # the wide outcome reservoir. Logged every iteration so the
        # decision stays reviewable (memory_games = the independent-
        # outcome sample the value head actually saw).
        if getattr(policy, "_value_memory_games", 0) > 0:
            _vm_step = getattr(policy, "value_memory_step", None)
            if _vm_step is not None:
                _vm = _vm_step()
                if _vm:
                    log.info(
                        f"iter {iter_idx}: value_memory step -- "
                        f"games={_vm.get('memory_games')} "
                        f"states={_vm.get('memory_states')} "
                        f"value_loss={_vm.get('value_loss', 0):.4f} "
                        f"grad_norm={_vm.get('grad_norm', 0):.3f}")
        train_dt = time.perf_counter() - train_t0
        # getattr-guarded: non-trainable/stub policies (dummy)
        # may return a minimal stats object without the aux field.
        _aux = getattr(train_stats, "aux_loss", 0.0)
        aux_str = f" aux={_aux:.4f}" if _aux else ""
        _ml = getattr(train_stats, "moves_left_loss", 0.0)
        aux_str += f" moves_left={_ml:.4f}" if _ml else ""
        _gbc = getattr(train_stats, "gbc_loss", 0.0)
        aux_str += f" gbc={_gbc:.4f}" if _gbc else ""
        _zw = getattr(train_stats, "z_win_frac", float("nan"))
        if _zw == _zw:
            aux_str += (f" z_comp={_zw:.2f}/"
                        f"{getattr(train_stats, 'z_loss_frac', 0):.2f}/"
                        f"{getattr(train_stats, 'z_draw_frac', 0):.2f}")
        # Weighted variant = the gradient's actual composition
        # (per-game normalization); the census above is length-
        # inflated by long games (misread 2026-07-22).
        _zww = getattr(train_stats, "z_win_frac_w", float("nan"))
        if _zww == _zww:
            aux_str += (
                f" z_comp_w={_zww:.2f}/"
                f"{getattr(train_stats, 'z_loss_frac_w', 0):.2f}/"
                f"{getattr(train_stats, 'z_draw_frac_w', 0):.2f}")
        _fce = getattr(train_stats, "fresh_value_ce", float("nan"))
        if _fce == _fce:                       # not NaN
            # DEFAULT success metric (user 2026-07-22): distribution-
            # matched, pre-update. +-std says whether a move is
            # signal or ~256-state probe noise. The frozen holdout
            # CE is the stall TRIPWIRE, not the success gauge (it
            # drifts off-distribution as play evolves).
            _fstd = getattr(train_stats, "fresh_ce_std", float("nan"))
            aux_str += (f" fresh_value_ce={_fce:.4f}"
                        + (f"+-{_fstd:.4f}" if _fstd == _fstd else ""))
            _fent = getattr(train_stats, "fresh_pred_entropy",
                            float("nan"))
            _ffloor = getattr(train_stats, "fresh_ce_floor",
                              float("nan"))
            if _fent == _fent:
                aux_str += f" fresh_pred_entropy={_fent:.4f}"
            if _ffloor == _ffloor:
                aux_str += f" fresh_ce_floor={_ffloor:.4f}"
        # Boundary-consistency telemetry (T1-F, 2026-07-29): mean
        # V(pre)+V(post) at sampled side switches. ~0 = calibrated;
        # +0.4..+0.6 = the fogged-play WYSIATI bias (both sides read
        # optimistic at a turn handoff). NaN until >=4 pairs exist.
        _bsum = getattr(train_stats, "boundary_sum", float("nan"))
        if _bsum == _bsum:
            aux_str += (f" boundary_sum={_bsum:+.3f}"
                        f"/n={getattr(train_stats, 'boundary_pairs_n', 0)}"
                        f"/pool={getattr(train_stats, 'boundary_pool_n', 0)}")
        log.info(
            f"iter {iter_idx}: train_step in {train_dt:.1f}s "
            f"trajectories={train_stats.n_trajectories} transitions={train_stats.n_transitions} "
            f"loss={train_stats.total_loss:.4f} policy={train_stats.policy_loss:.4f} "
            f"value={train_stats.value_loss:.4f}{aux_str} entropy={train_stats.entropy:.4f} "
            f"mean_return={train_stats.mean_return:+.3f} grad_norm={train_stats.grad_norm:.3f}"
        )

    # Held-out value CE with the post-update net (--holdout-size;
    # MCTSPolicy only -- getattr-guarded so every other policy type is
    # untouched). This is the generalization probe: the train value
    # loss above is measured on replay samples the net already fit.
    holdout_loss = holdout_n = None
    holdout_fn = getattr(policy, "holdout_metrics", None)
    if train_at_end and holdout_fn is not None:
        hm = holdout_fn()
        if hm is not None:
            holdout_loss, holdout_n = hm
            log.info(
                f"iter {iter_idx}: holdout value CE={holdout_loss:.4f} "
                f"on {holdout_n} held-out states (never trained on)")

    # Optional snapshot sink: the main loop passes a callable that
    # appends a CSV row to disk. Tests don't pass one, so this is
    # a no-op there. Keeping the return type a plain List[GameOutcome]
    # avoids touching every test that asserts on it.
    # This iteration's trainer backward peak (gpu_mem_peak_mb).
    _peak_mb = _gpu_mem_peak_mb(reset=True)
    # Floor-relative fresh holdout CE (fresh_value_ce - fresh_ce_floor):
    # the A5 stall-tripwire metric (raw CE moves with the outcome-label
    # mix; the raw version mis-fired twice in the 72h run). Stashed on
    # the policy for the main loop's tripwire, INDEPENDENT of
    # snapshot_sink; None when either side is missing/NaN (replay
    # buffer off, iter-0-after-restart).
    _fv = (getattr(train_stats, "fresh_value_ce", float("nan"))
           if train_stats else float("nan"))
    _fl = (getattr(train_stats, "fresh_ce_floor", float("nan"))
           if train_stats else float("nan"))
    _fresh_rel_ce = (_fv - _fl) if (_fv == _fv and _fl == _fl) else None
    policy.last_fresh_rel_ce = _fresh_rel_ce
    if snapshot_sink is not None:
        # Econ aggregates across all games this iter.
        mean_end_gold_s1 = (
            sum(o.side1_end_gold for o in outcomes) / len(outcomes)
            if outcomes else 0.0)
        mean_end_gold_s2 = (
            sum(o.side2_end_gold for o in outcomes) / len(outcomes)
            if outcomes else 0.0)
        n_recruits_s1 = sum(o.n_recruits_s1 for o in outcomes)
        n_recruits_s2 = sum(o.n_recruits_s2 for o in outcomes)
        n_recruit_attempts_s1 = sum(o.n_recruit_attempts_s1 for o in outcomes)
        n_recruit_attempts_s2 = sum(o.n_recruit_attempts_s2 for o in outcomes)
        # Per-component reward sums (read+snapshot, then leave on
        # reward_fn for any other consumer; the next iter resets).
        reward_components = dict(
            getattr(reward_fn, "_component_acc", None) or {})
        # Gold-hoarding watch (user 2026-07-20): time-averaged bank
        # gold per side (sampled at the side's own end_turns) next to
        # the end-of-game snapshot -- a hoard spent late shows in the
        # bank mean even when end gold looks lean.
        _b1 = eng_agg.get("eng_gold_bank_s1")
        _b2 = eng_agg.get("eng_gold_bank_s2")
        _bank_str = ""
        if _b1 is not None or _b2 is not None:
            _f = lambda v: f"{v:.1f}" if v is not None else "n/a"
            _bank_str = f"bank_mean s1={_f(_b1)} s2={_f(_b2)}; "
        log.info(
            f"iter {iter_idx}: econ "
            f"mean_end_gold s1={mean_end_gold_s1:.1f} s2={mean_end_gold_s2:.1f}; "
            f"{_bank_str}"
            f"recruits s1={n_recruits_s1}(of {n_recruit_attempts_s1}) "
            f"s2={n_recruits_s2}(of {n_recruit_attempts_s2})"
        )
        if reward_components:
            comp_str = ", ".join(
                f"{k}={v:+.3f}" for k, v in sorted(
                    reward_components.items(), key=lambda kv: -abs(kv[1])))
            log.info(f"iter {iter_idx}: reward components -- {comp_str}")
        # Distillation-target telemetry (prior-ratchet repair
        # observability): drained per iteration from MCTSPolicy;
        # None-safe for the REINFORCE / actor-pool paths.
        distill = (getattr(policy, "drain_distill_stats", None)
                   and policy.drain_distill_stats()) or {}
        # Under the actor pool the LEARNER's policy never searches, so
        # its drain is empty -- the telemetry rode only the in-process
        # path and an entire leg ran with every distill_* column dark
        # (2026-08-12 diagnosis, finding F4). The pool now aggregates
        # its actors' drains per iteration.
        if not distill and actor_pool is not None:
            distill = getattr(actor_pool, "last_distill_stats",
                              None) or {}
        if distill and "distill_sharpen_top" in distill:
            _et_p, _et_t = (distill.get("distill_et_prior"),
                            distill.get("distill_et_target"))
            _et_str = (f"et prior {_et_p:.3f} -> target {_et_t:.3f}; "
                       if _et_p is not None and _et_t is not None
                       else "")
            log.info(
                f"iter {iter_idx}: distill target -- "
                f"sharpen_top {distill['distill_sharpen_top']:+.4f} "
                f"(<0 = ratchet damped), "
                f"prior>0.8 rate "
                f"{distill['distill_prior_top80']:.3f}; "
                f"{_et_str}"
                f"H(target) {distill['distill_tgt_entropy']:.3f} vs "
                f"H(prior) {distill['distill_prior_entropy']:.3f}")
        # TCS planning telemetry (2026-08-17: rode NO path during the
        # leg-3 collapse -- accept-rate drift was unobservable).
        if distill and distill.get("tcs_plans"):
            _proj = distill.get("tcs_projections_per_plan", 0.0) or 0.0
            log.info(
                f"iter {iter_idx}: tcs -- plans/actor "
                f"{distill['tcs_plans']:.0f}, accepts/plan "
                f"{distill.get('tcs_accepts_per_plan', 0.0):.2f}, "
                f"replans/plan "
                f"{distill.get('tcs_replans_per_plan', 0.0):.2f}, "
                f"projections/plan {_proj:.1f}")
        _sig = {k: (getattr(train_stats, k, None)
                    if train_stats else None)
                for k in ("sig_policy_norm", "sig_value_game_norm",
                          "sig_value_ground_norm",
                          "sig_value_consist_norm", "sig_seconds",
                          "sig_dv_consult_mean", "sig_dv_consult_n",
                          "consist_bias_hat", "consist_sigma2_hat",
                          "consist_pair_n", "consist_loss",
                          "trust_loss", "trust_lambda",
                          "consist_var_diff", "consist_roll_noise",
                          "consist_sigma2_point", "consist_sigma2_se",
                          "consist_head_minus_truth",
                          "consist_label_minus_truth")}
        _fbd = (getattr(train_stats, "fresh_by_decade", None)
                if train_stats else None) or {}
        _fresh_decades = {
            f"fresh_{m}_{d}": (_fbd.get(d) or {}).get(m)
            for d in ("d1_10", "d11_20", "d21_30", "d31_40",
                      "d41_50", "d51_60", "d61p")
            for m in ("ce", "floor", "auc", "n")}
        snapshot_sink({
            **distill,
            **_fresh_decades,
            **_sig,
            "iter":                iter_idx,
            "n_games":             len(outcomes),
            "rollout_seconds":     rollout_dt,
            "n_actions":           n_actions,
            "mean_turns":          mean_turns,
            "s1_wins":             s1_wins,
            "s2_wins":             s2_wins,
            "draws":               draws,
            # ended_by demix + discard telemetry (2026-08-17 rulings)
            "ended_leader":        ended_leader,
            "ended_max_turns":     ended_cap,
            "ended_max_actions":   ended_actions,
            "ended_no_progress":   ended_noprog,
            "ended_other":         ended_other,
            "abandoned_actors":    abandoned,
            "action_recruit_pct":  action_pcts["recruit"],
            "action_move_pct":     action_pcts["move"],
            "action_attack_pct":   action_pcts["attack"],
            "action_end_turn_pct": action_pcts["end_turn"],
            "action_other_pct":    action_pcts["other"],
            "mean_units_end_s1":   mean_u1,
            "mean_units_end_s2":   mean_u2,
            "closest_approach_s1": mean_ca1,
            "closest_approach_s2": mean_ca2,
            "train_stats":         train_stats,
            # Per-component reward + econ diagnostics.
            "reward_components":   reward_components,
            "mean_end_gold_s1":    mean_end_gold_s1,
            "mean_end_gold_s2":    mean_end_gold_s2,
            "n_recruits_s1":       n_recruits_s1,
            "n_recruits_s2":       n_recruits_s2,
            "n_recruit_attempts_s1": n_recruit_attempts_s1,
            "n_recruit_attempts_s2": n_recruit_attempts_s2,
            "holdout_value_loss":  holdout_loss,
            "holdout_n":           holdout_n,
            "fresh_value_ce":      (getattr(train_stats, "fresh_value_ce",
                                            None) if train_stats else None),
            "fresh_ce_std":        (getattr(train_stats, "fresh_ce_std",
                                            None) if train_stats else None),
            "fresh_pred_entropy":  (getattr(train_stats,
                                            "fresh_pred_entropy",
                                            None) if train_stats else None),
            "fresh_ce_floor":      (getattr(train_stats, "fresh_ce_floor",
                                            None) if train_stats else None),
            "ladder_s1_wins":      ladder_s1,
            "ladder_s2_wins":      ladder_s2,
            "other_s1_wins":       other_s1,
            "other_s2_wins":       other_s2,
            # Cumulative training-unit progress, so cost-per-unit is
            # computable from the CSV alone (2026-07-07 reporting
            # rule: runway in training units, not wall-clock).
            "decision_step":       getattr(
                getattr(policy, "_base", policy), "_decision_step",
                None),
            # Trainer-process GPU memory (MB). The undiagnosed creep
            # (2026-07-10: OOM'd a 16GB card in ~3h) becomes a curve:
            # linear slope = leak-like, staircase = allocator
            # high-water ratchet on variable-length batches.
            "gpu_mem_alloc_mb":    _gpu_mem_mb()[0],
            "gpu_mem_reserved_mb": _gpu_mem_mb()[1],
            "gpu_mem_peak_mb":     _peak_mb,
            "z_win_frac":   (getattr(train_stats, "z_win_frac", None)
                             if train_stats else None),
            "z_loss_frac":  (getattr(train_stats, "z_loss_frac", None)
                             if train_stats else None),
            "z_draw_frac":  (getattr(train_stats, "z_draw_frac", None)
                             if train_stats else None),
            # game_weight-normalized composition = actual gradient
            # share (the unweighted census above is length-inflated).
            "z_win_frac_w": (getattr(train_stats, "z_win_frac_w", None)
                             if train_stats else None),
            "z_loss_frac_w": (getattr(train_stats, "z_loss_frac_w",
                                      None) if train_stats else None),
            "z_draw_frac_w": (getattr(train_stats, "z_draw_frac_w",
                                      None) if train_stats else None),
            "human_anchor_loss": human_anchor_loss,
            "human_anchor_policy_ce": human_anchor_policy_ce,
            "fresh_rel_ce": _fresh_rel_ce,
            "value_signal_states": (getattr(train_stats,
                                            "value_signal_states", None)
                                    if train_stats else None),
            "fresh_decisive_ce": (getattr(train_stats,
                                          "fresh_decisive_ce", None)
                                  if train_stats else None),
            "boundary_sum": (getattr(train_stats, "boundary_sum", None)
                             if train_stats else None),
            "boundary_pairs_n": (getattr(train_stats,
                                         "boundary_pairs_n", None)
                                 if train_stats else None),
            "ladder_games":        ladder_n,
            "ladder_decisive":     ladder_dec,
            "other_games":         other_n,
            "other_decisive":      other_dec,
            "actions_per_turn_mean":   apt_mean,
            "actions_per_turn_median": apt_median,
            # Ladder fog/fogless condition split (2026-07-11).
            "ladder_fog_games":        fog_stats["n"],
            "ladder_fog_decisive":     fog_stats["dec"],
            "ladder_fog_villages_per_turn": fog_stats["vpt"],
            "ladder_fog_villages_end": fog_stats["vend"],
            "ladder_fog_approach":     fog_stats["appr"],
            "ladder_fogless_games":    fogless_stats["n"],
            "ladder_fogless_decisive": fogless_stats["dec"],
            "ladder_fogless_villages_per_turn": fogless_stats["vpt"],
            "ladder_fogless_villages_end": fogless_stats["vend"],
            "ladder_fogless_approach": fogless_stats["appr"],
            **eng_agg,
        })
    return outcomes


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

class _TrainerHistoryCSV:
    """Lazy CSV writer for per-iter training stats. Writes the header
    on the first row so the file is self-describing.

    Columns are stable across iters (every per-game / per-iter
    diagnostic the operator might want to plot over time). When a
    field is None (e.g. closest-approach with no living units to
    measure) we emit an empty cell so pandas reads it as NaN.

    Atomic-ish: each row is one append, line-buffered. A crash
    mid-row would truncate to a partial line that pandas / awk
    would skip cleanly.
    """

    # Stable column order. Adding new columns at the end is
    # backwards-compatible (older pandas reads will just see fewer
    # columns); reordering or removing columns is a breaking change.
    # Stable per-component reward column names. Match the keys
    # WeightedReward.__call__ writes into its `_component_acc`.
    # Aggregated across all games in the iter and across both sides.
    # Empty cells (= 0) mean that component didn't fire this iter.
    _REWARD_COMPONENT_KEYS = (
        "gold_killed", "village_delta", "damage_dealt",
        "unit_recruited_cost", "per_turn_penalty",
        "leader_move_penalty", "invalid_action",
        "min_enemy_distance", "approach_mp", "unused_mp",
        "fog_reveal", "attack_attempt", "unit_type_bonus",
        "turn_cond_bonus", "terminal",
    )

    # Per-side mean gold / recruits at end of game, for diagnosing
    # recruit-underuse. mean_end_gold_s1 = "side 1 ended its game
    # with X gold left on average" -- if high, the policy is not
    # spending. n_recruits_s1 = "total successful recruits by side 1
    # across all games this iter."
    _ECON_KEYS = (
        "mean_end_gold_s1", "mean_end_gold_s2",
        "n_recruits_s1", "n_recruits_s2",
        "n_recruit_attempts_s1", "n_recruit_attempts_s2",
    )

    COLUMNS = [
        "iter", "timestamp", "n_games", "rollout_seconds",
        "n_actions", "mean_turns",
        "s1_wins", "s2_wins", "draws",
        "action_recruit_pct", "action_move_pct",
        "action_attack_pct", "action_end_turn_pct",
        "action_other_pct",
        "mean_units_end_s1", "mean_units_end_s2",
        "closest_approach_s1", "closest_approach_s2",
        "train_n_trajectories", "train_n_transitions",
        "train_loss", "train_policy_loss", "train_value_loss",
        "train_entropy", "train_mean_return", "train_grad_norm",
    ] + [f"r_{k}" for k in _REWARD_COMPONENT_KEYS] + list(_ECON_KEYS) + [
        # Held-out generalization probe (--holdout-size; MCTS only).
        # Appended LAST so rows appended to a pre-existing CSV stay
        # column-compatible with its older header.
        "holdout_value_loss", "holdout_n",
        # Pre-update value CE on this iter's fresh games (2026-07-07;
        # distribution-matched generalization, no training data lost)
        # + prediction entropy (overconfidence curve) + the state-
        # blind marginal floor (outcome-mix predictability cap).
        "fresh_value_ce", "fresh_ce_std", "fresh_pred_entropy",
        "fresh_rel_ce",
        "fresh_ce_floor",
        # Per-map-class decisive split (2026-07-03; aggregate decisive
        # over a mixed curriculum is misleading) + per-class SIDE
        # split (2026-07-07; asymmetries were anecdotes before).
        "ladder_games", "ladder_decisive",
        "other_games", "other_decisive",
        "ladder_s1_wins", "ladder_s2_wins",
        "other_s1_wins", "other_s2_wins",
        # Actions-per-side-turn distribution (MCTS depth calibration).
        "actions_per_turn_mean", "actions_per_turn_median",
        # Cumulative decision counter (training-unit progress; makes
        # cost-per-unit computable from the CSV alone).
        "decision_step",
        # Trainer-process CUDA memory (creep tracking, 2026-07-10).
        "gpu_mem_alloc_mb", "gpu_mem_reserved_mb",
        # Per-iteration CUDA high-water mark (reset each iteration):
        # the trainer's TRUE backward peak, which the worker-split
        # VRAM budget must reserve for (2026-07-18).
        "gpu_mem_peak_mb",
        # Value-target composition + human rehearsal (2026-07-10
        # draw-spike diagnosis / anchor fix).
        "z_win_frac", "z_loss_frac", "z_draw_frac",
        # game_weight-normalized composition (gradient share).
        "z_win_frac_w", "z_loss_frac_w", "z_draw_frac_w",
        "human_anchor_loss", "human_anchor_policy_ce",
        "value_signal_states",
        "fresh_decisive_ce",
        # Ladder fog/fogless condition split (2026-07-11): is the
        # fogless mix doing work? villages_per_turn = time-averaged
        # villages owned (both sides summed) -- the capture-activity
        # curve the never-meet pathology flatlines at ~0.
        "ladder_fog_games", "ladder_fog_decisive",
        "ladder_fog_villages_per_turn", "ladder_fog_villages_end",
        "ladder_fog_approach",
        "ladder_fogless_games", "ladder_fogless_decisive",
        "ladder_fogless_villages_per_turn", "ladder_fogless_villages_end",
        "ladder_fogless_approach",
        # NB (2026-07-14): ladder_*/other_*/ladder_fog*/ladder_fogless*
        # columns count FRESH games only; midgame continuations are
        # reported in midgame_games/midgame_decisive exclusively.
        # Engagement telemetry aggregates (2026-07-12 user spec; the
        # per-game detail incl. kill breakdowns lives in the per-
        # iteration games.jsonl). *_pg = mean per game, both sides
        # summed unless it is a fraction (then side-averaged).
        # eng_invalid/rejected_attacks are iteration SUMS: tripwires
        # for mask/sim/Wesnoth divergence, expected 0.
        "midgame_games", "midgame_decisive",
        "eng_games", "eng_attacks_pg",
        "eng_invalid_attacks", "eng_rejected_attacks",
        "eng_damage_pg", "eng_kills_pg", "eng_kills_value_pg",
        "eng_heal_village_pg", "eng_heal_rest_pg", "eng_heal_ability_pg",
        "eng_advancements_pg",
        "eng_poison_cured_pg", "eng_poison_damage_pg",
        "eng_contact_rate", "eng_first_contact_turn",
        "eng_scouted_frac",
        # Per-side watch metrics (2026-07-20): unspent-MP fraction
        # and time-averaged treasury at the side's own end_turns.
        "eng_unused_mp_s1", "eng_unused_mp_s2",
        "eng_gold_bank_s1", "eng_gold_bank_s2",
        "eng_villages_frac", "eng_material_end_pg",
        "search_q_spread", "search_overturn_frac",
        # Boundary-consistency telemetry (T1-F, 2026-07-29): mean
        # V(pre)+V(post) over sampled side-switch pairs (~0 =
        # calibrated; +0.4..0.6 = fogged WYSIATI bias).
        "boundary_sum", "boundary_pairs_n",
        # Distillation-target telemetry (2026-08-05 prior-ratchet
        # repair; MCTSPolicy.drain_distill_stats). sharpen_top > 0 =
        # the target re-teaches the prior's own top action sharper
        # (ratchet ON); prior_top80 = collapse-rate endpoint the
        # passivity probes use. Appended LAST; a pre-existing CSV
        # gets rotated to .oldschema by the header-mismatch guard.
        "distill_tgt_entropy", "distill_prior_entropy",
        "distill_sharpen_top", "distill_prior_top80",
        "distill_et_prior", "distill_et_target",
        # KL(target||prior) per decision (2026-08-12 diagnosis
        # instrument): the size of the perturbation the search injects
        # into the policy target. Near-constant = distilling noise;
        # should drop toward 0 on low-spread roots under the one-atom
        # rescale floor.
        "distill_kl_prior",
        # ended_by demix + pool-discard telemetry (2026-08-17 user
        # rulings A5(iii)/A6): a cap game, a stalemate, and a mutual
        # elimination are different events; abandoned_actors counts
        # in-flight games the pool's HARD deadline discarded after
        # the drain grace (0 = drain sufficed).
        "ended_leader", "ended_max_turns", "ended_max_actions",
        "ended_no_progress", "ended_other", "abandoned_actors",
        # TCS linear-link clip telemetry (2026-08-17).
        "link_clip_frac",
        # TCS planning + gate-effectiveness telemetry (2026-08-21:
        # these rode the drain all of leg 4 but were never in this
        # list -- INFO-log only). blind_coord_frac is the in-vivo
        # fog-blindness gauge (E2 baseline ~8% opponent-frame);
        # gate_flip/delta measure what the gate's re-grade changes
        # (the Q7 quantity when projection reval is on);
        # gate_shorten_per_plan is the passivity direction of
        # accepted swaps. gbc/aux: auxiliary losses, same gap.
        "tcs_plans", "tcs_accepts_per_plan", "tcs_replans_per_plan",
        "tcs_projections_per_plan", "tcs_blind_coord_frac",
        "tcs_gate_flip_frac", "tcs_gate_delta_reval",
        "tcs_gate_shorten_per_plan", "gbc_loss", "aux_loss",
        # Plan-tournament telemetry (proposition 1, 2026-08-26). The
        # beta columns calibrate the starvation tripwire before it
        # is armed (user ruling: log first, arm after).
        "pt_tournaments", "pt_arm_prefix_frac",
        "pt_cert_rate", "pt_cert_attempt_rate",
        "pt_cert_replicates_mean", "pt_cert_starved_rate",
        "pt_renorm_factor_mean",
        "pt_beta_mean", "pt_beta_p50", "pt_beta_p90",
        "pt_cert_margin_mean", "pt_margin_mean",
        "pt_abstain_events_per_turn", "pt_forwards_per_turn",
        "pt_challengers_per_tournament", "pt_replans_per_turn",
        "pt_grades_per_tournament", "pt_half_est",
        "pt_cert_rate_f0", "pt_cert_rate_f1", "pt_cert_rate_f2p",
        "pt_cert_attempt_frac_f0", "pt_cert_attempt_frac_f1",
        "pt_cert_attempt_frac_f2p",
        "pt_cert_len_delta_mean", "pt_cert_shorten_rate",
        "pt_half_cap_hit_rate",
        # Value-grounding telemetry (arm VG, 2026-09-01): captures /
        # rollout labels / outcome mix / consistency labels per
        # drain (per-actor means under the pool).
        "ground_games", "ground_captures", "ground_rollouts",
        "ground_censored", "ground_win", "ground_loss", "ground_draw",
        "consist_n", "consist_abs_mean",
        # In-training signal telemetry (2026-09-01): per-source
        # gradient norms + per-step value movement on consulted
        # states (the erosion gauge, ~0.08 = search's 2-atom
        # decision threshold).
        "sig_policy_norm", "sig_value_game_norm",
        "sig_value_ground_norm", "sig_value_consist_norm", "sig_seconds",
        "sig_dv_consult_mean", "sig_dv_consult_n",
        # Arm VG2 principled mixture: estimated bootstrap bias /
        # residual variance, paired-label count, the two new loss
        # terms, and the trust-region multiplier after dual ascent.
        "consist_bias_hat", "consist_sigma2_hat", "consist_pair_n",
        "consist_loss", "trust_loss", "trust_lambda",
        "consist_var_diff", "consist_roll_noise",
        "consist_sigma2_point", "consist_sigma2_se",
        "consist_head_minus_truth", "consist_label_minus_truth",
        # Per-turn-decade fresh-probe decomposition (user ruling
        # 2026-09-01): fresh CE / state-blind floor / outcome AUC /
        # n per game-turn decade; the pooled fresh_value_ce column
        # stays the usual read. d61p pools turns 61+.
    ] + [f"fresh_{m}_{d}"
         for d in ("d1_10", "d11_20", "d21_30", "d31_40",
                   "d41_50", "d51_60", "d61p")
         for m in ("ce", "floor", "auc", "n")]

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Appending rows with MORE columns than the file's existing
        # header silently orphans the extra cells (DictReader maps
        # them to None) -- exactly what hid the holdout column on the
        # 2026-07-06 Vast run, whose clone shipped an old-header CSV.
        # On mismatch, rotate the old file aside and start fresh.
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("r", encoding="utf-8", newline="") as f:
                existing = f.readline().rstrip("\r\n").split(",")
            if existing != self.COLUMNS:
                rotated = self.path.with_suffix(
                    self.path.suffix + ".oldschema")
                self.path.replace(rotated)
                logging.getLogger("sim_self_play").warning(
                    f"trainer-history CSV header mismatch; rotated "
                    f"old file to {rotated.name}")
        # Open in append mode; write header only if file is empty
        # (or doesn't exist). Line buffering (buffering=1) so each
        # row is flushed; cluster jobs that get walltime-killed
        # leave the partial CSV recoverable.
        write_header = (not self.path.exists()
                        or self.path.stat().st_size == 0)
        self._f = self.path.open("a", encoding="utf-8",
                                 newline="", buffering=1)
        import csv as _csv
        self._writer = _csv.DictWriter(self._f, fieldnames=self.COLUMNS,
                                       extrasaction="ignore")
        if write_header:
            self._writer.writeheader()

    def append(self, snapshot: Dict) -> None:
        import datetime as _dt
        row = dict(snapshot)
        row["timestamp"] = _dt.datetime.now().isoformat(timespec="seconds")
        ts = snapshot.get("train_stats")
        if ts is not None:
            row["train_n_trajectories"] = ts.n_trajectories
            row["train_n_transitions"]  = ts.n_transitions
            row["train_loss"]           = ts.total_loss
            row["train_policy_loss"]    = ts.policy_loss
            row["train_value_loss"]     = ts.value_loss
            row["train_entropy"]        = ts.entropy
            row["train_mean_return"]    = ts.mean_return
            row["train_grad_norm"]      = ts.grad_norm
            # Auxiliary-objective losses: computed all of leg 4,
            # written nowhere (workflow finding 2026-08-20).
            row["gbc_loss"] = getattr(ts, "gbc_loss", None)
            row["aux_loss"] = getattr(ts, "aux_loss", None)
        # Per-component reward sums (from WeightedReward._component_acc).
        acc = snapshot.get("reward_components") or {}
        for k in self._REWARD_COMPONENT_KEYS:
            v = acc.get(k)
            if v is not None:
                row[f"r_{k}"] = v
        # Econ diagnostics for recruit-underuse investigation.
        for k in self._ECON_KEYS:
            v = snapshot.get(k)
            if v is not None:
                row[k] = v
        # `extrasaction="ignore"` drops the `train_stats` object
        # itself; csv would otherwise stringify it as the repr.
        self._writer.writerow(row)

    def close(self) -> None:
        try:
            self._f.close()
        except OSError:
            pass


def _default_history_csv() -> Path:
    """Default path for the per-iter CSV. SLURM jobs get a unique
    file per job so chain links don't overwrite each other; local
    runs share one rolling `trainer_history_local.csv` so iterating
    locally accumulates."""
    import os as _os
    jobid = _os.environ.get("SLURM_JOB_ID")
    name = (f"trainer_history_{jobid}.csv" if jobid
            else "trainer_history_local.csv")
    return Path("training/logs") / name


def _parse_time_budget(spec: Optional[str]) -> Optional[int]:
    """Parse a wall-time budget string into seconds. Accepts:

      - None / empty -> None (no budget)
      - raw integer seconds:        "13800"
      - MM:SS:                      "50:00"      (50 minutes)
      - HH:MM:SS:                   "03:50:00"   (3h 50m)

    Returns the budget in integer seconds, or None when the input is
    None / empty. Raises ValueError on a malformed spec so the
    caller can fail loudly rather than silently disabling the
    budget.
    """
    if not spec:
        return None
    s = spec.strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 2:
            mm, ss = parts
            return int(mm) * 60 + int(ss)
        if len(parts) == 3:
            hh, mm, ss = parts
            return int(hh) * 3600 + int(mm) * 60 + int(ss)
        raise ValueError(
            f"--time-budget {spec!r}: expected 'HH:MM:SS', 'MM:SS', "
            f"or integer seconds"
        )
    return int(s)


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-in", type=Path, default=None,
                    help="Optional starting checkpoint. Default: random init.")
    ap.add_argument("--checkpoint-out", type=Path,
                    default=Path("training/checkpoints/sim_selfplay.pt"),
                    help="Where to write checkpoints.")
    ap.add_argument("--reset-decision-step", action="store_true",
                    help="After loading --checkpoint-in, reset the "
                         "training-progress counter (decision_step) to 0. "
                         "Use when warm-starting a checkpoint as WEIGHTS "
                         "ONLY for a fresh training campaign: the combat-"
                         "oracle anneal (combat_alphas_at) then runs from "
                         "full strength over ~COMBAT_ANNEAL_HORIZON "
                         "decisions instead of inheriting a counter that "
                         "may already be past the anneal floor. Do NOT use "
                         "when resuming an in-progress run -- it would "
                         "restart the anneal mid-training.")
    ap.add_argument("--iterations", type=int, default=10,
                    help="Hard ceiling on train_step iterations. When "
                         "`--time-budget` is set, this is mostly a "
                         "safety cap -- the time budget will normally "
                         "exit first.")
    ap.add_argument("--trainer-history-csv", type=Path, default=None,
                    help="Path to append one CSV row per iter with "
                         "rollout + train stats. Default: "
                         "training/logs/trainer_history_<SLURM_JOB_ID>.csv "
                         "on cluster, training/logs/trainer_history_"
                         "local.csv otherwise. Pass an empty string "
                         "(--trainer-history-csv='') to disable.")
    ap.add_argument("--time-budget", type=str, default=None,
                    help="Stop after this much elapsed wall time, "
                         "saving a final checkpoint before exit. "
                         "Accepts 'HH:MM:SS', 'MM:SS', or a raw "
                         "integer of seconds (e.g. '03:50:00' = "
                         "13800 = 3h50m). On cluster jobs this is "
                         "the right way to bound runtime -- pair with "
                         "a large --iterations ceiling so the time "
                         "budget is the practical exit. Leave None "
                         "(default) to run until --iterations runs "
                         "out or the operator kills the process.")
    ap.add_argument("--games-per-iter", type=int, default=4,
                    help="Self-play games rolled out per train_step.")
    ap.add_argument("--max-turns", type=int, default=100,
                    help="Per-game turn cap (upper end of the jitter "
                         "range, see --max-turns-min). DEFAULT 100 "
                         "with min 60 = the A3 jittered 60-100 cap "
                         "as CODE DEFAULT (user ruling 2026-08-17: "
                         "the 60-100 jitter is standing training "
                         "behavior; leg 3 accidentally ran [60,200] "
                         "because the launch env carried the min "
                         "flag but not this one -- values this "
                         "load-bearing don't live in launch flags). "
                         "Capped games carry zero value weight, so "
                         "turns past ~100 are pure planning-compute "
                         "waste.")
    ap.add_argument("--no-progress-turns", type=int, default=0,
                    help="Stalemate rule (chess 50-move analog, "
                         "2026-07-21): end a game as a draw after N "
                         "consecutive FULL turns with no objective "
                         "progress (no damage, kill, recruit, or "
                         "village-ownership change). 0 = rule off; "
                         "the tracker still logs would-fire stats "
                         "per game (games.jsonl 'noprogress').")
    ap.add_argument("--max-turns-min", type=int, default=60,
                    help="Per-game turn-cap jitter: each training "
                         "game's cap is drawn uniformly from "
                         "[this, --max-turns] (anti-horizon-gaming, "
                         "2026-07-20: a FIXED cap taught banking "
                         "until a known last turn). DEFAULT 60 "
                         "(user ruling 2026-08-17, standing 60-100 "
                         "jitter). Pass 0 for a fixed cap. Training "
                         "only; eval/demo caps stay fixed.")
    ap.add_argument("--save-every", type=int, default=10,
                    help="Save checkpoint every N iterations.")
    ap.add_argument("--holdout-size", type=int, default=0,
                    help="MCTS only: divert whole games into a frozen "
                         "held-out set until it reaches N experiences, "
                         "then log the net's value CE on it each iter "
                         "(generalization probe -- the train value "
                         "loss is measured on replay samples the net "
                         "already fit). 0 = off. PERSISTED across resumes "
                         "via the <checkpoint>.holdout sidecar "
                         "(crash-safe partial saves), so the curve's "
                         "baseline survives restarts.")
    ap.add_argument("--holdout-per-game-cap", type=int, default=64,
                    help="Max states RANDOMLY SAMPLED into the holdout "
                         "from each diverted game (default 64), so a "
                         "512-state holdout spans ~8 games instead of "
                         "~2 whole ones (2026-07-07: a 2-game holdout "
                         "measured those games' idiosyncrasies, not "
                         "generalization). Diverted games' remaining "
                         "states are discarded, not trained on.")
    ap.add_argument("--value-label-smoothing", type=float, default=0.0,
                    help="Mix this much uniform mass into the C51 "
                         "value TRAIN target (eval CE stays "
                         "unsmoothed). Counters extreme-atom collapse "
                         "under many replay updates on hard terminal "
                         "targets (2026-07-07 diagnosis: Z entropy "
                         "1.86->1.13 in 46 iters while holdout CE "
                         "diverged). Try 0.02. 0 = off (default).")
    ap.add_argument("--abort-k-median", type=float, default=None,
                    help="K-collapse tripwire (user 2026-08-21): abort "
                         "(exit 7) when the iteration's median "
                         "actions-per-side-turn is below this for 3 "
                         "consecutive iterations. Leg 3 collapsed to "
                         "K~2 with every other guard green. DEFAULT "
                         "OFF by user ruling -- pass explicitly (e.g. "
                         "10) on future training legs.")
    ap.add_argument("--abort-decisive-rate", type=float, default=None,
                    help="Abort tripwire: once the trailing "
                         "--abort-window iterations are full, stop if "
                         "the fraction of decisive (non-draw) games "
                         "falls below this value. Saves a final "
                         "checkpoint and exits with code 4 so a "
                         "wrapper can distinguish 'tripwire' from "
                         "'done'. Guards paid GPU runs against the "
                         "known all-draws failure shape (see "
                         "tier_a_runbook.md). Off by default.")
    ap.add_argument("--abort-window", type=int, default=20,
                    help="Trailing iteration count for "
                         "--abort-decisive-rate (default 20). The "
                         "tripwire only arms once this many "
                         "iterations have completed, so it doubles "
                         "as the burn-in period.")
    ap.add_argument("--abort-holdout-stall", type=int, default=None,
                    help="Memorization tripwire (needs --replay-buffer "
                         "fresh-CE telemetry): stop if the FLOOR-"
                         "RELATIVE fresh holdout CE (fresh_value_ce - "
                         "fresh_ce_floor) has not made a new best (by "
                         "--abort-holdout-min-delta) for this many "
                         "consecutive iterations. Saves a final "
                         "checkpoint and exits with code 5. The "
                         "2026-07-02 Kaggle data motivates it: train "
                         "value loss fell 3.8->1.15 while holdout CE "
                         "sat flat at ~3.1 -- pure buffer memorization. "
                         "FLOOR-RELATIVE because raw CE moves with the "
                         "outcome-label mix (the raw-CE version "
                         "mis-fired twice in the 72h run; A5 spec, "
                         "user ruling 2026-08-10). Pick a GENEROUS "
                         "window: short windows false-trip on long "
                         "runs.")
    ap.add_argument("--abort-holdout-min-delta", type=float,
                    default=0.01,
                    help="Improvement below this doesn't count as a "
                         "new holdout best (default 0.01 CE nats; "
                         "guards against noise resetting the stall "
                         "counter).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed of the run: the scenario draws, torch (a fresh "
                         "network's init), the search's root sampling and noise "
                         "(MCTSPolicy rng_seed) and replay sampling, so a run "
                         "repeats given the same seed.")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING"])
    ap.add_argument("--no-map-settings", dest="use_map_settings",
                    action="store_false",
                    help="Do NOT override the source replay's economy "
                         "settings with PvP defaults. Use this when "
                         "you want the sim to track the source's "
                         "actual settings (e.g. for replay-recon "
                         "smoke tests). Default: ON -- self-play uses "
                         "standard 2p ladder defaults regardless of "
                         "source-replay quirks.")
    ap.set_defaults(use_map_settings=True)
    ap.add_argument("--starting-gold", type=int, default=100,
                    help="PvP defaults: starting gold per side.")
    ap.add_argument("--village-gold", type=int, default=2,
                    help="PvP defaults: gold per village per turn.")
    ap.add_argument("--village-support", type=int, default=1,
                    help="PvP defaults: upkeep absorbed per village.")
    ap.add_argument("--value-coef", type=float, default=None,
                    help="Override the trainer's value-loss weight "
                         "(TrainerConfig default 0.5). Higher weights "
                         "value learning harder -- relevant because "
                         "the value head is the diagnosed training "
                         "bottleneck (2026-06-15). Default None = leave "
                         "at 0.5.")
    ap.add_argument("--exp-modifier", type=int, default=70,
                    help="PvP defaults: experience required modifier (%%).")

    # Reward customization. Defaults to "off" (existing WeightedReward
    # defaults) so omitting the flag reproduces prior behavior.
    # REINFORCE-mode only: shaping rewards are structurally inert
    # under --mcts (MCTSPolicy.observe is a no-op), so a non-default
    # config there is refused at startup (see main).
    ap.add_argument("--reward-config", type=Path, default=None,
                    help="Path to a JSON or YAML reward-shaping config "
                         "(see rewards.load_reward_config). Lets you "
                         "tune weights, add per-unit-type recruit "
                         "bonuses, and define turn-conditional "
                         "bonuses without code edits. Default: use "
                         "WeightedReward() defaults. REINFORCE-mode "
                         "only; refused under --mcts (shaping is "
                         "inert there -- the live seams are "
                         "--draw-tiebreak-cap and the terminal z).")
    ap.add_argument("--torch-threads", type=int, default=None,
                    help="Cap torch's intra-op CPU thread pool "
                         "(torch.set_num_threads). CPU-ONLY tuning: by "
                         "default this is applied only on a CPU device "
                         "(to ~4, or $WAI_TORCH_THREADS) -- on the tiny "
                         "per-leaf states the all-cores pool loses to "
                         "oversubscription (measured ~1.3-2.3x on CPU). "
                         "On a CUDA/DML device it is NOT capped by "
                         "default (the CPU-side MCTS/encoding wants all "
                         "cores while the GPU does forwards). Pass an "
                         "explicit value to force a cap on ANY device; "
                         "pass 0 to force torch's default. Only affects "
                         "float reduction order (~1e-7); combat RNG / "
                         "state_key / mask are integer + untouched.")
    ap.add_argument("--train-batch-size", type=int, default=None,
                    help="forward_batch chunk size for the trainer "
                         "(TrainerConfig.train_batch_size). THE key GPU "
                         "knob: default None is device-aware -- 1 on "
                         "CPU (batching these shapes doesn't help), 2 on "
                         "DML (AMD-driver-stable), 128 on CUDA so the "
                         "replay minibatch forwards as ONE batched call "
                         "instead of 128 batch-1 calls (near-worst GPU "
                         "utilization otherwise). Raise on a large GPU; "
                         "lower if you hit OOM.")
    # --- fresh-init architecture (model scaling, plan §3.2) ----------
    # Default None = "not specified" -> fall back to the checkpoint's
    # saved arch (warm-start) or TransformerPolicy's ctor default
    # (fresh). Provide these to scale the net up for a campaign; the
    # 0.47M weights WON'T load into a wider/deeper net, so passing a
    # size that differs from --checkpoint-in starts FRESH at the
    # requested size (logged). See docs/archive/superhuman_training_plan.md
    # §3.2 for param targets (Tier-a ~3-10M = 384/6/8/1536, etc.).
    ap.add_argument("--d-model", type=int, default=None,
                    help="Transformer width for FRESH init (model "
                         "scaling). Default None -> checkpoint arch if "
                         "warm-starting, else TransformerPolicy default. "
                         "Differing from --checkpoint-in starts fresh.")
    ap.add_argument("--num-layers", type=int, default=None,
                    help="Transformer depth for fresh init (see "
                         "--d-model).")
    ap.add_argument("--num-heads", type=int, default=None,
                    help="Attention heads for fresh init (see "
                         "--d-model). Must divide --d-model.")
    ap.add_argument("--d-ff", type=int, default=None,
                    help="Feed-forward hidden dim for fresh init (see "
                         "--d-model).")
    ap.add_argument("--device", default=None,
                    help="Torch device for the policy. Accepts "
                         "'cuda' (cluster), 'cpu', 'dml' (local "
                         "AMD/Intel GPU via DirectML -- ~4x CPU "
                         "speed on the RX 6600 since the scatter "
                         "backward fix), or 'auto' (DML > CUDA > "
                         "CPU). Default None lets TransformerPolicy "
                         "pick (CPU).")
    ap.add_argument("--workers", type=int, default=0,
                    help="Rollout worker threads per iteration. 0 "
                         "(default) = serial on the main thread. "
                         ">= 1 = spawn N workers feeding trajectories "
                         "to the policy concurrently. Safe via the "
                         "policy's snapshot+lock design (see "
                         "transformer_policy.TransformerPolicy._lock). "
                         "On CPU, ~30%% speedup at workers=4 (GIL "
                         "limits gains); on GPU, larger speedup as "
                         "forwards dispatch in parallel.")
    ap.add_argument("--actor-pool", type=int, default=0,
                    help="MCTS only. >0 = run self-play in N "
                         "WEIGHTLESS actor PROCESSES feeding a central "
                         "batched-inference server (this main process "
                         "owns the GPU and batches forwards across "
                         "actors -- SEED-RL pattern, tools/actor_pool). "
                         "Escapes the GIL that caps --workers threads "
                         "and keeps a GPU fed on a CPU-rich host. "
                         "Mutually exclusive with --workers. Training "
                         "is no longer bit-deterministic (dynamic "
                         "cross-actor batching).")
    ap.add_argument("--actor-max-batch", type=int, default=0,
                    help="Max LEAVES the actor-pool server fuses per "
                         "GPU forward (requests are batch-granular "
                         "since 2026-07-22). 0 (default) = "
                         "max(64, 2 * --actor-pool).")
    ap.add_argument("--forced-faction", default=None,
                    help="If set, every game has at least one side "
                         "playing this faction. Pass 'none' to "
                         "explicitly disable the module default "
                         "(currently 'Knalgan Alliance'). Pass any "
                         "default-era faction name (e.g. 'Drakes') "
                         "to lock that faction instead.")
    ap.add_argument("--mini-maps", action="store_true",
                    help="Restrict scenario sampling to the smallest "
                         "5 Ladder maps (Sablestone Delta, Weldyn "
                         "Channel, Den of Onis, Swamp of Dread, "
                         "Hamlets). Cells per map: 690-870 vs the "
                         "full pool's 690-2352. Used for the "
                         "engagement-curriculum phase: leaders "
                         "start ~12-15 hexes apart, so the policy "
                         "can discover engagement before the "
                         "long-march cost dominates. See "
                         "scenario_pool.MINI_MAP_SCENARIO_IDS.")
    # Training-mix ratios (2026-07-20 redesign): each is an ABSOLUTE
    # fraction of ALL games -- one categorical roll per game, no
    # cascading remainders. The four ratios (mini, midgame,
    # fogless, ladder) must sum to exactly 1 or startup errors.
    ap.add_argument("--mini-ratio", type=float, default=0.0,
                    help="Absolute fraction of all games sampled from "
                         "the mini-maps pool (cheap engagement "
                         "signal). Ignored when --mini-maps is set "
                         "(already 100%% mini). All five *-ratio "
                         "flags must sum to 1.")
    ap.add_argument("--midgame-ratio", type=float, default=0.0,
                    help="Absolute fraction of all games starting "
                         "from a human-corpus MID-GAME position "
                         "(uniform game, uniform turn in [1, end]) "
                         "played out by self-play -- backward "
                         "curriculum vs the never-meet stalemate "
                         "equilibrium (2026-07-12). Requires the "
                         "value corpus at --midgame-dataset.")
    ap.add_argument("--ladder-ratio", type=float, default=1.0,
                    help="Absolute fraction of all games on the "
                         "regular fogged ladder pool (the production "
                         "distribution). Default 1.0 = pure fogged "
                         "ladder; lower it explicitly when raising "
                         "the other ratios so the five sum to 1.")
    ap.add_argument("--midgame-dataset", type=Path,
                    default=Path("replays_dataset"),
                    help="Value-corpus dir (index + json.gz games).")
    ap.add_argument("--validate-export-every", type=int, default=100,
                    help="Export every Nth finished game PER CATEGORY "
                         "(mini / ladder / ladder_fogless / midgame) "
                         "as a Wesnoth-loadable replay for offline "
                         "strict-sync verification (user spec "
                         "2026-07-15). 0 disables. Counters are "
                         "per process.")
    ap.add_argument("--validate-export-dir", type=Path,
                    default=Path("training/validate_exports"),
                    help="Root dir for validation replay exports "
                         "(one subdir per category; swept by the HF "
                         "uploader on training boxes).")
    ap.add_argument("--game-record-dir", type=Path,
                    default=Path(os.environ.get("WESNOTH_GAME_RECORD_DIR",
                                                "training/game_records")),
                    help="Every finished game is recorded whole here "
                         "(tools/game_record.py), one gzip JSON-lines file "
                         "per process under a per-run subdirectory. Pass "
                         "an empty string to disable. Default: "
                         "$WESNOTH_GAME_RECORD_DIR, else training/game_records "
                         "(the test suite points the variable at a temporary "
                         "directory).")
    ap.add_argument("--game-log-dir", type=Path,
                    default=Path("training/logs/games"),
                    help="Per-game JSONL telemetry root; each "
                         "iteration writes its own subdirectory "
                         "(iter_NNNNNN/games.jsonl). Pass an empty "
                         "string to disable.")
    ap.add_argument("--fogless-ratio", type=float, default=0.0,
                    help="Absolute fraction of all games played on "
                         "the ladder pool with fog of war OFF "
                         "(mini games always keep fog). "
                         "Full-information games give the value head "
                         "mutually-visible armies -- an engagement-"
                         "learning aid.")
    # MCTS-mode flags. Default ON; --reinforce runs the REINFORCE path.
    # Under --mcts, action selection runs an AlphaZero-style
    # tree search and the trainer minimizes CE against visit-count
    # distributions instead of policy gradient. Per-step shaping
    # rewards are silently ignored in MCTS mode (AlphaZero distills
    # the terminal z onto every visited state); --reward-config is
    # still parsed so REINFORCE / MCTS configs can share JSON files.
    ap.add_argument("--mcts", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="MCTS action selection + distillation "
                         "targets. DEFAULT ON since 2026-08-10 (user "
                         "decision, technique review D1): the default "
                         "now describes the measured production path, "
                         "so a bare invocation is a small campaign, "
                         "not a different algorithm. Adds N_sim model "
                         "forwards per move. Use --reinforce (or "
                         "--no-mcts) for the legacy REINFORCE path — "
                         "the light smoke-test flow and the only "
                         "consumer of the shaping-reward seam.")
    ap.add_argument("--reinforce", dest="mcts", action="store_false",
                    help="Alias for --no-mcts: legacy REINFORCE "
                         "training (raw policy sampling, shaping "
                         "rewards live).")
    ap.add_argument("--plan-tournament",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="Incumbent-anchored turn-plan tournament with "
                         "certify-or-abstain distillation (proposition "
                         "1, user-approved 2026-08-26; "
                         "tools/plan_tournament.py). The policy's own "
                         "sampled turn plays unless a challenger "
                         "certifies an improvement under paired "
                         "projection grading; certified turns distill "
                         "at policy_weight=beta, abstained turns are "
                         "value-only. Takes precedence over "
                         "--turn-search.")
    ap.add_argument("--pt-challengers", type=int, default=6)
    ap.add_argument("--pt-depths", type=str, default="1,3",
                    help="Comma-separated SELECTION depths (odd "
                         "half-turns; even/zero rejected -- "
                         "own-frame invariant).")
    ap.add_argument("--pt-redraws", type=int, default=1)
    ap.add_argument("--pt-cert-depth", type=int, default=3,
                    help="Fixed certification depth (odd half-turns).")
    ap.add_argument("--pt-cert-redraws", type=int, default=3,
                    help="Independent certification pairs (n-aware "
                         "accept_rule; clamped to the _T_CRIT "
                         "table range).")
    ap.add_argument("--pt-budget-forwards", type=int, default=900,
                    help="Hard per-side-turn forward cap; exhaustion "
                         "abstains (fixed-budget rule 2026-08-26).")
    ap.add_argument("--pt-margin-band", type=float, default=0.08,
                    help="Certification band (pre-registered v1: 2 "
                         "C51 atoms; branch audit will replace).")
    ap.add_argument("--pt-beta-max", type=float, default=0.25)
    ap.add_argument("--pt-margin-ref", type=float, default=0.32)
    ap.add_argument("--turn-search", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Turn-Commitment Search (docs/archive/tcs_spec.md): "
                         "plan complete side-turns by counterfactual "
                         "coordinate refinement graded at turn "
                         "boundaries, instead of a per-micro-action "
                         "Gumbel root search. DEFAULT ON (user ruling "
                         "2026-08-14, after the rung-1 probe: "
                         "revalidated accept 0.64, median accepted "
                         "delta ~2 C51 atoms, placebo-separated). "
                         "Rides --mcts mode (targets, trainer, buffer "
                         "unchanged); --no-turn-search restores the "
                         "per-decision Gumbel MCTS generator.")
    ap.add_argument("--turn-alt", type=int, default=4,
                    help="TCS alternatives per coordinate per round.")
    ap.add_argument("--turn-rounds", type=int, default=3,
                    help="TCS hill-climb rounds on full turns.")
    ap.add_argument("--turn-fast-rounds", type=int, default=1,
                    help="TCS rounds on cheap (no-target) turns -- "
                         "the playout-cap analog's fast tier.")
    ap.add_argument("--turn-reval-salts", type=int, default=3,
                    help="Fresh salts in TCS acceptance stage 2.")
    ap.add_argument("--turn-min-delta", type=float, default=0.01,
                    help="TCS accept floor (float-jitter guard).")
    ap.add_argument("--turn-full-prob", type=float, default=0.25,
                    help="Fraction of TURNS planned at full budget "
                         "with targets recorded (playout-cap analog).")
    ap.add_argument("--turn-project", choices=("none", "reval", "all"),
                    default="none",
                    help="Multi-turn projection (tcs_spec.md par.3, "
                         "2026-08-17): grade candidate turns by the "
                         "value --turn-project-halfturns half-turns "
                         "past our boundary, each played closed-loop "
                         "by the same policy (linear cost, no "
                         "branching) -- the guard against value-head "
                         "tempo blindness. 'reval' gates stage-2 "
                         "acceptance only; 'all' also drives stage-1 "
                         "selection and the distill targets. OFF by "
                         "default.")
    ap.add_argument("--turn-project-halfturns", type=int, default=1,
                    help="Projection depth in half-turns.")
    ap.add_argument("--turn-project-max-actions", type=int, default=40,
                    help="Per projected half-turn action cap "
                         "(end_turn forced at the cap).")
    ap.add_argument("--pool-drain-grace", type=float, default=1800.0,
                    help="Actor-pool drain window (s): at the "
                         "iteration deadline actors FINISH their "
                         "current game (drain) instead of having it "
                         "discarded; this grace bounds the overrun "
                         "before the hard abandon backstop (user "
                         "ruling 2026-08-17; leg 3 discarded 30-60%% "
                         "of some iterations' games at the old "
                         "single hard deadline).")
    ap.add_argument("--turn-boundary-frame",
                    choices=("opponent", "mover"), default="opponent",
                    help="TCS boundary evaluation frame (2026-08-21 "
                         "fog finding): 'opponent' (status quo) "
                         "grades the post-end_turn state through the "
                         "OPPONENT's fogged view -- structurally "
                         "blind to whatever the opponent cannot see "
                         "of the mover's turn. 'mover' grades the "
                         "PRE-end_turn state from the mover's own "
                         "information set. Default stays 'opponent' "
                         "until the probes re-baseline; leg-5 config "
                         "must assert this explicitly.")
    ap.add_argument("--turn-target-link", choices=("linear", "exp"),
                    default="linear",
                    help="TCS distill-target link function (user "
                         "ruling 2026-08-17: evaluation exposure "
                         "must not buy probability under an "
                         "uninformative grader). 'linear' (DEFAULT): "
                         "prior^lam * max(0, 1 + beta*(q - LOO mean "
                         "of the other evaluated q)) -- exposure-"
                         "invariant, noise-robust. 'exp': the "
                         "AlphaZero sigma tilt shared with the MCTS "
                         "path -- concentrates faster but pays the "
                         "evaluated-actions convexity bonus under a "
                         "noisy value head (the leg-3 end_turn "
                         "ratchet).")
    ap.add_argument("--turn-target-beta", type=float, default=5.0,
                    help="Linear-link advantage gain (see "
                         "docs/design_constants.md: 5 C51 atoms "
                         "below the evaluated peers' mean clips to "
                         "zero mass).")
    ap.add_argument("--turn-reply", choices=("none", "reval", "all"),
                    default="none",
                    help="DEPRECATED alias: depth-1 projection with "
                         "--turn-reply-max-actions as the cap. Use "
                         "--turn-project.")
    ap.add_argument("--turn-reply-max-actions", type=int, default=4)
    ap.add_argument("--turn-max-spine", type=int, default=40,
                    help="Hard cap on TCS spine length.")
    ap.add_argument("--gbc", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="GBC event supervision (docs/archive/gbc_spec.md): "
                         "small heads predict fog-censored dies/flips "
                         "within k turns from hindsight labels; their "
                         "BCE gradient repairs the trunk's value-"
                         "relevant features (approved as the value-"
                         "head repair after the 0d attribution test: "
                         "events predict outcomes at AUC 0.79 while "
                         "the head's turn movement is noise). DEFAULT "
                         "ON (user ruling 2026-08-14). Adds heads to "
                         "the model (checkpoint-sticky, aux "
                         "precedent) + a label pass per finalized "
                         "game.")
    ap.add_argument("--gbc-coef", type=float, default=0.1,
                    help="Weight of the GBC event-supervision BCE in "
                         "the training loss.")
    ap.add_argument("--mcts-sims", type=int, default=50,
                    help="Number of MCTS simulations per move "
                         "(--mcts only). 50 is a reasonable "
                         "starting budget; AlphaZero used 800 for "
                         "chess, but Wesnoth's branching factor and "
                         "model-forward latency push us to fewer "
                         "sims early in development.")
    ap.add_argument("--mcts-c-puct", type=float, default=1.5,
                    help="PUCT exploration constant (--mcts only).")
    ap.add_argument("--distill-prior-discount", type=float, default=1.0,
                    help="Lambda on log(prior) in the Gumbel "
                         "distillation target (--mcts only; 1.0 = "
                         "legacy). <1 damps the target's self-"
                         "reference: equilibrium prior sharpness "
                         "becomes sigma_gap/(1-lambda), bounded by "
                         "value evidence -- the prior-ratchet repair "
                         "(2026-08-05). Try 0.9.")
    ap.add_argument("--mcts-hierarchical-gumbel", action="store_true",
                    help="Two-level Gumbel root: actors compete with "
                         "full prior mass, then edges within actors. "
                         "Default off; A/B lever (BACKLOG 3c).")
    ap.add_argument("--mini-random-tod", action="store_true",
                    help="Force a RANDOM start-ToD slot on the "
                         "fixed-ToD mini templates (the 3 passivity-"
                         "asymmetry maps). De-confound lever, BACKLOG "
                         "3c; env-inherited by the pool's actors.")
    ap.add_argument("--infer-bf16", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="bf16 autocast for INFERENCE forwards "
                         "(trainer stays fp32; outputs cast back to "
                         "fp32). Training default OFF (2026-08-29: "
                         "unvalidated on the pool path; eval keeps "
                         "the 2026-08-28 cuda-auto compile+bf16 "
                         "default).")
    ap.add_argument("--infer-compile", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="torch.compile the inference model "
                         "(reduce-overhead). Training default OFF "
                         "(2026-08-29: an in-process compile "
                         "deadlocked the spool e2e on cuda; flip "
                         "after a pool-path validation smoke). "
                         "Kernel cache via TORCHINDUCTOR_CACHE_DIR.")
    ap.add_argument("--distill-target-temp", type=float, default=1.0,
                    help="Temperature dividing the Gumbel "
                         "distillation-target logits (--mcts only; "
                         "1.0 = legacy). >1 softens the whole "
                         "target incl. the value term; prefer "
                         "--distill-prior-discount.")
    ap.add_argument("--mcts-batch-size", type=int, default=None,
                    help="Batched leaf evaluation (--mcts only). Default is "
                         "device-aware: B=1 on CPU, B=16 on CUDA (a batched "
                         "leaf forward amortizes per-forward kernel-launch + "
                         "host-sync overhead — the per-leaf forward is the "
                         "dominant GPU cost of an --mcts run, so leaving it "
                         "at 1 on CUDA starves the device). Pass an explicit "
                         "value to override; profile on the GPU node to pick "
                         "B (8-32). Applies to BOTH the default Gumbel root "
                         "(each sequential-halving phase evaluates its leaves "
                         "through one forward_batch with virtual loss) and "
                         "the classic root. Composes with "
                         "--workers (cross-game batching).")
    ap.add_argument("--mcts-fpu-reduction", type=float, default=0.25,
                    help="First-play urgency: unvisited edges score "
                         "as (parent value - this) instead of 0, so "
                         "small sim budgets deepen instead of "
                         "sweeping every legal action once. "
                         "Negative value disables (legacy Q=0).")
    ap.add_argument("--mcts-temperature", type=float, default=1.0,
                    help="Root sampling temperature for the first "
                         "--mcts-temperature-decisions of each game "
                         "(AlphaZero tau). <=0 = always argmax.")
    ap.add_argument("--mcts-temperature-decisions", type=int,
                    default=30,
                    help="How many decisions per game are sampled "
                         "proportional to visits^(1/tau) before "
                         "switching to argmax-visits.")
    ap.add_argument("--mcts-classic-root", action="store_true",
                    help="Use classic AlphaZero root handling "
                         "(Dirichlet noise + visit-count temperature "
                         "+ visit-count targets) instead of the "
                         "default Gumbel root (Gumbel-Top-k "
                         "candidates + sequential halving + "
                         "completed-Q targets; Danihelka 2022).")
    ap.add_argument("--mcts-gumbel-m", type=int, default=16,
                    help="Gumbel root: number of candidate actions "
                         "sampled without replacement for "
                         "sequential halving.")
    ap.add_argument("--mcts-gumbel-rescale-floor", type=float,
                    default=0.04,
                    help="Min completed-Q spread the sigma rescale "
                         "divides by. Spreads below it scale the "
                         "target perturbation down proportionally "
                         "(fade to prior on no-signal roots). Default "
                         "0.04 = one C51 atom, the value head's own "
                         "resolution; the legacy 1e-8 amplified value "
                         "noise into a fixed ~5-logit target "
                         "perturbation on every low-signal decision "
                         "(2026-08-12 diagnosis).")
    ap.add_argument("--mcts-no-exact-outcomes", action="store_true",
                    help="Disable exact combat-outcome enumeration "
                         "at chance nodes (tools/combat_outcomes "
                         "prob-matrix DP); falls back to pure "
                         "sampled outcomes.")
    ap.add_argument("--mcts-no-tree-reuse", action="store_true",
                    help="Disable state-key-checked subtree reuse "
                         "across consecutive decisions (on by "
                         "default; reuse only fires when the live "
                         "successor state exactly matches the "
                         "searched child, so combat RNG divergence "
                         "auto-rebuilds).")
    ap.add_argument("--mcts-playout-cap",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Playout-cap randomization (KataGo): only a "
                         "random fraction of self-play moves "
                         "(--mcts-playout-cap-prob) run the full sim "
                         "budget AND record a policy target; the rest "
                         "run a cheap budget (--mcts-playout-cap-fast-"
                         "sims) and record nothing. ~3-10x more games "
                         "per GPU-hour; value targets still attach to "
                         "every recorded full move. ON by default "
                         "(user ruling 2026-08-05) for the TRAINING "
                         "entry point only -- library MCTSConfig and "
                         "eval paths stay uncapped; disable with "
                         "--no-mcts-playout-cap.")
    ap.add_argument("--mcts-playout-cap-prob", type=float, default=0.25,
                    help="P(full-budget, recorded move) under "
                         "--mcts-playout-cap (default 0.25).")
    ap.add_argument("--mcts-playout-cap-fast-sims", type=int, default=0,
                    help="Sim budget for fast (unrecorded) moves under "
                         "--mcts-playout-cap. 0 = max(1, --mcts-sims//4).")
    ap.add_argument("--mcts-aux-score", action="store_true",
                    help="Add the auxiliary margin head (KataGo §3.5): "
                         "the model predicts the final MATERIAL margin "
                         "(tanh, denser than win/loss z), trained with "
                         "an MSE term (--mcts-aux-coef) against "
                         "draw_tiebreak.material_margin. Changes the "
                         "model arch (adds a head) -> fresh init unless "
                         "warm-starting an aux-on checkpoint. Targets "
                         "are produced only in MCTS mode.")
    ap.add_argument("--mcts-aux-coef", type=float, default=0.15,
                    help="Weight of the auxiliary margin MSE loss "
                         "(--mcts-aux-score). KataGo uses ~0.15.")
    ap.add_argument("--relevant-set-hexes",
                    action=argparse.BooleanOptionalAction,
                    default=None,
                    help="Encode only the RELEVANT hexes (unit reach + "
                         "villages + castles + visible units) instead of the "
                         "whole board. Measured: mean 0.30 of the board, "
                         "zero superset violations over 1,840 decisions, "
                         "4.3-4.8x rollout-forward speedup. Changes the "
                         "ACTION SPACE's index basis, so a checkpoint or "
                         "replay buffer built one way is meaningless the "
                         "other; learner and workers MUST agree.")
    ap.add_argument("--mcts-moves-left", action="store_true",
                    help="Add the Lc0-style moves-left head: the model "
                         "predicts the fraction of the turn budget "
                         "still to be played (sigmoid), trained with an "
                         "MSE term (--mcts-moves-left-coef) against the "
                         "game's actual remaining turns. A dense TEMPO "
                         "signal the sparse z can't provide (2026-07-04 "
                         "action-spam diagnosis). Adds a head -> "
                         "partial-load on aux-off checkpoints. The "
                         "search-side utility consumer is separate and "
                         "default-off pending calibration.")
    ap.add_argument("--mcts-moves-left-coef", type=float, default=0.1,
                    help="Weight of the moves-left MSE loss "
                         "(--mcts-moves-left).")
    ap.add_argument("--mcts-aux-value-bonus", type=float, default=0.0,
                    help="Add aux_value_bonus * aux_pred (tanh-bounded "
                         "material margin) to every leaf value the "
                         "search uses, clamped to [-1, 1]. 0.3 spans "
                         "(-0.3, +0.3): material/village gains become "
                         "visible within the search horizon "
                         "(2026-07-11; anatomy showed zero village "
                         "captures without it). NOTE: in gumbel-root "
                         "mode (default) the aux-shaped Q also shapes "
                         "the distilled POLICY TARGET -- intended: "
                         "the policy learns to value material at up "
                         "to 0.3 of a win. 0 = off.")
    ap.add_argument("--mcts-moves-left-utility", type=float, default=0.0,
                    help="Lc0-style search utility: winning lines are "
                         "nudged toward FEWER expected remaining moves "
                         "(and losing lines toward more) by this weight "
                         "in PUCT selection. 0 = off (default). Needs "
                         "the moves-left head (--mcts-moves-left) to "
                         "carry any signal; start ~0.2.")
    ap.add_argument("--replay-buffer", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="AlphaZero-style experience replay + multi-"
                         "epoch training (MCTS mode). DEFAULT ON at "
                         "this CLI since 2026-08-10 (user decision, "
                         "technique review A4): every campaign passed "
                         "it manually, and fresh_value_ce -- the "
                         "success metric -- is only computed on the "
                         "replay path, so a forgotten flag silently "
                         "lost the metric. --no-replay-buffer restores "
                         "the legacy one-pass-then-discard flow "
                         "(debugging fallback; diagnosis 2026-06-15: "
                         "one-pass is severely sample-inefficient, the "
                         "value head stalled at the ~uniform floor). "
                         "Library ReplayConfig default stays False for "
                         "byte-stable tests/eval.")
    ap.add_argument("--replay-updates", type=int, default=8,
                    help="Gradient steps per iteration when "
                         "--replay-buffer is on (default 8).")
    ap.add_argument("--replay-minibatch", type=int, default=128,
                    help="Experiences per gradient step under "
                         "--replay-buffer (default 128).")
    ap.add_argument("--replay-capacity", type=int, default=4000,
                    help="Max experiences retained in the replay "
                         "buffer (default 4000). Each holds a "
                         "deepcopied game state -- watch memory on "
                         "modest hardware.")
    ap.add_argument("--replay-min-size", type=int, default=512,
                    help="Warm up with legacy one-pass training until "
                         "the replay buffer holds this many "
                         "experiences (default 512).")
    ap.add_argument("--train-draw-tiebreak", action="store_true",
                    help="LEGACY: label drawn games' TRAINING z with "
                         "the material tiebreak (search always uses "
                         "the tiebreak regardless). Default off since "
                         "2026-07-10: material-z draw labels made "
                         "'predict material' the dominant value "
                         "lesson and eroded win/loss discrimination.")
    ap.add_argument("--human-anchor-file", type=Path, default=None,
                    help="Pre-encoded human-corpus anchor cache "
                         "(tools/build_human_anchor.py). Enables "
                         "value-only rehearsal steps each iteration.")
    ap.add_argument("--human-anchor-updates", type=int, default=4,
                    help="Rehearsal gradient steps per iteration "
                         "(default 4; vs --replay-updates self-play "
                         "steps).")
    ap.add_argument("--human-anchor-batch", type=int, default=128)
    ap.add_argument("--human-anchor-policy-file", type=Path, default=None,
                    help="Pre-encoded winner-side human pair cache "
                         "(tools/policy_anchor.py). Enables POLICY-"
                         "head rehearsal (the imitation four-head CE) "
                         "each iteration -- the RLPD-shaped prior "
                         "protection. Default OFF; run ONE prior "
                         "protection per leg (this vs "
                         "--distill-prior-discount vs piKL) so the "
                         "human-holdout CE observable stays "
                         "attributable (F1 ruling 2026-08-10).")
    ap.add_argument("--human-anchor-policy-updates", type=int, default=4,
                    help="Policy-rehearsal gradient steps per "
                         "iteration (--human-anchor-policy-file).")
    ap.add_argument("--human-anchor-policy-batch", type=int, default=128)
    ap.add_argument("--draw-value-weight", type=float, default=0.0,
                    help="Weight of drawn games' states in the MCTS "
                         "value loss (aux/moves-left heads always get "
                         "them). 0 = decisive-only value learning "
                         "(2026-07-10: the 71%%-draw gradient mass "
                         "flattened the value head even with honest "
                         "z=0 labels and a rehearsal anchor).")
    ap.add_argument("--signal-telemetry", action=argparse.BooleanOptionalAction, default=True,
                    help="Per-source gradient-norm telemetry every "
                         "iteration (sig_*_norm columns; 4 extra "
                         "backward passes on a 128-state subsample, "
                         "~3-5%%), its cost in the sig_seconds column. "
                         "On by default (user, 2026-09-25: telemetry in "
                         "every trainer; the recorded cost answers the "
                         "2026-09-02 ruling that no overhead runs "
                         "unnoticed). dv_consult logs regardless (the "
                         "VG2 trust region needs it).")
    ap.add_argument("--value-ground", action="store_true",
                    help="Value grounding on search-consulted states "
                         "(arm VG, 2026-09-01; tools/value_grounding). "
                         "Captures stage-2 projection boundary states "
                         "during TCS planning and trains the value "
                         "head on them two ways: raw-policy rollout "
                         "outcomes (full weight) and the search's own "
                         "projected values (down-weighted). Requires "
                         "TCS with --turn-project reval|all.")
    ap.add_argument("--value-ground-capture-prob", type=float,
                    default=0.10,
                    help="Per projected stage-2 evaluation, capture "
                         "probability.")
    ap.add_argument("--value-ground-rollouts", type=int, default=4,
                    help="Rollout-labeled states per game (each costs "
                         "one raw-policy playout at finalize).")
    ap.add_argument("--value-ground-consist", type=int, default=8,
                    help="Consistency-labeled states per game (free).")
    ap.add_argument("--value-ground-weight", type=float, default=1.0,
                    help="value_weight of rollout-labeled states.")
    ap.add_argument("--value-consist-weight", type=float, default=0.25,
                    help="value_weight of projected-label states "
                         "(self-referential; keep low — leg-3 ratchet "
                         "guard).")
    ap.add_argument("--value-memory-iters", type=int, default=0,
                    help="VALUE-head outcome memory span, in "
                         "iterations of games (user ruling "
                         "2026-08-30: the arm-T oscillation was "
                         "diagnosed as value-fit noise from ~35 "
                         "independent game outcomes per update, "
                         "amplified by search's max-operator; the "
                         "weight-averaging test confirmed the "
                         "catastrophic component is noise). Each "
                         "iteration adds one value-only gradient "
                         "step over states sampled game-uniformly "
                         "from the last N iterations' outcomes. "
                         "0 = off (unchanged behavior).")
    ap.add_argument("--value-memory-batch", type=int, default=256,
                    help="States per value-memory gradient step.")
    ap.add_argument("--value-memory-states-per-game", type=int,
                    default=32,
                    help="Reservoir cap per game (stride-thinned): "
                         "outcome INDEPENDENCE scales with games, "
                         "not transitions, so a long game must not "
                         "dominate its slot.")
    ap.add_argument("--draw-tiebreak-cap", type=float, default=0.3,
                    help="MCTS draws score by material differential "
                         "(villages + gold + unit value) in "
                         "(-cap, +cap) instead of a flat z=0, both "
                         "at search turn-cap terminals and in the "
                         "trainer's z target. <=0 disables. See "
                         "configs/draw_tiebreak.json.")
    ap.add_argument("--draw-tiebreak-config", type=Path, default=None,
                    help="JSON overriding the draw-tiebreak weights "
                         "(takes precedence over --draw-tiebreak-cap).")
    args = ap.parse_args(argv[1:])
    # One run = one provenance tag, shared with every spawned worker
    # via the environment (see validation_exports.run_tag).
    import time as _time
    os.environ.setdefault(
        "WESNOTH_RUN_TAG",
        _time.strftime("%Y%m%d-%H%M%S", _time.gmtime()))
    if getattr(args, "mini_random_tod", False):
        os.environ["WESNOTH_MINI_RANDOM_TOD"] = "1"
    # Mix guard (2026-07-20): the five category ratios are absolute
    # proportions and must account for the full distribution.
    from tools.scenario_pool import validate_mix
    try:
        validate_mix(midgame=args.midgame_ratio, mini=args.mini_ratio,
                     fogless=args.fogless_ratio,
                     ladder=args.ladder_ratio)
    except ValueError as e:
        ap.error(str(e))
    # Jitter floor sanity. 0/None = fixed cap (explicit opt-out).
    # A floor ABOVE the cap clamps to a fixed cap with a warning
    # rather than erroring: the floor now has a nonzero DEFAULT (60,
    # user ruling 2026-08-17), so a small --max-turns (smokes, minis)
    # must not require also remembering to lower the floor --
    # _roll_max_turns already treats min >= max as fixed-cap.
    if args.max_turns_min and args.max_turns_min > args.max_turns:
        log.warning(f"--max-turns-min {args.max_turns_min} > "
                    f"--max-turns {args.max_turns}; jitter disabled "
                    f"(fixed cap {args.max_turns}).")
    if args.max_turns_min is not None and args.max_turns_min < 0:
        ap.error(f"--max-turns-min {args.max_turns_min} negative")
    if args.reward_config is not None and args.mcts:
        # Shaping rewards are STRUCTURALLY INERT under --mcts:
        # MCTSPolicy.observe is a no-op, so every weight in the config
        # would be silently ignored -- the trap that hid the
        # weight_gold=0 non-fix for weeks (diagnosed 2026-07-31).
        # Refuse rather than warn: the combination has no valid
        # meaning, and a missed warning on an unattended box run costs
        # the whole leg (F5 user ruling 2026-08-10).
        ap.error(
            "--reward-config has no effect under --mcts (AlphaZero "
            "distills the terminal z; MCTSPolicy.observe is a no-op). "
            "The live shaping seams in MCTS mode are "
            "--draw-tiebreak-cap / --draw-tiebreak-config and the "
            "terminal outcome itself. Did you mean --reinforce?")
    if args.game_log_dir is not None and str(args.game_log_dir) in ("", "."):
        args.game_log_dir = None
    if args.game_record_dir is not None and str(args.game_record_dir) in ("", "."):
        args.game_record_dir = None
    if args.game_record_dir is not None:
        from tools.game_record import configure as _configure_records
        from tools.validation_exports import run_tag as _run_tag
        _configure_records(args.game_record_dir / _run_tag(), f"learner_p{os.getpid()}")
    if int(getattr(args, "validate_export_every", 0)) > 0:
        from tools.validation_exports import ValidationExporter
        selfplay_game.VALIDATION_EXPORTER = ValidationExporter(
            args.validate_export_dir,
            every=args.validate_export_every)
    if float(getattr(args, "midgame_ratio", 0.0)) > 0:
        from tools.midgame_starts import midgame_available
        if not midgame_available(args.midgame_dataset):
            # Hard error since the 2026-07-20 absolute-mix redesign:
            # silently degrading midgame games to fresh starts would
            # falsify the requested distribution (2026-07-15 lesson:
            # 56 min trained with midgame 0/0 on a warning nobody
            # saw). Stage the corpus or zero the ratio.
            ap.error(
                f"--midgame-ratio {args.midgame_ratio} requested but "
                f"no value corpus at {args.midgame_dataset}")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve the device FIRST so device-specific tuning below (thread
    # cap, train batch size) can key off it. (Was previously chosen
    # later; moved up for cluster-readiness, 2026-06-15.)
    import os as _os
    import torch
    if args.device in ("auto", "dml", "directml"):
        # Route through the helper so DML's `privateuseone:N` device
        # object (not a valid torch.device string) is resolved.
        from tools.device_select import select_inference_device
        device = select_inference_device(args.device)
    else:
        device = torch.device(args.device) if args.device else None
    _is_cpu = (device is None) or str(device).startswith("cpu")

    # Optimization #1 (CPU-ONLY): cap torch's intra-op thread pool
    # before the first forward. On CPU the tiny per-leaf states
    # oversubscribe the all-cores pool; capping to ~4 measured
    # ~1.3-2.3x. On a GPU device, leave the default so CPU-side
    # MCTS/encoding can use all cores while the GPU does forwards --
    # capping there would throttle the host pipeline. An explicit
    # --torch-threads forces a cap on ANY device. See --torch-threads.
    if args.torch_threads is not None:
        _n_threads = int(args.torch_threads)           # explicit, any device
    elif _is_cpu:
        _n_threads = int(_os.environ.get("WAI_TORCH_THREADS", "4"))
    else:
        _n_threads = 0                                 # GPU: torch default
    if _n_threads > 0:
        _was = torch.get_num_threads()
        torch.set_num_threads(_n_threads)
        log.info(f"torch intra-op threads {_was} -> {_n_threads} "
                 f"(CPU-only tuning; device={device})")

    rng = random.Random(args.seed)
    # Self-play seeds are scenarios + random factions/leaders
    # (tools.scenario_pool), not replay starting states. pool_files
    # stays as an explicit None for the legacy plumbing downstream
    # (the --replay-pool flag was removed 2026-08-10, X6: it was
    # parsed and then ignored).
    pool_files = None
    cost_lookup = _recruit_cost_lookup()
    # Eagerly load factions to surface any setup issue NOW rather
    # than on the first worker thread.
    from tools.scenario_pool import (load_factions, LADDER_SCENARIO_IDS,
                                     MINI_MAP_SCENARIO_IDS)
    factions = load_factions()
    active_pool = (MINI_MAP_SCENARIO_IDS if args.mini_maps
                   else LADDER_SCENARIO_IDS)
    pool_label = "mini-maps" if args.mini_maps else "ladder"
    log.info(f"scenario pool: {len(active_pool)} {pool_label} maps "
             f"x {len(factions)} factions = "
             f"{len(active_pool) * len(factions) ** 2} "
             f"setup combinations (faction matchups with replacement)")
    if args.mini_maps:
        log.info(f"  mini-maps active: {', '.join(active_pool)}")

    # (device already resolved above, before the thread-cap.)
    # When warm-starting from a checkpoint with non-default arch
    # (e.g. supervised_epoch3.pt at d_model=128 while
    # TransformerPolicy's default is d_model=512), build the policy
    # at the SAVED arch so load_checkpoint can resume weights
    # rather than discarding everything and starting from random
    # init. The cluster job's chain logic relies on this -- losing
    # the warm-start every iteration would burn cluster time.
    arch_kwargs: Dict[str, int] = {}
    ckpt_aux_score = False
    ckpt_moves_left = False
    ckpt_gbc = False
    ckpt_relevant_set = None
    if args.checkpoint_in and (
            args.checkpoint_in.exists()
            or args.checkpoint_in.with_suffix(
                args.checkpoint_in.suffix + ".bak").exists()):
        # Resolve to a LOADABLE checkpoint: prefer the primary, but if it's
        # unreadable (truncated by a kill mid-write on a preemptible node)
        # or ABSENT (a kill inside save_checkpoint's rename window leaves
        # only the .bak -- project round-1 C5: the old primary-exists gate
        # made that window a random-init restart), fall back to the rolling
        # `.bak` that save_checkpoint keeps. Doing
        # this here — before load_checkpoint below — means both the arch
        # peek and the weight load use the same good file, so a spot
        # preemption costs at most the last save interval, not the whole run.
        candidates = [
            args.checkpoint_in,
            args.checkpoint_in.with_suffix(args.checkpoint_in.suffix + ".bak"),
        ]
        raw = None
        for cand in candidates:
            if not cand.exists():
                continue
            try:
                raw = torch.load(cand, map_location="cpu",
                                 weights_only=False)
                if cand != args.checkpoint_in:
                    log.warning(
                        f"primary checkpoint {args.checkpoint_in} unreadable;"
                        f" resuming from backup {cand}")
                    args.checkpoint_in = cand
                break
            except Exception as e:
                log.warning(f"checkpoint {cand} unreadable: {e!r}")
        if raw is not None:
            saved_arch = raw.get("arch", {}) or {}
            for k in ("d_model", "num_layers", "num_heads", "d_ff"):
                if k in saved_arch:
                    arch_kwargs[k] = int(saved_arch[k])
            ckpt_aux_score = bool(raw.get("aux_score", False))
            ckpt_moves_left = bool(raw.get("moves_left", False))
            ckpt_gbc = bool(raw.get("gbc", False))
            ckpt_relevant_set = raw.get("relevant_set_hexes", None)
            if arch_kwargs:
                log.info(f"warm-start arch from checkpoint: {arch_kwargs}"
                         f"{' +aux_score' if ckpt_aux_score else ''}"
                         f"{' +moves_left' if ckpt_moves_left else ''}")
        else:
            log.warning(
                "no loadable checkpoint (primary or .bak); falling back to "
                "TransformerPolicy defaults / random init"
            )

    # --- model scaling (plan §3.2): explicit CLI arch flags WIN over
    # the checkpoint's saved arch. If they differ from a warm-start
    # checkpoint, build at the requested size and let load_checkpoint
    # below hit "arch mismatch" -> discard weights -> fresh init (the
    # plan's intended "scaling means fresh init"). Logged loudly so a
    # warm-start isn't silently dropped.
    cli_arch = {k: v for k, v in (
        ("d_model", args.d_model), ("num_layers", args.num_layers),
        ("num_heads", args.num_heads), ("d_ff", args.d_ff),
    ) if v is not None}
    if cli_arch:
        conflicts = {k: (arch_kwargs[k], cli_arch[k]) for k in cli_arch
                     if k in arch_kwargs and arch_kwargs[k] != cli_arch[k]}
        arch_kwargs.update(cli_arch)
        log.info(f"arch from CLI flags: {cli_arch}")
        if conflicts:
            log.warning(
                "CLI arch flags differ from --checkpoint-in arch "
                f"({conflicts} as checkpoint->cli); the warm-start "
                "will be DISCARDED and training starts fresh at the "
                "requested size (model scaling).")
    if "num_heads" in arch_kwargs and "d_model" in arch_kwargs:
        if arch_kwargs["d_model"] % arch_kwargs["num_heads"] != 0:
            raise SystemExit(
                f"--num-heads ({arch_kwargs['num_heads']}) must divide "
                f"--d-model ({arch_kwargs['d_model']}).")

    # Auxiliary margin head (plan §3.5): on if requested OR if warm-
    # starting an aux-on checkpoint (so the head isn't dropped on
    # resume). It's a model-arch change, so it must be set at
    # construction.
    aux_score_flag = bool(args.mcts_aux_score) or ckpt_aux_score
    moves_left_flag = bool(args.mcts_moves_left) or ckpt_moves_left
    # GBC heads: same peek-and-OR as aux (a gbc-on checkpoint keeps
    # its trained heads on resume even under --no-gbc).
    gbc_flag = bool(getattr(args, "gbc", False)) or ckpt_gbc
    # Write the RESOLVED flag back (project round-5: the ActorPool
    # call reads args, so without this a --no-gbc resume of a gbc
    # checkpoint built+logged the heads ON learner-side while every
    # producer attached gbc_labels=None -- zero GBC gradient for the
    # leg).
    args.gbc = gbc_flag
    # Basis resolution (project round-1 C2: the ONLY structural
    # flag not peeked from the checkpoint -- a resume that omitted
    # the flag silently rebased the action space while every weight
    # loaded cleanly). CLI None = inherit; an explicit CLI value
    # that CONTRADICTS the checkpoint halts, since flipping the
    # basis mid-lineage is a deliberate act.
    _cli_rsh = getattr(args, "relevant_set_hexes", None)
    if _cli_rsh is None:
        relevant_set_flag = bool(ckpt_relevant_set)
        if ckpt_relevant_set is not None:
            log.info(f"--relevant-set-hexes inherited from "
                     f"checkpoint: {relevant_set_flag}")
    else:
        relevant_set_flag = bool(_cli_rsh)
        if (ckpt_relevant_set is not None
                and bool(ckpt_relevant_set) != relevant_set_flag):
            raise SystemExit(
                f"--relevant-set-hexes={relevant_set_flag} "
                f"contradicts the checkpoint "
                f"({bool(ckpt_relevant_set)}): flipping the action-"
                f"space basis mid-lineage rebases every policy "
                f"target. Pass the matching value, or omit the "
                f"flag to inherit.")
    # Write the RESOLVED basis back so every later consumer (the
    # actor pool, telemetry) sees one truth.
    args.relevant_set_hexes = relevant_set_flag
    # Inference precision/compile resolution (user ruling
    # 2026-08-28: compile+bf16 default on CUDA; measured 2.0x
    # together, ~1x each alone).
    # TRAINING default: OFF (2026-08-29). The 2026-08-28 compile+
    # bf16 ruling stands for EVAL (bench-validated, match-proven);
    # on the TRAINING path an in-process compile deadlocked on a
    # CUDA box (spool e2e hang, 33 min at 3% CPU -- suspected
    # cudagraph capture under the learner's threads), and the
    # teacher arms must not carry an unvalidated numerics change.
    # Flip AFTER a dedicated pool-path validation smoke (BACKLOG).
    if args.infer_bf16 is None:
        args.infer_bf16 = False
    if args.infer_compile is None:
        args.infer_compile = False
    log.info(f"inference config: bf16={args.infer_bf16} "
             f"compile={args.infer_compile} "
             f"(training default OFF pending pool-path validation; "
             f"eval keeps the 2026-08-28 cuda-auto default)")
    # The run's seed governs torch too: a fresh network's init read
    # torch's global generator, whose state is whatever the process
    # did before (the tripwire test's outcome depended on it,
    # 2026-09-18).
    import torch as _torch
    _torch.manual_seed(args.seed)
    random.seed(args.seed)
    policy = TransformerPolicy(device=device, aux_score=aux_score_flag,
                               moves_left=moves_left_flag,
                               gbc=gbc_flag,
                               relevant_set_hexes=relevant_set_flag,
                               infer_bf16=args.infer_bf16,
                               infer_compile=args.infer_compile,
                               **arch_kwargs)
    # The trainer's subsampling caps draw from its own generator; the
    # run's seed makes that sequence part of the run.
    policy._trainer.rng.seed(args.seed)
    if relevant_set_flag:
        log.info("relevant-hex encoding ON (action-space index basis "
                 "differs from full-board runs; see docs/archive/autonomous_run.md)")
    if aux_score_flag:
        policy._trainer.config.aux_coef = float(args.mcts_aux_coef)
        log.info(f"auxiliary margin head ON (aux_coef="
                 f"{args.mcts_aux_coef}; KataGo §3.5)")
    if moves_left_flag:
        policy._trainer.config.moves_left_coef = float(
            args.mcts_moves_left_coef)
        log.info(f"moves-left head ON (moves_left_coef="
                 f"{args.mcts_moves_left_coef}; Lc0-style tempo signal)")
    # train_batch_size: forward_batch chunk size. THE key GPU knob --
    # TransformerPolicy defaults to 1 (CPU) / 2 (DML) and leaves CUDA
    # at 1, so a CUDA run would forward the replay minibatch as
    # batch-1 calls. Override to a batched size on CUDA. See
    # --train-batch-size. (Read per-chunk as max(1, config.
    # train_batch_size), so a post-construction set takes effect.)
    if args.train_batch_size is not None:
        _tbs = max(1, int(args.train_batch_size))
    elif (device is not None) and str(device).startswith("cuda"):
        _tbs = 128
    else:
        _tbs = policy._trainer.config.train_batch_size  # keep 1/2 default
    policy._trainer.config.train_batch_size = _tbs
    log.info(f"train_batch_size = {_tbs} (forward_batch chunk; "
             f"device={device})")
    if args.value_coef is not None:
        # Override the value-loss weight (TrainerConfig default 0.5).
        # The trainer reads self.config.value_coef each step, so a
        # post-construction set takes effect; used by the replay
        # sweep to test whether weighting value learning harder helps
        # (the value head is the diagnosed bottleneck).
        policy._trainer.config.value_coef = float(args.value_coef)
        log.info(f"value_coef override -> {args.value_coef}")
    # GBC loss weight: follows the value_coef override pattern. Zeroed
    # when the model has no gbc heads so a stray coef can't no-op
    # silently (the gate also checks has_gbc; this keeps logs honest).
    policy._trainer.config.gbc_coef = (
        float(args.gbc_coef) if gbc_flag else 0.0)
    if gbc_flag:
        log.info(f"GBC event supervision ON (gbc_coef="
                 f"{args.gbc_coef}; heads checkpoint-sticky; labels "
                 f"attached in finalize_game)")
    if args.value_label_smoothing:
        policy._trainer.config.value_label_smoothing = float(
            args.value_label_smoothing)
        log.info(f"value label smoothing -> "
                 f"{args.value_label_smoothing} (train loss only)")
    policy._trainer.config.draw_value_weight = float(
        args.draw_value_weight)
    if args.draw_value_weight != 0.0:
        log.info(f"draw value weight -> {args.draw_value_weight} "
                 f"(winnerless states now FEED the value head at "
                 f"this weight; 0 = the truncation-ruling default)")
    if args.checkpoint_in and args.checkpoint_in.exists():
        log.info(f"loading checkpoint {args.checkpoint_in}")
        try:
            policy.load_checkpoint(args.checkpoint_in)
        except RuntimeError as e:
            # Arch mismatch (e.g., default size bumped from 0.5M to
            # 26M without re-warmstarting). load_checkpoint raises a
            # RuntimeError with "arch mismatch" prefix; treat that as
            # "start fresh" rather than aborting the job — chain
            # links survive across model-size changes that way.
            if "arch mismatch" in str(e).lower():
                log.warning(
                    f"checkpoint {args.checkpoint_in} has incompatible "
                    f"arch ({e}); discarding and training from random "
                    f"init."
                )
            else:
                raise
    else:
        log.warning("no input checkpoint -- training from random init")

    # Weights-only warm-start: forget the loaded training-progress
    # counter so the combat-oracle anneal restarts from full strength.
    if args.reset_decision_step:
        prev = policy._decision_step
        policy._decision_step = 0
        log.info(f"--reset-decision-step: decision_step {prev} -> 0 "
                 f"(combat-oracle anneal restarts at full strength)")

    # Vocab stays dynamically growable (config-driven customization: a
    # new id appends a fresh learnable embedding row; existing ids never
    # shift -- see encoder.register_names). Once the base roster is in
    # place (a warm-started checkpoint already carries it), arm growth
    # logging so any NEW unit type appearing during self-play is a
    # visible breadcrumb rather than a silent change on a long run. Skip
    # for a fresh init, whose first iteration legitimately populates the
    # whole roster.
    if len(policy._encoder.unit_type_to_id) > 0:
        policy.watch_vocab_growth()

    # Optional MCTS wrapper: replaces raw policy sampling with an
    # AlphaZero-style tree search. Same duck-typed
    # `select_action` / `observe` / `finalize_game` / `train_step`
    # interface as the underlying TransformerPolicy, so the rollout
    # loop doesn't branch.
    if args.mcts:
        from tools.draw_tiebreak import DrawTiebreakConfig
        from tools.mcts import MCTSConfig
        from tools.mcts_policy import MCTSPolicy, ReplayConfig
        from tools.turn_search import (
            config_from_args as turn_config_from_args,
        )
        if args.draw_tiebreak_config is not None:
            tiebreak = DrawTiebreakConfig.from_json(
                args.draw_tiebreak_config)
        elif args.draw_tiebreak_cap > 0:
            tiebreak = DrawTiebreakConfig(cap=args.draw_tiebreak_cap)
        else:
            tiebreak = None
        # mcts_batch_size: device-aware default (mirrors train_batch_size).
        # The per-leaf model forward is the dominant cost of an --mcts run;
        # leaving B=1 on CUDA issues one un-batched forward per simulation
        # and starves the GPU. Auto-bump to 16 on CUDA when the flag is
        # unset; an explicit --mcts-batch-size always wins.
        if args.mcts_batch_size is not None:
            _mbs = max(1, int(args.mcts_batch_size))
        elif (device is not None) and str(device).startswith("cuda"):
            _mbs = 16
        else:
            _mbs = 1
        if ((device is not None) and str(device).startswith("cuda")
                and _mbs == 1):
            log.warning(
                "MCTS leaf batching is B=1 on CUDA: each simulation runs an "
                "un-batched leaf forward, which starves the GPU. Set "
                "--mcts-batch-size 8-32 (profile to pick B).")
        log.info(f"mcts_batch_size = {_mbs} (leaf forward_batch; "
                 f"device={device})")
        mcts_cfg = MCTSConfig(
            n_simulations=args.mcts_sims,
            moves_left_utility=args.mcts_moves_left_utility,
            aux_value_bonus=args.mcts_aux_value_bonus,
            c_puct=args.mcts_c_puct,
            batch_size=_mbs,
            fpu_reduction=(None if args.mcts_fpu_reduction < 0
                           else args.mcts_fpu_reduction),
            temperature=args.mcts_temperature,
            temperature_decisions=args.mcts_temperature_decisions,
            draw_tiebreak=tiebreak,
            tree_reuse=not args.mcts_no_tree_reuse,
            gumbel_root=not args.mcts_classic_root,
            gumbel_m=args.mcts_gumbel_m,
            gumbel_rescale_floor=args.mcts_gumbel_rescale_floor,
            exact_outcome_enumeration=not args.mcts_no_exact_outcomes,
            playout_cap_randomization=args.mcts_playout_cap,
            playout_cap_prob=args.mcts_playout_cap_prob,
            playout_cap_fast_sims=args.mcts_playout_cap_fast_sims,
            distill_prior_discount=args.distill_prior_discount,
            distill_target_temp=args.distill_target_temp,
            gumbel_hierarchical=getattr(
                args, "mcts_hierarchical_gumbel", False),
        )
        root_desc = (f"gumbel(m={mcts_cfg.gumbel_m})"
                     if mcts_cfg.gumbel_root else
                     f"classic(tau={mcts_cfg.temperature}"
                     f"x{mcts_cfg.temperature_decisions})")
        log.info(
            f"MCTS mode enabled: sims={mcts_cfg.n_simulations} "
            f"c_puct={mcts_cfg.c_puct} batch_size={mcts_cfg.batch_size} "
            f"fpu={mcts_cfg.fpu_reduction} root={root_desc} "
            f"tree_reuse={mcts_cfg.tree_reuse} "
            f"playout_cap="
            f"{'on(p=%.2f,fast=%d)' % (mcts_cfg.playout_cap_prob, mcts_cfg.playout_cap_fast_sims or max(1, mcts_cfg.n_simulations // 4)) if mcts_cfg.playout_cap_randomization else 'off'} "
            f"draw_tiebreak_cap="
            f"{tiebreak.cap if tiebreak else 'off'}."
        )
        replay_cfg = ReplayConfig(
            enabled=args.replay_buffer,
            capacity=args.replay_capacity,
            updates_per_iter=args.replay_updates,
            minibatch=args.replay_minibatch,
            min_size=args.replay_min_size,
        )
        if replay_cfg.enabled:
            log.info(
                f"replay buffer ON: capacity={replay_cfg.capacity} "
                f"updates/iter={replay_cfg.updates_per_iter} "
                f"minibatch={replay_cfg.minibatch} "
                f"warmup>={replay_cfg.min_size} (multi-epoch training; "
                f"value head gets {replay_cfg.updates_per_iter}x the "
                f"gradient steps per iter vs legacy one-pass)")
        if (args.mcts_aux_value_bonus
                and not getattr(policy._model, "has_aux_score", False)):
            log.warning(
                "--mcts-aux-value-bonus is set but the model has NO "
                "aux head (--mcts-aux-score): the bonus is a silent "
                "no-op (reviewer finding m3, 2026-07-11).")
        from tools.plan_tournament import (
            config_from_args as pt_config_from_args,
        )
        pt_cfg = pt_config_from_args(args)
        turn_cfg = turn_config_from_args(args)
        # Report every kernel through its own phase gate: a wheel that
        # imports may be too old for some of them (tools/kernel_status.py).
        from tools.kernel_status import banner as _kernel_banner
        log.info("%s (WESNOTH_RUST=%s)", _kernel_banner(),
                 os.environ.get("WESNOTH_RUST", "1"))
        from tools.value_grounding import (
            config_from_args as ground_config_from_args,
        )
        ground_cfg = ground_config_from_args(args, turn_cfg)
        if ground_cfg is not None:
            log.info(
                f"VALUE GROUNDING on: capture_prob="
                f"{ground_cfg.capture_prob} rollouts/game="
                f"{ground_cfg.max_rollout_per_game} consist/game="
                f"{ground_cfg.max_consist_per_game} weights="
                f"{ground_cfg.ground_value_weight}/"
                f"{ground_cfg.consist_value_weight}")
        if pt_cfg is not None:
            from tools.plan_tournament import PlanTournamentPolicy
            policy = PlanTournamentPolicy(
                policy, mcts_cfg, replay_config=replay_cfg,
                holdout_size=args.holdout_size,
                holdout_per_game_cap=args.holdout_per_game_cap,
                train_draw_tiebreak=args.train_draw_tiebreak,
                draw_value_weight=args.draw_value_weight,
                value_memory_games=(args.value_memory_iters
                                    * args.games_per_iter),
                value_memory_states_per_game=(
                    args.value_memory_states_per_game),
                value_memory_batch=args.value_memory_batch,
                gbc_labels=gbc_flag,
                rng_seed=args.seed,
                tournament_config=pt_cfg)
            from tools.plan_tournament import launch_echo_schedule
            _n12, _d12, _dem12, _ph12 = launch_echo_schedule(pt_cfg)
            log.info(
                f"PLAN TOURNAMENT on (proposition 1, 2026-08-26): "
                f"challengers={pt_cfg.n_challengers} "
                f"depths={pt_cfg.depths} redraws={pt_cfg.redraws} "
                f"cert={pt_cfg.cert_depth}x{pt_cfg.cert_redraws} "
                f"budget={pt_cfg.budget_forwards}fwd "
                f"band={pt_cfg.margin_band} "
                f"beta_max={pt_cfg.beta_max}; certify-or-abstain, "
                f"beta tripwire DISARMED (calibration iteration); "
                f"at K=12 cold-start (per_half={_ph12}) the budget "
                f"funds challengers={_n12} depths={_d12} "
                f"(demand {_dem12}; schedule auto-sizes per turn)")
            if _n12 == 0:
                log.error(
                    "PLAN TOURNAMENT: the budget cannot fund even "
                    "the floor schedule at K=12 -- every mid-length "
                    "side-turn will abstain. Raise "
                    "--pt-budget-forwards.")
        elif turn_cfg is not None:
            from tools.turn_policy import TurnCommitPolicy
            policy = TurnCommitPolicy(
                policy, mcts_cfg, replay_config=replay_cfg,
                holdout_size=args.holdout_size,
                holdout_per_game_cap=args.holdout_per_game_cap,
                train_draw_tiebreak=args.train_draw_tiebreak,
                draw_value_weight=args.draw_value_weight,
                value_memory_games=(args.value_memory_iters
                                    * args.games_per_iter),
                value_memory_states_per_game=(
                    args.value_memory_states_per_game),
                value_memory_batch=args.value_memory_batch,
                gbc_labels=gbc_flag,
                rng_seed=args.seed,
                turn_config=turn_cfg,
                grounding_config=ground_cfg,
                signal_telemetry=args.signal_telemetry)
            log.info(
                f"TURN-COMMITMENT SEARCH on (docs/archive/tcs_spec.md): "
                f"alt={turn_cfg.n_alt} rounds={turn_cfg.rounds}/"
                f"{turn_cfg.fast_rounds} reval={turn_cfg.reval_salts} "
                f"full_prob={turn_cfg.turn_full_prob} "
                f"project={turn_cfg.project}"
                f":{turn_cfg.project_halfturns}"
                f"x{turn_cfg.project_max_actions} "
                f"link={turn_cfg.target_link}"
                f"(beta={turn_cfg.target_beta}); "
                f"--no-turn-search restores per-decision Gumbel MCTS")
        else:
            policy = MCTSPolicy(
                policy, mcts_cfg, replay_config=replay_cfg,
                holdout_size=args.holdout_size,
                holdout_per_game_cap=args.holdout_per_game_cap,
                train_draw_tiebreak=args.train_draw_tiebreak,
                draw_value_weight=args.draw_value_weight,
                value_memory_games=(args.value_memory_iters
                                    * args.games_per_iter),
                value_memory_states_per_game=(
                    args.value_memory_states_per_game),
                value_memory_batch=args.value_memory_batch,
                gbc_labels=gbc_flag,
                rng_seed=args.seed,
                signal_telemetry=args.signal_telemetry)
        if args.train_draw_tiebreak:
            log.info("LEGACY draw labels: training z = material "
                     "tiebreak on draws")
        else:
            log.info("honest draw labels: training z = 0 on draws "
                     "(tiebreak remains search-only)")
        if args.holdout_size > 0:
            # Works on both the in-process path (finalize_game) and
            # --actor-pool (the pool drain offers each per-game
            # _R_EXPS payload to offer_holdout_game).
            log.info(
                f"holdout probe ON: first ~{args.holdout_size} "
                f"experiences (whole games) are held out of "
                f"training and scored each iter.")
            # Persist the probe across supervisor relaunches so
            # holdout CE stays ONE comparable curve (2026-07-18:
            # per-restart resampling made capacity trends unreadable
            # -- levels jumped 0.44<->0.88 on set changes). The file
            # rides beside the campaign checkpoint; a fresh campaign
            # (new checkpoint path / cleared checkpoint) should clear
            # it too (onstart's HF-seed block does).
            holdout_file = Path(str(args.checkpoint_out) + ".holdout")
            if policy.load_holdout(holdout_file):
                pass  # restored (full -> frozen; partial -> resumes)
            setattr(policy, "_holdout_persist_path", holdout_file)

    if args.reward_config is not None:
        # (mcts+reward_config was refused at arg validation.)
        try:
            reward_fn = load_reward_config(args.reward_config)
        except (KeyError, ValueError, ImportError) as e:
            log.error(f"failed to load reward config: {e}")
            return 2
        log.info(
            f"reward config: {args.reward_config} "
            f"({len(reward_fn.unit_type_bonuses)} unit-type bonuses, "
            f"{len(reward_fn.turn_conditional_bonuses)} "
            f"turn-conditional bonuses)")
    else:
        reward_fn = WeightedReward()

    pvp_defaults: Optional[PvPDefaults] = None
    if args.use_map_settings:
        pvp_defaults = PvPDefaults(
            starting_gold=args.starting_gold,
            village_gold=args.village_gold,
            village_support=args.village_support,
            experience_modifier=args.exp_modifier,
        )
        log.info(f"using PvP defaults: {pvp_defaults}")
    else:
        log.info("--no-map-settings: keeping source-replay settings")

    # Translate the --forced-faction CLI string into the right value
    # for random_setup: ... = "use module default", None = disabled,
    # str = lock to that faction.
    forced_faction_arg: object = ...
    if args.forced_faction is not None:
        if args.forced_faction.lower() == "none":
            forced_faction_arg = None
        else:
            forced_faction_arg = args.forced_faction

    # Parse --time-budget once. Wall-time exit lets cluster jobs run
    # for "as much as SLURM will give me, minus a couple minutes of
    # headroom to save the checkpoint" rather than guessing how many
    # iterations fit in a walltime window.
    time_budget_s = _parse_time_budget(args.time_budget)
    t_start = time.perf_counter()
    if time_budget_s is not None:
        log.info(
            f"time-budget: {time_budget_s} s "
            f"({time_budget_s // 3600:02d}:"
            f"{(time_budget_s % 3600) // 60:02d}:"
            f"{time_budget_s % 60:02d} HH:MM:SS); will save + exit "
            f"after the first iteration that finishes past this "
            f"mark. --iterations is a ceiling only."
        )

    # Trainer history CSV. Empty string disables; None falls back
    # to the default per-job path. The writer is line-buffered so a
    # walltime-killed job leaves a recoverable file.
    # Human-corpus rehearsal anchor (2026-07-10): pre-encoded
    # (RawEncoded, z, moves_left) tuples; value-only steps each iter.
    human_anchor = None
    if getattr(args, "human_anchor_file", None):
        import pickle as _pkl
        from tools.build_human_anchor import check_anchor_gate
        check_anchor_gate(args.human_anchor_file,
                          bool(getattr(policy, "_fog_hides_enemy_villages", False)),
                          bool(getattr(policy, "_terrain_multi_hot", False)))
        with args.human_anchor_file.open("rb") as _f:
            _pool = _pkl.load(_f)
        if _pool:
            human_anchor = (_pool, max(0, args.human_anchor_updates),
                            max(1, args.human_anchor_batch),
                            random.Random(args.seed ^ 0xA17C4))
            log.info(
                f"human anchor ON: {len(_pool)} pre-encoded states "
                f"from {args.human_anchor_file.name}; "
                f"{args.human_anchor_updates} value-only updates x "
                f"{args.human_anchor_batch} per iteration")
        else:
            log.warning("human anchor file empty; rehearsal OFF")
    human_anchor_policy = None
    if getattr(args, "human_anchor_policy_file", None):
        from tools.policy_anchor import load_policy_anchor
        _pgames = load_policy_anchor(
            args.human_anchor_policy_file,
            fog_hides_enemy_villages=bool(getattr(policy, "_fog_hides_enemy_villages", False)),
            terrain_multi_hot=bool(getattr(policy, "_terrain_multi_hot", False)))
        if _pgames:
            _npairs = sum(len(g) for g in _pgames)
            human_anchor_policy = (
                _pgames, max(0, args.human_anchor_policy_updates),
                max(1, args.human_anchor_policy_batch),
                random.Random(20260810))
            log.info(
                f"POLICY-head human anchor ON: {_npairs} pairs across "
                f"{len(_pgames)} games (game-normalized draw, cache "
                f"v2) from {args.human_anchor_policy_file.name}; "
                f"{args.human_anchor_policy_updates} imitation-CE "
                f"updates x {args.human_anchor_policy_batch} per "
                f"iteration. Reminder: ONE prior protection per leg "
                f"(F1 ruling) -- don't combine with "
                f"--distill-prior-discount < 1 in the same leg.")
        else:
            log.warning("policy anchor file empty; rehearsal OFF")

    history_csv: Optional[_TrainerHistoryCSV] = None
    csv_arg = args.trainer_history_csv
    if csv_arg is None:
        csv_path = _default_history_csv()
    elif str(csv_arg) == "":
        csv_path = None
    else:
        csv_path = csv_arg
    if csv_path is not None:
        try:
            history_csv = _TrainerHistoryCSV(csv_path)
            log.info(f"trainer history CSV: {csv_path}")
        except OSError as e:
            log.warning(f"couldn't open trainer history CSV "
                        f"{csv_path}: {e}; continuing without")

    # Trip if N iterations in a row produce zero games. The most
    # common cause is missing data files (terrain_db.json,
    # unit_stats.json) — every scenario gets skipped silently and
    # the trainer happily burns walltime on empty iters with
    # `loss=0.0000`. Observed 2026-05-09 in selfplay job 17834
    # which "completed" 999 iters of 0 games. Fail fast instead.
    # Graceful-cancel sentinel: fixed canonical path so writer (GUI)
    # and reader (here) agree regardless of where args.checkpoint_out
    # points. The GUI's "Stop training (save)" button writes a file
    # at this path; the loop below polls for it after each iter.
    # Living at training/checkpoints/ because that's where every
    # other training-state file lives.
    _CANCEL_SENTINEL = (Path("training") / "checkpoints"
                        / ".cancel_local")
    # Clear any leftover sentinel from a previous run that didn't
    # clean up (e.g. force-killed after detecting but before
    # removing). Without this, a fresh training would detect the
    # stale sentinel at iter 0 and exit immediately, leaving the
    # operator confused about why their run "finished" with no
    # progress.
    if _CANCEL_SENTINEL.exists():
        try:
            _CANCEL_SENTINEL.unlink()
            log.info(f"cleared stale graceful-cancel sentinel "
                     f"({_CANCEL_SENTINEL})")
        except OSError:
            pass

    # Optional multiprocess actor pool (MCTS only). Built here so all
    # scenario/device/pvp settings are resolved; started before the
    # loop and torn down via atexit (daemon actors also die with the
    # parent). Mutually exclusive with --workers.
    actor_pool = None
    if args.actor_pool > 0:
        if not args.mcts:
            log.error("--actor-pool requires --mcts")
            return 2
        if args.workers > 0:
            log.error("--actor-pool is mutually exclusive with --workers")
            return 2
        import atexit
        from tools.actor_pool import ActorPool
        scenario_opts = dict(
            forced_faction=forced_faction_arg,
            mini_maps=args.mini_maps,
            mini_ratio=float(args.mini_ratio),
            fogless_ratio=float(args.fogless_ratio),
            ladder_ratio=float(args.ladder_ratio),
            # Actors splice midgame starts (2026-07-22 port).
            midgame_ratio=float(args.midgame_ratio),
            midgame_dataset=args.midgame_dataset,
        )
        from tools.plan_tournament import (
            config_from_args as _pt_config_from_args,
        )
        from tools.value_grounding import (
            config_from_args as _ground_config_from_args,
        )
        _pool_turn_cfg = (None if _pt_config_from_args(args) is not None
                          else turn_config_from_args(args))
        actor_pool = ActorPool(
            policy, args.actor_pool, mcts_cfg,
            turn_cfg=_pool_turn_cfg,
            ground_cfg=_ground_config_from_args(args, _pool_turn_cfg),
            pt_cfg=_pt_config_from_args(args),
            gbc_labels=gbc_flag,
            train_kwargs={
                "draw_value_weight": float(args.draw_value_weight),
                "train_draw_tiebreak": bool(
                    args.train_draw_tiebreak),
            },
            scenario_opts=scenario_opts, max_turns=args.max_turns,
            max_turns_min=args.max_turns_min,
            pvp_defaults=pvp_defaults, device=device,
            max_batch=(args.actor_max_batch or None),
            drain_grace=float(getattr(args, "pool_drain_grace",
                                      1800.0)),
            log_level=logging.getLogger().level,
            game_records_dir=(None if args.game_record_dir is None
                              else args.game_record_dir / _records_run_tag()))
        actor_pool.start()
        atexit.register(actor_pool.shutdown)
        log.info(f"actor pool: {args.actor_pool} processes "
                 f"(GIL-free; --workers thread-path disabled)")

    # K-collapse tripwire state (--abort-k-median).
    k_low_streak = 0
    if args.abort_k_median is not None:
        log.info(f"K-collapse tripwire ON: stop if median actions/"
                 f"side-turn < {args.abort_k_median} for 3 "
                 f"consecutive iterations.")
    # Decisive-rate abort tripwire state (--abort-decisive-rate).
    # Trailing window of (decisive_games, total_games) per iteration;
    # armed once the window is full (= burn-in of --abort-window iters).
    from collections import deque as _deque
    abort_rate = args.abort_decisive_rate
    abort_hist: "_deque[Tuple[int, int]]" = _deque(
        maxlen=max(1, args.abort_window))
    if abort_rate is not None:
        log.info(
            f"abort tripwire ON: stop if decisive-game rate over the "
            f"trailing {args.abort_window} iters drops below "
            f"{abort_rate:.0%} (armed after iter {args.abort_window}).")
    if args.holdout_size > 0 and not args.mcts:
        log.warning("--holdout-size requires --mcts; probe inactive.")
    # Holdout-stall (memorization) tripwire state. Tracks the running
    # best holdout CE; an iteration only resets the stall counter when
    # it beats the best by at least min_delta.
    holdout_stall_limit = args.abort_holdout_stall
    if holdout_stall_limit is not None and not args.replay_buffer:
        log.warning("--abort-holdout-stall needs --replay-buffer "
                    "(fresh-CE telemetry); tripwire inactive.")
        holdout_stall_limit = None
    holdout_best: Optional[float] = None
    holdout_stall = 0
    if holdout_stall_limit is not None:
        log.info(
            f"holdout-stall tripwire ON: stop if FLOOR-RELATIVE fresh "
            f"holdout CE (fresh_value_ce - fresh_ce_floor) makes no "
            f"new best (min delta {args.abort_holdout_min_delta}) for "
            f"{holdout_stall_limit} consecutive iters.")

    DEAD_ITER_LIMIT = 5
    consecutive_dead = 0
    for it in range(args.iterations):
        outcomes = run_iteration(
            policy, pool_files, reward_fn, cost_lookup,
            iter_idx=it,
            games_per_iter=args.games_per_iter,
            max_turns=args.max_turns,
            rng=rng,
            pvp_defaults=pvp_defaults,
            workers=args.workers,
            forced_faction=forced_faction_arg,
            mini_maps=args.mini_maps,
            mini_ratio=float(args.mini_ratio),
            fogless_ratio=float(args.fogless_ratio),
            midgame_ratio=float(args.midgame_ratio),
            ladder_ratio=float(args.ladder_ratio),
            max_turns_min=args.max_turns_min,
            no_progress_turns=int(args.no_progress_turns),
            midgame_dataset=args.midgame_dataset,
            game_log_dir=args.game_log_dir,
            snapshot_sink=(history_csv.append if history_csv else None),
            actor_pool=actor_pool,
            human_anchor=human_anchor,
            human_anchor_policy=human_anchor_policy,
        )
        # Persist the holdout probe when it grew (crash-safe partial
        # saves; no-op once frozen+saved). getattr-guarded: REINFORCE
        # lacks the method -- the probe simply isn't persisted there.
        getattr(policy, "maybe_persist_holdout", lambda: None)()
        if selfplay_game.VALIDATION_EXPORTER is not None:
            _bundle_validation_exports(
                getattr(args, "validate_export_dir", None), it)
        if not outcomes:
            consecutive_dead += 1
            if consecutive_dead >= DEAD_ITER_LIMIT:
                log.error(
                    f"{DEAD_ITER_LIMIT} consecutive iterations rolled "
                    f"zero games. Likely cause: missing scrape file "
                    f"(terrain_db.json / unit_stats.json) or scenario "
                    f"setup error. Aborting; check earlier WARNING "
                    f"logs for the underlying skip reason."
                )
                return 3
        else:
            consecutive_dead = 0
        # Decisive-rate tripwire: on a paid GPU node, a policy that
        # draws essentially every game generates almost no win/loss
        # signal -- the known failure shape (the 2026-05 iter-168
        # baseline had ZERO leaderkills on full maps). Predefined
        # abort > deciding at hour two with money burning. State is
        # saved before exiting (checkpoint + line-buffered CSV), so
        # nothing is lost: diagnose, adjust, resume from the same
        # checkpoint (WITHOUT --reset-decision-step).
        if abort_rate is not None and outcomes:
            abort_hist.append(
                (sum(1 for o in outcomes if o.winner != 0),
                 len(outcomes)))
            if len(abort_hist) == abort_hist.maxlen:
                dec = sum(d for d, _ in abort_hist)
                tot = sum(t for _, t in abort_hist)
                rate = (dec / tot) if tot else 0.0
                if rate < abort_rate:
                    policy.save_checkpoint(args.checkpoint_out)
                    log.error(
                        f"ABORT TRIPWIRE: decisive-game rate "
                        f"{rate:.1%} ({dec}/{tot} over the last "
                        f"{len(abort_hist)} iters) is below the "
                        f"--abort-decisive-rate {abort_rate:.0%} "
                        f"threshold after iter {it + 1}. Final "
                        f"checkpoint saved to {args.checkpoint_out}; "
                        f"trainer-history CSV is flushed per row. "
                        f"Diagnose before re-launching: check "
                        f"closest_approach / attack%% trends in the "
                        f"CSV, consider --mini-ratio (engagement "
                        f"curriculum), a higher --draw-tiebreak-cap, "
                        f"or a longer --max-turns. Exit code 4."
                    )
                    if history_csv is not None:
                        history_csv.close()
                    return 4
        # K-collapse tripwire: leg 3's turn-length collapse (K median
        # 14 -> 2) ran for days with every other guard green -- the
        # search-driven passivity failure shape is invisible to the
        # decisive-rate, CE, and AUC tripwires by construction.
        if args.abort_k_median is not None and outcomes:
            k_med = k_median_of(outcomes)
            if k_med is not None and k_med < args.abort_k_median:
                k_low_streak += 1
            else:
                k_low_streak = 0
            if k_low_streak >= 3:
                policy.save_checkpoint(args.checkpoint_out)
                log.error(
                    f"ABORT TRIPWIRE: median actions/side-turn "
                    f"{k_med} < --abort-k-median "
                    f"{args.abort_k_median} for 3 consecutive "
                    f"iterations (iter {it + 1}). Turn-length "
                    f"collapse = the leg-3 passivity shape. Final "
                    f"checkpoint saved to {args.checkpoint_out}. "
                    f"Exit code 7."
                )
                if history_csv is not None:
                    history_csv.close()
                return 7
        # Holdout-stall (memorization) tripwire: the fresh holdout CE
        # is the only metric that distinguishes value learning from
        # replay-buffer fitting (measured 2026-07-02: train value loss
        # fell 3.8->1.15 while holdout CE sat flat at ~3.1). Tracked
        # FLOOR-RELATIVE (fresh_value_ce - fresh_ce_floor) because raw
        # CE moves with the outcome-label mix -- the raw version
        # mis-fired twice in the 72h run (A5 spec, user ruling
        # 2026-08-10). If it makes no new best for the configured
        # stretch, training is either memorizing or stalled -- either
        # way, on paid compute, stop and diagnose. State is saved
        # (checkpoint + flushed CSV). Iterations with no reading
        # (iter-0-after-restart, replay-off) skip the counter.
        if holdout_stall_limit is not None:
            hl = getattr(policy, "last_fresh_rel_ce", None)
            if hl is not None:
                if (holdout_best is None
                        or hl < holdout_best - args.abort_holdout_min_delta):
                    holdout_best = hl
                    holdout_stall = 0
                else:
                    holdout_stall += 1
                    if holdout_stall >= holdout_stall_limit:
                        policy.save_checkpoint(args.checkpoint_out)
                        log.error(
                            f"ABORT TRIPWIRE (holdout stall): floor-"
                            f"relative fresh holdout CE has not beaten "
                            f"its best ({holdout_best:.4f}) by "
                            f"{args.abort_holdout_min_delta} for "
                            f"{holdout_stall} consecutive iters "
                            f"(latest {hl:.4f}) after iter {it + 1}, "
                            f"while training continued -- the "
                            f"memorization signature. Final "
                            f"checkpoint saved to "
                            f"{args.checkpoint_out}; CSV flushed. "
                            f"Compare train_value_loss vs "
                            f"fresh_value_ce/fresh_ce_floor columns "
                            f"before re-launching (capacity? signal? "
                            f"label-mix shift?). Exit code 5."
                        )
                        if history_csv is not None:
                            history_csv.close()
                        return 5
        # Time-budget early exit. Check AFTER the iteration finishes
        # rather than mid-iteration: a partial iteration's gradient
        # update wouldn't have happened yet, so cutting mid-iter
        # would waste the rollout work AND the loop's invariant
        # (every emitted "iter K done" message reflects a completed
        # train_step) would break. Save a checkpoint at the natural
        # save-every cadence below, then break.
        elapsed = time.perf_counter() - t_start
        time_budget_exceeded = (
            time_budget_s is not None and elapsed >= time_budget_s
        )
        # Graceful-cancel sentinel: the GUI's "Stop training (save)"
        # button writes this file. We detect it between iters,
        # save, delete the sentinel (so a re-launch doesn't
        # immediately exit), and break out of the loop. Worst-case
        # latency = one iteration of rollout. Symmetric with the
        # time-budget exit -- both want "save what you've got and
        # stop cleanly" semantics.
        cancel_requested = _CANCEL_SENTINEL.exists()
        if (it + 1) % args.save_every == 0 \
                or (it + 1) == args.iterations \
                or time_budget_exceeded \
                or cancel_requested:
            policy.save_checkpoint(args.checkpoint_out)
        if cancel_requested:
            log.info(
                f"graceful-cancel sentinel detected at "
                f"{_CANCEL_SENTINEL}; checkpoint saved at iter "
                f"{it + 1}, exiting cleanly.")
            try:
                _CANCEL_SENTINEL.unlink()
            except OSError as e:
                log.warning(f"couldn't remove sentinel "
                            f"{_CANCEL_SENTINEL}: {e}; next "
                            f"training run will exit immediately "
                            f"unless you delete it manually.")
            break
        if time_budget_exceeded:
            log.info(
                f"time-budget exhausted ({elapsed:.0f}s >= "
                f"{time_budget_s}s) after iter {it + 1}; "
                f"checkpoint saved, exiting cleanly so the next "
                f"chain link can pick up where this one left off."
            )
            break
    if history_csv is not None:
        history_csv.close()
    return 0


if __name__ == "__main__":
    # Canonicalize the module identity: an import of
    # "tools.sim_self_play" while this file runs as __main__ (an
    # unpickled object naming the module, a late import) resolves to
    # THIS module object. Without the alias it would RE-EXECUTE the
    # whole module and hold two copies of every module global and
    # class (dual-import hazard, audited 2026-07-30). The game outcomes
    # the pool's actors send are tools.selfplay_game objects, a module
    # the learner imports under that one name.
    sys.modules.setdefault("tools.sim_self_play", sys.modules[__name__])
    if sys.modules["tools.sim_self_play"] is sys.modules[__name__]:
        # Also pin the parent-package attribute so attribute-style
        # access (`import tools.sim_self_play; tools.sim_self_play.X`)
        # sees the same object; a sys.modules cache hit alone does not
        # set it. `tools` is already imported (module-level imports
        # above pulled tools.scenario_pool).
        setattr(sys.modules["tools"], "sim_self_play",
                sys.modules[__name__])
    sys.exit(main(sys.argv))
