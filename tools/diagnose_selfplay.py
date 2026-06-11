"""What is the policy actually doing?

Load a self-play checkpoint, play N games locally through the
simulator, print headline behavioral metrics. The smallest piece
of telemetry that answers "why isn't the policy winning":

  - Action mix per side (recruit / move / attack / end_turn %s)
  - Mean turns per game
  - Mean units alive at game end per side
  - **Closest approach**: min hex distance any of a side's units
    ever got to the enemy leader across the whole game. The
    headline no-kills signal -- if both sides' armies never close
    on the opposing leader, the policy is hoarding gold, not
    threatening.
  - **Leader movement**: did each side's leader leave its
    starting hex? Max distance from start across the game. A
    leader pinned to its start hex by the `leader_move_penalty`
    is fine; a leader that never gets engaged because the army
    doesn't push toward the enemy isn't.
  - **Attack attempts**: how often did the policy pick an attack
    action? "0 attacks across 20 games" is the smoking gun.
  - **Recruit type histogram**: which units does the policy pick?
    Strong skew to one type suggests the action sampler's prior
    or the reward shape is funneling all play through one slot.
  - **Game-ending breakdown**: max_turns / max_actions / leader_kill.

No training, no gradient updates. Each game's pending Transitions
are dropped at game end so the policy's queue stays clean for
the next iteration.

Usage:
    python tools/diagnose_selfplay.py
    python tools/diagnose_selfplay.py --checkpoint training/checkpoints/sim_selfplay_archive_20260512-180822.pt
    python tools/diagnose_selfplay.py --games 50 --max-turns 60
    python tools/diagnose_selfplay.py --seed 1 --device cuda
"""

from __future__ import annotations

import argparse
import copy
import logging
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

from classes import GameState, Unit
from rewards import hex_distance
from tools.scenario_pool import build_scenario_gamestate, random_setup
from transformer_policy import TransformerPolicy
from wesnoth_sim import PvPDefaults, WesnothSim
from tools.sim_self_play import _would_recruit_bounce, _recruit_cost_lookup


log = logging.getLogger("diagnose_selfplay")


# ---------------------------------------------------------------------
# Per-game telemetry
# ---------------------------------------------------------------------

@dataclass
class GameDiagnostics:
    """Behavioral metrics for one diagnostic game. All fields default
    to "empty" so a crash mid-game still yields a printable record."""
    game_label:        str
    winner:            int = 0
    ended_by:          str = ""
    turns:             int = 0
    # Per side: side -> {atype: count}. Side keys appear lazily as
    # each side first acts (matches play_one_game's _last_acting_side
    # pattern for N-side robustness).
    actions:           Dict[int, Counter] = field(default_factory=dict)
    # Per side -> {unit_type: count} for recruits actually committed.
    recruit_types:     Dict[int, Counter] = field(default_factory=dict)
    # Per side: min distance ANY of this side's units got to the
    # opposing side's leader, across all states. None when the
    # opposing leader was never visible / hasn't been computed yet.
    closest_approach:  Dict[int, Optional[int]] = field(default_factory=dict)
    # Per side: max hex distance the leader ever was from its
    # starting hex. 0 = leader never moved.
    leader_max_dist:   Dict[int, int] = field(default_factory=dict)
    # Per side: final living-unit count.
    units_end:         Dict[int, int] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Per-state metric updates
# ---------------------------------------------------------------------

def _leader_of(gs: GameState, side: int) -> Optional[Unit]:
    """First (and ordinarily only) is_leader=True unit for `side`,
    or None if the leader is dead / hasn't been placed yet."""
    for u in gs.map.units:
        if u.side == side and u.is_leader:
            return u
    return None


def _update_proximity(gs: GameState, diag: GameDiagnostics) -> None:
    """For each side present on the board, compute min distance from
    any of its units to the OPPOSING side's leader. Update the
    running per-game minimum. We track both sides every state so
    the metric is symmetric and decoupled from whose turn it is."""
    sides_present = sorted({u.side for u in gs.map.units})
    # Cache leader positions per side: at most 2-4 sides in a
    # standard 2p replay (plus scenery sides). Avoiding the O(N^2)
    # naive lookup for the common case.
    leader_pos: Dict[int, Tuple[int, int]] = {}
    for s in sides_present:
        leader = _leader_of(gs, s)
        if leader is not None:
            leader_pos[s] = (leader.position.x, leader.position.y)
    # For each side, find the closest of OUR units to ANY enemy
    # leader. "Enemy" = any other side in {1, 2} (we don't track
    # scenery sides as targets).
    for my_side in (1, 2):
        my_units = [u for u in gs.map.units if u.side == my_side]
        if not my_units:
            continue
        enemy_leaders = [pos for s, pos in leader_pos.items()
                         if s != my_side and s in (1, 2)]
        if not enemy_leaders:
            continue
        local_min = min(
            hex_distance(u.position.x, u.position.y, ex, ey)
            for u in my_units for (ex, ey) in enemy_leaders
        )
        prev = diag.closest_approach.get(my_side)
        if prev is None or local_min < prev:
            diag.closest_approach[my_side] = local_min


def _update_leader_movement(
    gs: GameState, diag: GameDiagnostics,
    leader_starts: Dict[int, Tuple[int, int]],
) -> None:
    """Track per-side leader max-distance-from-start. `leader_starts`
    is captured once at game start; this update inspects the live
    state. Leader killed mid-game keeps its last known max distance
    (we just don't update it once it's missing)."""
    for s, (sx, sy) in leader_starts.items():
        leader = _leader_of(gs, s)
        if leader is None:
            continue
        d = hex_distance(leader.position.x, leader.position.y, sx, sy)
        if d > diag.leader_max_dist.get(s, 0):
            diag.leader_max_dist[s] = d


# ---------------------------------------------------------------------
# One game
# ---------------------------------------------------------------------

def diagnostic_play(
    sim:         WesnothSim,
    policy:      TransformerPolicy,
    *,
    game_label:  str,
) -> GameDiagnostics:
    """Drive `sim` to completion under `policy`. No reward, no
    training -- pure behavioral instrumentation. Pending Transitions
    are dropped at the end so the policy's queue stays empty for
    the next game."""
    diag = GameDiagnostics(game_label=game_label)

    # Snapshot each side's leader starting position. If a side has
    # no leader at start (unusual scenarios) we just skip its
    # leader-movement metric.
    leader_starts: Dict[int, Tuple[int, int]] = {}
    for s in (1, 2):
        leader = _leader_of(sim.gs, s)
        if leader is not None:
            leader_starts[s] = (leader.position.x, leader.position.y)
            diag.leader_max_dist[s] = 0

    # Initial-state metrics so a side that never acts still
    # contributes one proximity sample.
    _update_proximity(sim.gs, diag)
    _update_leader_movement(sim.gs, diag, leader_starts)

    while not sim.done:
        acting_side = sim.gs.global_info.current_side
        # Same deepcopy-then-select-action pattern as the trainer:
        # the policy stashes a reference to `pre_state` for later,
        # which sim.step would otherwise stomp on.
        pre_state = copy.deepcopy(sim.gs)
        action = policy.select_action(pre_state, game_label=game_label, sim=sim)

        # Recruit-bounce retry (god-view occupied hex). Mirrors the
        # training loop; without this the action counts skew toward
        # "recruit-then-no-op" which isn't what we want to measure.
        while _would_recruit_bounce(action, sim.gs):
            tgt = action["target_hex"]
            rejected = (getattr(sim.gs.global_info,
                                "_recruit_rejected_hexes", None) or set())
            rejected.add((tgt.x, tgt.y))
            setattr(sim.gs.global_info,
                    "_recruit_rejected_hexes", rejected)
            pre_state = copy.deepcopy(sim.gs)
            action = policy.select_action(pre_state, game_label=game_label, sim=sim)

        atype = action.get("type", "end_turn")
        diag.actions.setdefault(acting_side, Counter())[atype] += 1
        if atype == "recruit":
            ut = action.get("unit_type", "")
            diag.recruit_types.setdefault(acting_side, Counter())[ut] += 1

        sim.step(action)
        _update_proximity(sim.gs, diag)
        _update_leader_movement(sim.gs, diag, leader_starts)

    # End-game metrics.
    diag.winner   = sim.winner
    diag.ended_by = sim.ended_by
    diag.turns    = sim.gs.global_info.turn_number
    for s in (1, 2):
        diag.units_end[s] = sum(1 for u in sim.gs.map.units if u.side == s)

    # Drop the pending Transitions the policy queued during
    # select_action. This is the diagnostic equivalent of
    # play_one_game's terminal observe + train_step calls: we don't
    # want to leak per-game state into the policy's training queue.
    policy.drop_pending(game_label)

    return diag


# ---------------------------------------------------------------------
# Aggregate + print
# ---------------------------------------------------------------------

def _pct(part: int, whole: int) -> str:
    return "  n/a" if whole == 0 else f"{100.0 * part / whole:4.1f}%"


def _fmt_mean(xs: List[float], width: int = 5) -> str:
    if not xs:
        return "n/a".rjust(width)
    return f"{sum(xs) / len(xs):>{width}.1f}"


def aggregate_and_print(diags: List[GameDiagnostics]) -> None:
    """Roll up per-game diagnostics, print a single text block."""
    n = len(diags)
    if n == 0:
        print("no games to summarize")
        return

    # ----- ended-by + winner breakdown -----
    ended_by_counts: Counter = Counter(d.ended_by for d in diags)
    winners = Counter(d.winner for d in diags)
    decisive = winners[1] + winners[2]

    # ----- action mix per side, aggregated -----
    KNOWN = ["recruit", "move", "attack", "end_turn"]
    totals_by_side: Dict[int, Counter] = {1: Counter(), 2: Counter()}
    for d in diags:
        for s, c in d.actions.items():
            if s in totals_by_side:
                totals_by_side[s].update(c)

    # ----- closest approach + leader movement -----
    def _present(field_name: str, side: int) -> List[int]:
        out = []
        for d in diags:
            v = getattr(d, field_name).get(side)
            if v is not None:
                out.append(v)
        return out

    # ----- recruits: top types per side -----
    recruit_totals: Dict[int, Counter] = {1: Counter(), 2: Counter()}
    for d in diags:
        for s, c in d.recruit_types.items():
            if s in recruit_totals:
                recruit_totals[s].update(c)

    # ----- print -----
    print()
    print("=" * 72)
    print(f"Diagnostic over {n} games")
    print("=" * 72)

    # Outcome line.
    print(f"Outcomes: side1_wins={winners[1]}  side2_wins={winners[2]}  "
          f"draws={winners[0]}  (decisive: {decisive}/{n})")
    print(f"Ended by: " + ", ".join(
        f"{k}={v}" for k, v in sorted(ended_by_counts.items(),
                                       key=lambda kv: -kv[1])))
    print()

    # Turns + units alive at end.
    mean_turns = sum(d.turns for d in diags) / n
    print(f"Mean turns/game: {mean_turns:.1f}")
    s1_units = [d.units_end.get(1, 0) for d in diags]
    s2_units = [d.units_end.get(2, 0) for d in diags]
    print(f"Mean units alive at end:  s1={_fmt_mean(s1_units)}  "
          f"s2={_fmt_mean(s2_units)}")
    print()

    # Action histogram per side.
    print("Action mix:")
    print(f"  {'side':<5}{'recruit':>10}{'move':>10}{'attack':>10}"
          f"{'end_turn':>10}{'other':>10}{'total':>10}")
    for s in (1, 2):
        tot = totals_by_side[s]
        total = sum(tot.values())
        other = sum(v for k, v in tot.items() if k not in KNOWN)
        row = (f"  s{s}   "
               + "".join(f"{_pct(tot.get(k, 0), total):>10}"
                         for k in KNOWN)
               + f"{_pct(other, total):>10}"
               + f"{total:>10d}")
        print(row)
    print()

    # Headline no-kills signals.
    print("No-kills indicators:")
    s1_approach = _present("closest_approach", 1)
    s2_approach = _present("closest_approach", 2)
    print(f"  closest approach to ENEMY leader (smaller = more "
          f"threatening):")
    print(f"    s1: min={min(s1_approach) if s1_approach else 'n/a'}  "
          f"median={_fmt_mean(sorted(s1_approach)[len(s1_approach)//2:len(s1_approach)//2+1])}  "
          f"mean={_fmt_mean(s1_approach)}")
    print(f"    s2: min={min(s2_approach) if s2_approach else 'n/a'}  "
          f"median={_fmt_mean(sorted(s2_approach)[len(s2_approach)//2:len(s2_approach)//2+1])}  "
          f"mean={_fmt_mean(s2_approach)}")
    s1_ldm = [d.leader_max_dist.get(1, 0) for d in diags]
    s2_ldm = [d.leader_max_dist.get(2, 0) for d in diags]
    print(f"  leader max distance from start (0 = never moved):")
    print(f"    s1: mean={_fmt_mean(s1_ldm)}  max={max(s1_ldm) if s1_ldm else 0}")
    print(f"    s2: mean={_fmt_mean(s2_ldm)}  max={max(s2_ldm) if s2_ldm else 0}")
    n_attack_games_s1 = sum(1 for d in diags if d.actions.get(1, Counter()).get("attack", 0) > 0)
    n_attack_games_s2 = sum(1 for d in diags if d.actions.get(2, Counter()).get("attack", 0) > 0)
    total_attacks_s1 = sum(d.actions.get(1, Counter()).get("attack", 0) for d in diags)
    total_attacks_s2 = sum(d.actions.get(2, Counter()).get("attack", 0) for d in diags)
    print(f"  attack attempts: s1={total_attacks_s1} across "
          f"{n_attack_games_s1}/{n} games; "
          f"s2={total_attacks_s2} across {n_attack_games_s2}/{n} games")
    print()

    # Top recruits.
    print("Top recruited unit types:")
    for s in (1, 2):
        top = recruit_totals[s].most_common(5)
        if not top:
            print(f"  s{s}: (no recruits)")
            continue
        total = sum(recruit_totals[s].values())
        print(f"  s{s}: " + ", ".join(
            f"{name}={count} ({100.0*count/total:.0f}%)"
            for name, count in top))
    print()

    print("=" * 72)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Checkpoint to evaluate. Default: "
                         "training/checkpoints/sim_selfplay.pt "
                         "(falls back to any *.pt under the dir).")
    ap.add_argument("--ckpt-dir", type=Path,
                    default=Path("training/checkpoints"),
                    help="Where to look when --checkpoint is omitted.")
    ap.add_argument("--games", type=int, default=20,
                    help="Number of diagnostic games to play.")
    ap.add_argument("--max-turns", type=int, default=60,
                    help="Per-game turn cap. Default 60; the trainer "
                         "uses 200 but for diagnostics we want fast "
                         "turnaround.")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for scenario/faction sampling.")
    ap.add_argument("--device", default="auto",
                    help="Torch device. Default 'auto' = DML "
                         "(discrete) > CUDA > CPU. Pass 'cpu' / "
                         "'cuda' / 'dml' to force.")
    ap.add_argument("--forced-faction", default=None,
                    help="Force a faction on at least one side. Pass "
                         "'none' to disable the module default "
                         "(Knalgan Alliance).")
    ap.add_argument("--log-level", default="WARNING",
                    choices=["DEBUG", "INFO", "WARNING"],
                    help="Default WARNING -- we want a clean stdout "
                         "for the summary, not per-game INFO chatter.")
    args = ap.parse_args(argv[1:])

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # -- 1. Pick the checkpoint --
    if args.checkpoint is not None:
        ckpt = args.checkpoint
        if not ckpt.exists():
            log.error(f"checkpoint not found: {ckpt}")
            return 1
    else:
        sp = args.ckpt_dir / "sim_selfplay.pt"
        if sp.exists():
            ckpt = sp
        else:
            cands = sorted(args.ckpt_dir.glob("*.pt"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
            if not cands:
                log.error(f"no *.pt under {args.ckpt_dir}")
                return 1
            ckpt = cands[0]
        print(f"auto-picked checkpoint: {ckpt}")

    # -- 2. Load policy (arch-peek matches sim_self_play.py) --
    import torch
    from tools.device_select import select_inference_device, describe_device
    device = select_inference_device(args.device)
    log.info(f"device: {describe_device(device)}")
    arch_kwargs: Dict[str, int] = {}
    try:
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
        for k in ("d_model", "num_layers", "num_heads", "d_ff"):
            v = (raw.get("arch") or {}).get(k)
            if v is not None:
                arch_kwargs[k] = int(v)
    except Exception as e:
        log.warning(f"couldn't peek arch from {ckpt}: {e!r}")
    policy = TransformerPolicy(device=device, **arch_kwargs)
    try:
        policy.load_checkpoint(ckpt)
    except RuntimeError as e:
        if "arch mismatch" in str(e).lower():
            log.warning(f"arch mismatch loading {ckpt}: {e}; using random init")
        else:
            raise

    # -- 3. Resolve forced-faction (... = module default, None = off) --
    forced_faction_arg: object = ...
    if args.forced_faction is not None:
        if args.forced_faction.lower() == "none":
            forced_faction_arg = None
        else:
            forced_faction_arg = args.forced_faction

    # -- 4. Cost lookup (for the bounce-retry recruit metadata) --
    cost_lookup = _recruit_cost_lookup()
    # cost_lookup is unused by the diagnostic loop directly (we don't
    # compute rewards), but loading it surfaces missing unit_stats.json
    # early rather than letting the first scenario silently fall over.
    _ = cost_lookup

    # -- 5. Play games --
    rng = random.Random(args.seed)
    pvp_defaults = PvPDefaults()  # standard 2p ladder economy
    diags: List[GameDiagnostics] = []
    t0 = time.perf_counter()
    for i in range(args.games):
        setup = random_setup(rng, forced_faction=forced_faction_arg)
        try:
            gs = build_scenario_gamestate(setup)
            sim = WesnothSim(gs, scenario_id=setup.scenario_id,
                             max_turns=args.max_turns)
        except Exception as e:
            log.warning(f"skipping {setup.label()}: {e}")
            continue
        if hasattr(policy, "reset_game"):
            policy.reset_game(f"diag{i}")
        try:
            diag = diagnostic_play(sim, policy, game_label=f"diag{i}")
        except Exception as e:
            log.exception(f"game diag{i} crashed: {e}")
            policy.drop_pending(f"diag{i}")
            continue
        diags.append(diag)
        # Tiny per-game pulse line so the operator knows it's
        # progressing during a long run. Goes to stderr so the
        # aggregate summary stays clean on stdout for piping.
        sys.stderr.write(
            f"  game {i+1}/{args.games}  ended_by={diag.ended_by}  "
            f"turns={diag.turns}  s1_units_end={diag.units_end.get(1, 0)}  "
            f"s2_units_end={diag.units_end.get(2, 0)}\n"
        )
        sys.stderr.flush()

    wall = time.perf_counter() - t0
    sys.stderr.write(f"all {len(diags)} games done in {wall:.1f}s\n")

    # -- 6. Aggregate + print --
    aggregate_and_print(diags)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
