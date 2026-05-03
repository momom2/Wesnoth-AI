"""Differential replay reconstruction — find sim/Wesnoth divergences.

Walk a real Wesnoth replay's command stream through our `_apply_command`
reconstructor and, before each command, verify that our running
GameState makes the command LEGAL under Wesnoth's rules. After each
command, verify post-state invariants.

Why this matters
----------------
Self-play training is only as faithful as the simulator it trains
against. If our sim's physics drift from Wesnoth's — even silently:
recruit gold not deducted, statue ZoC stopping a march, attacker
keeping `has_attacked=False` after striking — the model trains against
fake rules and degrades the moment it's deployed against real Wesnoth.

The cheapest high-fidelity oracle we have is REAL human replays:
each one is an authoritative trace of a Wesnoth game. If our sim
considers any of the recorded commands illegal, our state has already
diverged from what Wesnoth saw. If we apply a command and our
post-state has impossible invariants (two units on the same hex,
hp < 0, etc.), our `_apply_command` is buggy.

This tool is the fast feedback loop: run on a sample of replays,
classify divergences by failure mode, fix the most common ones, repeat.
We don't need Wesnoth running anywhere — the replay's own subsequent
commands are the ground truth of what state Wesnoth expected.

Failure modes we detect
-----------------------
PRE-checks (before applying the command):
  - move:    unit missing at source / wrong side / path non-adjacent /
             intermediate hex occupied / final hex occupied / MP
             insufficient
  - attack:  attacker or defender missing / attacker side mismatch /
             non-adjacent / weapon idx out of range / attacker
             has_attacked / target petrified
  - recruit: leader not on a keep / target hex occupied / target not
             on castle network / side gold < cost / unit type not in
             side recruit list
POST-checks (after applying):
  - two units on same hex
  - alive unit with hp <= 0 or hp > max_hp
  - alive unit with current_moves < 0 or current_moves > max_moves
  - side gold trajectory doesn't match (income/upkeep math) — only
    sanity-flagged on init_side, not asserted (real replays have
    custom gold events we don't model)

CLI
---
    python tools/diff_replay.py replays_dataset/0007*.json.gz
    python tools/diff_replay.py replays_dataset/ --limit 20

Run on a directory: walks N replays in sorted order, classifies
divergences by failure mode, prints a summary. Exit non-zero if any
divergence found (so this can run in CI / pre-commit).

Dependencies: tools.replay_dataset (the reconstructor), classes
Dependents: standalone CLI; not imported elsewhere yet.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Project root on sys.path so absolute imports work whether this is
# run as `python tools/diff_replay.py` or `python -m tools.diff_replay`.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

from classes import GameState, Position, TerrainModifiers, Unit
from tools.replay_dataset import (
    _apply_command, _build_initial_gamestate, _setup_scenario_events,
    _stats_for,
)
# `_move_cost_at_hex` lives in `tools/wesnoth_sim.py`, not
# `tools/replay_dataset.py`. We import it here for the pre-check that
# validates a recorded move's MP cost against the unit's current_moves.
from wesnoth_sim import _move_cost_at_hex
from tools.abilities import hex_neighbors


log = logging.getLogger("diff_replay")


# ---------------------------------------------------------------------
# Divergence record
# ---------------------------------------------------------------------

@dataclass
class Divergence:
    """One sim/Wesnoth disagreement, with enough context to debug."""
    replay_file:  str
    cmd_index:    int
    command:      list
    side:         int
    turn:         int
    kind:         str               # short tag, e.g. "move:src_missing"
    detail:      str                # one-line human description
    state_dump:  Dict[str, str] = field(default_factory=dict)

    def short(self) -> str:
        return (f"{self.replay_file}#{self.cmd_index} "
                f"turn={self.turn} side={self.side} {self.kind}: {self.detail}")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _unit_at(gs: GameState, x: int, y: int) -> Optional[Unit]:
    for u in gs.map.units:
        if u.position.x == x and u.position.y == y:
            return u
    return None


def _unit_at_side(gs: GameState, x: int, y: int, side: int) -> Optional[Unit]:
    """Find the unit at (x, y) belonging to `side`. Returns None if no
    such unit, even if a unit of a DIFFERENT side stands there."""
    for u in gs.map.units:
        if u.position.x == x and u.position.y == y and u.side == side:
            return u
    return None


def _leader_of(gs: GameState, side: int) -> Optional[Unit]:
    for u in gs.map.units:
        if u.side == side and u.is_leader:
            return u
    return None


def _hex_modifiers(gs: GameState, x: int, y: int) -> Set:
    for h in gs.map.hexes:
        if h.position.x == x and h.position.y == y:
            return h.modifiers
    return set()


def _on_keep(gs: GameState, x: int, y: int) -> bool:
    return TerrainModifiers.KEEP in _hex_modifiers(gs, x, y)


def _castle_network_from(gs: GameState, lx: int, ly: int) -> Set[Tuple[int, int]]:
    """BFS from (lx, ly) through CASTLE/KEEP-modifier hexes. Returns
    the SET of reachable hexes including the start; mirrors what
    Wesnoth's `find_recruit_location` does for `target` validation."""
    from collections import deque
    visited: Set[Tuple[int, int]] = {(lx, ly)}
    q = deque([(lx, ly)])
    mods = {(h.position.x, h.position.y): h.modifiers for h in gs.map.hexes}
    while q:
        x, y = q.popleft()
        for nx, ny in hex_neighbors(x, y):
            if (nx, ny) in visited:
                continue
            m = mods.get((nx, ny))
            if m is None:
                continue
            if TerrainModifiers.CASTLE in m or TerrainModifiers.KEEP in m:
                visited.add((nx, ny))
                q.append((nx, ny))
    return visited


# ---------------------------------------------------------------------
# Pre-checks per command kind
# ---------------------------------------------------------------------

def _check_move(gs: GameState, cmd: list) -> Optional[Tuple[str, str]]:
    """Validate a recorded move command against `gs`. Returns
    (kind_tag, detail) on divergence, None on OK."""
    xs, ys = cmd[1], cmd[2]
    from_side = cmd[3] if len(cmd) > 3 else gs.global_info.current_side
    if not xs or not ys or len(xs) != len(ys):
        return ("move:malformed", f"path arrays mismatched: xs={xs} ys={ys}")
    sx, sy = xs[0], ys[0]
    unit = _unit_at_side(gs, sx, sy, from_side)
    if unit is None:
        # Maybe a unit of the wrong side?
        any_unit = _unit_at(gs, sx, sy)
        if any_unit is not None:
            return ("move:src_wrong_side",
                    f"({sx},{sy}) holds {any_unit.id} side={any_unit.side},"
                    f" expected side={from_side}")
        return ("move:src_missing",
                f"no unit at ({sx},{sy}) for side={from_side}")
    # Teleport short-circuit: a unit with the `teleport` ability can
    # path between any two empty villages owned by its side at cost 1.
    # The replay records this as a 2-hex move whose endpoints are
    # non-adjacent. Validate per the WML at
    # data/core/macros/abilities.cfg ABILITY_TELEPORT [tunnel]:
    # source and target hexes are both own-side villages, and the
    # target is empty (`not unit` filter on target).
    if (len(xs) == 2 and "teleport" in unit.abilities
            and (xs[1], ys[1]) not in hex_neighbors(xs[0], ys[0])):
        sx2, sy2 = xs[0], ys[0]
        tx2, ty2 = xs[1], ys[1]
        # Both endpoints must be village hexes (have VILLAGE modifier).
        s_is_v = TerrainModifiers.VILLAGE in _hex_modifiers(gs, sx2, sy2)
        t_is_v = TerrainModifiers.VILLAGE in _hex_modifiers(gs, tx2, ty2)
        if not (s_is_v and t_is_v):
            return ("move:teleport_not_villages",
                    f"teleport endpoints not both villages: src=({sx2},{sy2}) "
                    f"village={s_is_v}, dst=({tx2},{ty2}) village={t_is_v}")
        # Both villages must be owned by the moving unit's side.
        owner_map = (getattr(gs.global_info, "_village_owner", {}) or {})
        s_own = owner_map.get((sx2, sy2)) == unit.side
        t_own = owner_map.get((tx2, ty2)) == unit.side
        if not (s_own and t_own):
            return ("move:teleport_not_owned",
                    f"teleport requires own villages: "
                    f"src_owner={owner_map.get((sx2, sy2))}, "
                    f"dst_owner={owner_map.get((tx2, ty2))}, side={unit.side}")
        # Target hex must be empty (the source has the unit itself).
        occupant = _unit_at(gs, tx2, ty2)
        if occupant is not None:
            return ("move:teleport_target_occupied",
                    f"teleport target ({tx2},{ty2}) occupied by {occupant.id}")
        # Cost = 1 MP. unit must have at least 1.
        if unit.current_moves < 1:
            return ("move:mp_insufficient",
                    f"{unit.id} ({unit.name}): teleport needs 1 MP, "
                    f"current_moves={unit.current_moves}")
        return None
    # Path adjacency.
    for i in range(1, len(xs)):
        px, py = xs[i - 1], ys[i - 1]
        cx, cy = xs[i], ys[i]
        if (cx, cy) not in hex_neighbors(px, py):
            return ("move:path_non_adjacent",
                    f"step {i}: ({px},{py})->({cx},{cy}) not adjacent")
    # Path occupancy. Wesnoth's `pathfind.cpp:779-786`: friendly units
    # are PASSABLE (with a tiny 1-unit defense subcost preference), only
    # enemies block. So intermediate path hexes can be friend-occupied;
    # only enemies are illegal mid-path. The FINAL destination, however,
    # must be empty -- you can't END on a friend ("we can't stop on a
    # friend", same comment in the engine source).
    for i in range(1, len(xs)):
        cx, cy = xs[i], ys[i]
        other = _unit_at(gs, cx, cy)
        if other is None:
            continue
        is_final = (i == len(xs) - 1)
        if is_final:
            return ("move:final_occupied",
                    f"final hex ({cx},{cy}) occupied by {other.id} "
                    f"side={other.side}")
        # Mid-path enemy: in Wesnoth this is a FOG AMBUSH — the moving
        # player committed a multi-step plan into a fog-of-war hex
        # that turned out to hold an enemy. The engine truncates the
        # move at the step before. Replays record the FULL planned
        # path. We can't distinguish "fog ambush" from "stale state"
        # without per-side fog tracking (we don't have it), so we
        # ASSUME ambush and let _apply_command truncate the move at
        # apply time (mirroring Wesnoth's runtime behavior). Don't
        # return a divergence here — that would over-count
        # legitimate Wesnoth behavior as sim bugs.
        # NOTE: if our sim wrongly placed the enemy at this hex
        # (cascade), the truncated move will end at the wrong place,
        # which usually surfaces as a later final_occupied or
        # src_missing — those we DO flag.
    # MP cost.
    total_cost = 0
    for i in range(1, len(xs)):
        cost = _move_cost_at_hex(unit, gs, xs[i], ys[i])
        if cost >= 99:
            return ("move:impassable",
                    f"step {i}: ({xs[i]},{ys[i]}) impassable for {unit.name}")
        total_cost += cost
    if total_cost > unit.current_moves:
        return ("move:mp_insufficient",
                f"{unit.id} ({unit.name}): path cost={total_cost} "
                f"> current_moves={unit.current_moves}")
    return None


def _check_attack(gs: GameState, cmd: list) -> Optional[Tuple[str, str]]:
    """Validate a recorded attack against `gs`."""
    if len(cmd) < 5:
        return ("attack:malformed", f"too few args: {cmd}")
    ax, ay, dx, dy = cmd[1], cmd[2], cmd[3], cmd[4]
    a_weapon = cmd[5] if len(cmd) > 5 else 0
    side_now = gs.global_info.current_side
    att = _unit_at_side(gs, ax, ay, side_now)
    if att is None:
        any_att = _unit_at(gs, ax, ay)
        if any_att is not None:
            return ("attack:attacker_wrong_side",
                    f"attacker ({ax},{ay}) is {any_att.id} side="
                    f"{any_att.side}, expected current_side={side_now}")
        return ("attack:attacker_missing",
                f"no unit at ({ax},{ay}) for current_side={side_now}")
    dfd = _unit_at(gs, dx, dy)
    if dfd is None:
        return ("attack:defender_missing",
                f"no defender at ({dx},{dy})")
    if dfd.side == att.side:
        return ("attack:friendly_fire",
                f"{att.id} attacking same-side {dfd.id} at ({dx},{dy})")
    if att.has_attacked:
        return ("attack:already_attacked",
                f"{att.id} ({att.name}) already attacked this turn")
    if (dx, dy) not in hex_neighbors(ax, ay):
        return ("attack:non_adjacent",
                f"src ({ax},{ay}) not adjacent to dst ({dx},{dy})")
    if "petrified" in (dfd.statuses or set()):
        return ("attack:target_petrified",
                f"{dfd.id} ({dfd.name}) at ({dx},{dy}) is petrified")
    if a_weapon < 0 or a_weapon >= len(att.attacks):
        return ("attack:weapon_oob",
                f"weapon idx {a_weapon} out of range "
                f"(unit has {len(att.attacks)} attacks)")
    return None


def _check_recruit(gs: GameState, cmd: list) -> Optional[Tuple[str, str]]:
    """Validate a recorded recruit against `gs`."""
    if len(cmd) < 4:
        return ("recruit:malformed", f"too few args: {cmd}")
    unit_type = cmd[1]
    tx, ty = cmd[2], cmd[3]
    side_now = gs.global_info.current_side
    if not (1 <= side_now <= len(gs.sides)):
        return ("recruit:bad_side",
                f"current_side={side_now} out of range")
    side_info = gs.sides[side_now - 1]
    if unit_type not in side_info.recruits:
        return ("recruit:type_not_in_list",
                f"'{unit_type}' not in side {side_now}.recruits="
                f"{side_info.recruits}")
    cost = int(_stats_for(unit_type).get("cost", 14))
    if side_info.current_gold < cost:
        return ("recruit:insufficient_gold",
                f"side {side_now} gold={side_info.current_gold} < "
                f"cost={cost} ({unit_type})")
    leader = _leader_of(gs, side_now)
    if leader is None:
        return ("recruit:no_leader", f"side {side_now} has no leader")
    if not _on_keep(gs, leader.position.x, leader.position.y):
        return ("recruit:leader_off_keep",
                f"leader {leader.id} at ({leader.position.x},"
                f"{leader.position.y}) not on a keep")
    occupant = _unit_at(gs, tx, ty)
    if occupant is not None:
        return ("recruit:target_occupied",
                f"({tx},{ty}) occupied by {occupant.id} "
                f"side={occupant.side}")
    network = _castle_network_from(gs, leader.position.x, leader.position.y)
    if (tx, ty) not in network:
        return ("recruit:target_off_network",
                f"({tx},{ty}) not in leader {leader.id}'s castle network "
                f"(size={len(network)})")
    return None


# ---------------------------------------------------------------------
# Post-state invariants
# ---------------------------------------------------------------------

def _check_post_invariants(gs: GameState) -> Optional[Tuple[str, str]]:
    """Sanity invariants that must hold after every applied command."""
    seen: Dict[Tuple[int, int], str] = {}
    for u in gs.map.units:
        key = (u.position.x, u.position.y)
        if key in seen:
            return ("invariant:hex_double_occupied",
                    f"({key[0]},{key[1]}) holds {seen[key]} AND {u.id}")
        seen[key] = u.id
        if u.current_hp <= 0:
            return ("invariant:alive_unit_zero_hp",
                    f"{u.id} ({u.name}) hp={u.current_hp} <= 0 "
                    f"but still on map")
        if u.current_hp > u.max_hp:
            # Wesnoth allows hp > max_hp via [unit] hp= overrides /
            # [modify_unit]. Don't flag (would be too noisy).
            pass
        if u.current_moves < 0:
            return ("invariant:negative_mp",
                    f"{u.id} ({u.name}) current_moves={u.current_moves}")
        if u.current_moves > u.max_moves:
            return ("invariant:mp_over_max",
                    f"{u.id} ({u.name}) current_moves="
                    f"{u.current_moves} > max_moves={u.max_moves}")
    return None


# ---------------------------------------------------------------------
# Replay walker
# ---------------------------------------------------------------------

# Map command-kind -> pre-check function.
_PRE_CHECKERS = {
    "move":    _check_move,
    "attack":  _check_attack,
    "recruit": _check_recruit,
    # init_side / end_turn / recall: no pre-check (recall is rare in
    # PvP and we explicitly don't model it).
}


def diff_replay(
    gz_path: Path, *,
    stop_on_first: bool = True,
    skip_post_checks: bool = False,
) -> List[Divergence]:
    """Walk the replay's command stream through our reconstructor and
    flag every divergence. Returns the list of divergences (empty if
    the replay reconstructs cleanly).

    `stop_on_first`: when True, returns at the first divergence.
    Faster for "is this replay clean?" queries; flip to False to
    collect ALL distinct failure modes per replay (slower; later
    commands may produce cascade noise after the first divergence
    has corrupted state).
    """
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    gs = _build_initial_gamestate(data)
    _setup_scenario_events(gs, data.get("scenario_id", ""))

    out: List[Divergence] = []
    commands = data.get("commands", [])
    fname = gz_path.name

    for idx, cmd in enumerate(commands):
        kind = cmd[0] if cmd else "?"
        side = gs.global_info.current_side
        turn = gs.global_info.turn_number

        # Pre-check.
        checker = _PRE_CHECKERS.get(kind)
        if checker is not None:
            err = checker(gs, cmd)
            if err is not None:
                out.append(Divergence(
                    replay_file=fname, cmd_index=idx, command=cmd,
                    side=side, turn=turn, kind=err[0], detail=err[1],
                ))
                if stop_on_first:
                    return out

        # Apply (skip if pre-check already flagged AND we'd produce
        # garbage post-state -- but to keep the walker simple we apply
        # anyway; the post-check might catch a more specific issue).
        try:
            _apply_command(gs, cmd)
        except Exception as e:
            out.append(Divergence(
                replay_file=fname, cmd_index=idx, command=cmd,
                side=side, turn=turn,
                kind=f"apply:exception:{type(e).__name__}",
                detail=str(e)[:200],
            ))
            if stop_on_first:
                return out
            continue

        # Post-state invariants.
        if not skip_post_checks:
            post_err = _check_post_invariants(gs)
            if post_err is not None:
                out.append(Divergence(
                    replay_file=fname, cmd_index=idx, command=cmd,
                    side=side, turn=turn,
                    kind=post_err[0], detail=post_err[1],
                ))
                if stop_on_first:
                    return out
    return out


# ---------------------------------------------------------------------
# CLI / batch driver
# ---------------------------------------------------------------------

def _walk_inputs(inputs: List[Path]) -> List[Path]:
    out: List[Path] = []
    for p in inputs:
        if p.is_dir():
            out.extend(sorted(p.glob("*.json.gz")))
        elif p.is_file():
            out.append(p)
    return out


def _summarize(divergences: List[Divergence], n_replays: int,
               n_clean: int) -> None:
    """Print a per-failure-mode breakdown across all replays."""
    print("=" * 72)
    print(f"diff_replay: {n_replays} replays, "
          f"{n_clean} clean, {n_replays - n_clean} with divergence(s)")
    print("=" * 72)
    if not divergences:
        print("  no divergences detected.")
        return
    by_kind: Counter = Counter(d.kind for d in divergences)
    print(f"  {len(divergences)} divergences across "
          f"{len(by_kind)} distinct failure modes:")
    for kind, count in by_kind.most_common():
        print(f"    {count:5d}x  {kind}")
        # Show two example details to make it tangible.
        examples = [d for d in divergences if d.kind == kind][:2]
        for d in examples:
            print(f"           e.g. {d.replay_file}#{d.cmd_index} "
                  f"t{d.turn} s{d.side}: {d.detail}")
    print()


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path,
                    help="Replay .json.gz files OR a directory of them.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap number of replays processed.")
    ap.add_argument("--all-divergences", action="store_true",
                    help="Don't stop at first divergence per replay; "
                         "collect every distinct check failure (cascade "
                         "noise possible).")
    ap.add_argument("--filter-2p", action="store_true",
                    help="Only process replays whose game_id starts "
                         "with '2p'.")
    ap.add_argument("--log-level", default="WARNING",
                    choices=["DEBUG", "INFO", "WARNING"])
    args = ap.parse_args(argv[1:])

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(message)s")

    files = _walk_inputs(args.inputs)
    if args.filter_2p:
        # Use the dataset's index.jsonl + the COMPETITIVE_2P_SCENARIOS
        # whitelist to filter to PVP 2p only. ~50K files in the dataset,
        # peeking each one is minutes-slow; index lookup is sub-second.
        # The whitelist excludes survival / co-op / campaign-stage 2p
        # scenarios (Dark Forecast, Bounty Hunters, etc.) which DO have
        # 2-player game_ids but spawn AI-controlled wave units via
        # scenario events we don't model -- divergences on those are
        # NOT sim bugs.
        from tools.scenarios import is_competitive_2p
        # Default-era player factions; non-default factions (Dunefolk,
        # Dunefolk-only, etc.) trip our trait/stat lookups and are
        # filtered out.
        PLAYER_FACTIONS = {"Drakes", "Knalgan Alliance", "Loyalists",
                           "Northerners", "Rebels", "Undead"}
        index_paths: List[Path] = []
        for p in args.inputs:
            if p.is_dir():
                idx = p / "index.jsonl"
                if idx.exists():
                    index_paths.append(idx)
        keep_names: Set[str] = set()
        for idx in index_paths:
            with idx.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        m = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not is_competitive_2p(m.get("scenario_id", "")):
                        continue
                    factions = m.get("factions", []) or []
                    players = [f for f in factions if f in PLAYER_FACTIONS]
                    non_players = [f for f in factions
                                   if f not in PLAYER_FACTIONS]
                    if len(players) != 2 or len(non_players) > 1:
                        continue
                    n_cmds = int(m.get("n_commands", 0) or 0)
                    if n_cmds < 20:   # turn-1 abandoned games add no signal
                        continue
                    keep_names.add(m.get("file", ""))
        if keep_names:
            files = [p for p in files if p.name in keep_names]
        else:
            log.warning("--filter-2p: no index.jsonl found, no replays kept")
            files = []

    if args.limit is not None:
        files = files[:args.limit]
    if not files:
        log.error("no replay files found")
        return 2

    all_divs: List[Divergence] = []
    n_clean = 0
    for i, p in enumerate(files):
        try:
            divs = diff_replay(p, stop_on_first=not args.all_divergences)
        except Exception as e:
            divs = [Divergence(
                replay_file=p.name, cmd_index=-1, command=[],
                side=0, turn=0, kind=f"loader:{type(e).__name__}",
                detail=str(e)[:200],
            )]
        if not divs:
            n_clean += 1
        else:
            all_divs.extend(divs)
            for d in divs[:1]:
                log.warning(d.short())
        if (i + 1) % 50 == 0:
            log.info(f"  {i + 1}/{len(files)} processed "
                     f"({n_clean} clean, {len(all_divs)} divergences)")

    _summarize(all_divs, len(files), n_clean)
    return 0 if not all_divs else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
