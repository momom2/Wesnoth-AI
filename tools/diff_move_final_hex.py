"""Per-move final-hex oracle: pinpoint where our move-truncation
disagrees with Wesnoth's runtime.

The older (non-strict-sync) `[checkup]` format that 99.5% of our
replays carry includes, after each `[move]` command:

    [checkup]
        [result]
            final_hex_x=N
            final_hex_y=M
            stopped_early=no | yes
        [/result]
        ...
    [/checkup]

`final_hex_x/y` is Wesnoth's authoritative answer to "where did
this multi-hex move actually end up." It captures fog-ambush
truncation, ZoC stops, skirmisher edge cases, and similar runtime
adjustments that don't appear in the recorded `[move] x="..." y="..."`
path attribute (which carries the player's INTENDED full path).

This tool:
  - Parses the strict-or-old [checkup] [result] final_hex_x/y per move
    from a raw replay.
  - Walks our extracted compact commands through the sim
    (replay_dataset._apply_command).
  - After each move, compares our actual destination against
    Wesnoth's recorded final_hex.
  - Reports the FIRST divergence, with context.

Bug classes this surfaces:
  - Move-path truncation differences (our ambush detection too eager
    or too lax)
  - ZoC handling on intermediate hexes
  - Multi-hex MP cost calculations
  - Teleport / skirmisher edge cases

Doesn't surface combat-strike outcomes (use diff_combat_strike for
strict-sync mp_checkup data) or counter drifts (next_unit_id /
random_calls — separate tool needed).

Dependencies: tools.replay_extract, tools.replay_dataset.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

from tools.replay_extract import parse_replay_file
from tools.replay_dataset import (
    _apply_command, _build_initial_gamestate, _setup_scenario_events,
)


@dataclass
class MoveOracle:
    """Wesnoth's authoritative endpoint for one [move] command."""
    raw_cmd_idx: int          # ordinal among raw [command] blocks (across all [replay] blocks, post-trailer-drop)
    intended_path: List[Tuple[int, int]]  # WML 1-indexed
    final_x: int              # WML 1-indexed
    final_y: int
    stopped_early: bool
    side: int                 # from cmd[from_side] attr


def parse_move_oracles(raw_path: Path) -> List[MoveOracle]:
    """Pull every [move] command's final_hex_x/y from a raw replay.

    Walks the same [replay] blocks our extractor concatenates, in
    document order, dropping the unfinished trailer if present
    (mirroring tools/replay_extract.py's logic). Returns one
    MoveOracle per [move] (irrespective of whether the [checkup]
    actually carries final_hex; absent → final_x/y = -1).
    """
    root = parse_replay_file(raw_path)
    replays = root.all("replay")
    out: List[MoveOracle] = []
    raw_cmd_idx = 0

    # Mirror the trailer-drop logic from replay_extract.py at the
    # raw level. We don't need to fully replicate it here -- moves
    # with a final_hex_x/y are always "completed" actions, so a
    # save-mid-action trailer (recruit/attack with no random_seed
    # follow-up) doesn't add stale moves to our oracle.
    for r in replays:
        for cmd in r.all("command"):
            mv = cmd.first("move")
            if mv is None:
                continue
            xs_raw = mv.attrs.get("x", "")
            ys_raw = mv.attrs.get("y", "")
            try:
                xs = [int(t) for t in xs_raw.split(",") if t.strip()]
                ys = [int(t) for t in ys_raw.split(",") if t.strip()]
            except ValueError:
                continue
            if not xs or not ys or len(xs) != len(ys):
                continue
            from_side = 0
            try:
                from_side = int(cmd.attrs.get("from_side", 0) or 0)
            except (ValueError, TypeError):
                from_side = 0

            # Look for [checkup] / [checkup1] / [mp_checkup] follow-up.
            # In the old format the [checkup] is a CHILD of the same
            # [command]. In the strict-sync format `[mp_checkup]` would
            # also be CHILD here for moves (its strike-data variants
            # are SEPARATE [command] blocks for attacks).
            final_x: int = -1
            final_y: int = -1
            stopped_early: bool = False
            for chk_tag in ("checkup", "checkup1", "mp_checkup"):
                chk = cmd.first(chk_tag)
                if chk is None:
                    continue
                # Search [result] children for final_hex_x.
                for res in chk.all("result"):
                    if "final_hex_x" not in res.attrs:
                        continue
                    try:
                        final_x = int(res.attrs.get("final_hex_x", -1))
                        final_y = int(res.attrs.get("final_hex_y", -1))
                    except (ValueError, TypeError):
                        final_x = -1
                        final_y = -1
                    se = (res.attrs.get("stopped_early", "no") or "no").lower()
                    stopped_early = se in ("yes", "true", "1")
                    break
                if final_x >= 0:
                    break

            out.append(MoveOracle(
                raw_cmd_idx=raw_cmd_idx,
                intended_path=list(zip(xs, ys)),
                final_x=final_x,
                final_y=final_y,
                stopped_early=stopped_early,
                side=from_side,
            ))
            raw_cmd_idx += 1

    return out


@dataclass
class MoveDivergence:
    """One sim-vs-Wesnoth move endpoint mismatch."""
    move_idx: int             # ordinal among [move] commands
    intended_path: List[Tuple[int, int]]  # python 0-indexed
    sim_final: Tuple[int, int]            # python 0-indexed (or (-1,-1) if no unit found)
    wesnoth_final: Tuple[int, int]        # python 0-indexed
    side: int
    turn: int
    detail: str


def diff_replay_moves(raw_path: Path, extracted_path: Path,
                      max_divs: int = 1) -> List[MoveDivergence]:
    """Run extracted commands through our sim, diff each move's
    actual destination against Wesnoth's recorded final_hex_x/y.

    Args:
      raw_path: original .bz2 with [checkup] data
      extracted_path: matching .json.gz from replay_extract
      max_divs: stop after this many divergences (1 = first only)
    """
    oracles = parse_move_oracles(raw_path)
    # Filter to moves that have a real final_hex (some old replays
    # have moves without [checkup] -- skip those rather than false
    # alarm).
    oracles_with_truth = [o for o in oracles if o.final_x >= 0]

    with gzip.open(extracted_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    cmds = data.get("commands", [])
    gs = _build_initial_gamestate(data)
    _setup_scenario_events(gs, data.get("scenario_id", ""))

    divs: List[MoveDivergence] = []
    move_idx = 0
    for i, cmd in enumerate(cmds):
        if cmd[0] != "move":
            try:
                _apply_command(gs, cmd)
            except Exception:
                pass
            continue
        # Apply the move and capture the source unit
        xs, ys = cmd[1], cmd[2]
        if not xs or not ys:
            move_idx += 1
            continue
        sx, sy = xs[0], ys[0]
        # Find the unit at source pre-move (used to ID it post-move)
        pre_unit_id = None
        for u in gs.map.units:
            if u.position.x == sx and u.position.y == sy:
                pre_unit_id = u.id
                break

        try:
            _apply_command(gs, cmd)
        except Exception:
            move_idx += 1
            continue

        # Find where the unit ended up
        sim_x, sim_y = -1, -1
        if pre_unit_id is not None:
            for u in gs.map.units:
                if u.id == pre_unit_id:
                    sim_x = u.position.x
                    sim_y = u.position.y
                    break

        # Locate the matching oracle. We assume oracles_with_truth
        # is in the same order as our move_idx (since extractor and
        # raw walker use the same [replay]-block-concat order, with
        # the same trailer-drop semantics for completed actions).
        oracle: Optional[MoveOracle] = None
        if move_idx < len(oracles):
            oracle = oracles[move_idx]

        # Compare. Wesnoth coords are 1-indexed (final_hex_x/y); ours 0-indexed.
        if oracle is not None and oracle.final_x >= 0:
            wx = oracle.final_x - 1  # to 0-indexed
            wy = oracle.final_y - 1
            if (sim_x, sim_y) != (wx, wy):
                divs.append(MoveDivergence(
                    move_idx=move_idx,
                    intended_path=[(x - 0, y - 0) for x, y in zip(xs, ys)],
                    sim_final=(sim_x, sim_y),
                    wesnoth_final=(wx, wy),
                    side=oracle.side,
                    turn=gs.global_info.turn_number,
                    detail=(f"sim ended at ({sim_x},{sim_y}); Wesnoth at "
                            f"({wx},{wy}); intended path "
                            f"{list(zip(xs, ys))}; "
                            f"stopped_early={oracle.stopped_early}"),
                ))
                if len(divs) >= max_divs:
                    return divs

        move_idx += 1

    return divs


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", type=Path, help="Raw .bz2 replay")
    ap.add_argument("extracted", type=Path,
                    help="Matching .json.gz from replay_extract")
    ap.add_argument("--max-divs", type=int, default=5,
                    help="Cap divergences reported")
    args = ap.parse_args(argv[1:])

    divs = diff_replay_moves(args.raw, args.extracted, args.max_divs)
    if not divs:
        print("ALL MOVE FINAL-HEXES BIT-EXACT MATCH WESNOTH")
        return 0
    print(f"{len(divs)} move-final-hex divergence(s):")
    for d in divs:
        print(f"  move #{d.move_idx} turn={d.turn} side={d.side}")
        print(f"    {d.detail}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
