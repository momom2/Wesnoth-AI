"""Audit `tools/replay_extract.py`'s save-mid-action trailer-drop
heuristic.

When Wesnoth saves DURING a player's turn, the unfinished player
action gets flushed to the end of `[replay][i]` without its
`[random_seed]`, then re-emitted at start of `[replay][i+1]` --
either the same action (with seed) or a replacement after undo.
Our extractor drops the unfinished trailer at each block boundary
to avoid duplicates.

Risk: if a LEGITIMATELY-completed action at end of block i has
no `[random_seed]` follow-up (because its action type doesn't
consume RNG), the heuristic drops it spuriously. The candidate
class is musthave-only recruits (Walking Corpse, mechanical
units, elemental) — they don't roll random traits and so don't
emit `[random_seed]`.

This script walks every continuation-save (replay with 2+ non-
empty `[replay]` blocks) in the raw corpus and, for each, prints:
  - block 0's last player action (the would-be-dropped action)
  - block 1's first player action (the post-load redo / replacement)
  - whether they match (genuine duplicate -> correct drop)
  - whether the dropped action is a recruit of a musthave-only
    type (false-drop risk)

Outcome: a single tally tells us if the heuristic is sound or
needs tightening.

Dependencies: tools.replay_extract.
Dependents: standalone CLI.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

from tools.replay_extract import parse_replay_file
from tools.replay_dataset import _stats_for


# Races whose unit types have ONLY musthave traits (no random pool),
# so a recruit of these types DOESN'T emit a [random_seed] even
# when fully completed in Wesnoth. A trailer-drop on such a recruit
# would be a FALSE drop (block 1 won't redo it).
_MUSTHAVE_ONLY_RACES = {"undead", "mechanical", "elemental"}


def _action_signature(cmd_node) -> Optional[Tuple]:
    """Return a content-key for the player action inside a [command]
    node, or None if it has no player action."""
    for sub in cmd_node.children:
        if sub.tag == "init_side":
            return ("init_side", sub.attrs.get("side_number", ""))
        if sub.tag == "end_turn":
            return ("end_turn",)
        if sub.tag == "move":
            return ("move", sub.attrs.get("x", ""), sub.attrs.get("y", ""))
        if sub.tag == "attack":
            src = sub.first("source")
            dst = sub.first("destination")
            if src is None or dst is None:
                return None
            return ("attack",
                    src.attrs.get("x", ""), src.attrs.get("y", ""),
                    dst.attrs.get("x", ""), dst.attrs.get("y", ""),
                    sub.attrs.get("weapon", ""),
                    sub.attrs.get("defender_weapon", ""))
        if sub.tag == "recruit":
            return ("recruit",
                    sub.attrs.get("type", ""),
                    sub.attrs.get("x", ""), sub.attrs.get("y", ""))
        if sub.tag == "recall":
            return ("recall",
                    sub.attrs.get("value", ""),
                    sub.attrs.get("x", ""), sub.attrs.get("y", ""))
    return None


def _last_action_in(block) -> Optional[Tuple]:
    """Last player action in a [replay] block."""
    cmds = block.all("command")
    for c in reversed(cmds):
        sig = _action_signature(c)
        if sig is not None:
            return sig
    return None


def _first_action_in(block) -> Optional[Tuple]:
    """First player action in a [replay] block (skipping non-action
    children like [speak], [chat], etc.)."""
    for c in block.all("command"):
        sig = _action_signature(c)
        if sig is not None:
            return sig
    return None


def _has_random_seed_after(block, target_action: Tuple) -> bool:
    """Did a [random_seed] command appear AFTER the last occurrence
    of `target_action` in the block? Naive but adequate for the
    save-mid-action question: we just want to know if the trailer
    got its seed within the block."""
    cmds = block.all("command")
    after_target = False
    for c in cmds:
        if not after_target:
            sig = _action_signature(c)
            if sig == target_action:
                after_target = True
            continue
        # We're past the target now — look for [random_seed] in any
        # subsequent command.
        for sub in c.children:
            if sub.tag == "random_seed":
                return True
    return False


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_dir", type=Path,
                    help="Directory of raw .bz2 replays")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--show-suspicious", type=int, default=10,
                    help="Print up to N suspicious cases (potential false drops)")
    args = ap.parse_args(argv[1:])

    files = sorted(args.raw_dir.rglob("*.bz2"))
    if args.limit:
        files = files[:args.limit]
    print(f"Scanning {len(files)} replays...")

    stats = Counter()
    suspicious: List[Tuple[str, Tuple, Tuple]] = []

    for i, p in enumerate(files):
        try:
            root = parse_replay_file(p)
        except Exception:
            stats["parse_error"] += 1
            continue

        replays = root.all("replay")
        if len(replays) < 2:
            stats["single_block"] += 1
            continue

        # Walk consecutive (block_i, block_{i+1}) pairs.
        for bidx in range(len(replays) - 1):
            b0 = replays[bidx]
            b1 = replays[bidx + 1]
            last0 = _last_action_in(b0)
            first1 = _first_action_in(b1)
            if last0 is None:
                stats["block_no_action"] += 1
                continue

            # Did our dropper fire?  Heuristic: drop iff last0 is
            # recruit/attack AND no [random_seed] in b0 after it.
            kind = last0[0]
            if kind not in ("recruit", "attack"):
                stats["trailer_not_rng_consumer"] += 1
                continue
            had_seed = _has_random_seed_after(b0, last0)
            if had_seed:
                stats["trailer_completed_no_drop"] += 1
                continue
            # Dropper fires here.
            stats["dropper_fired"] += 1
            if first1 is None:
                stats["dropper_fired_no_block1_action"] += 1
                continue

            if last0 == first1:
                stats["drop_genuine_duplicate"] += 1
            else:
                # Dropped action != redo. Either undo+redo (correct
                # drop) or false drop (musthave-only recruit).
                if kind == "recruit":
                    rtype = last0[1]
                    s = _stats_for(rtype) if rtype else {}
                    race = s.get("race", "")
                    trait_info = s.get("traits") or {}
                    pool = trait_info.get("pool", []) if isinstance(trait_info, dict) else []
                    if race in _MUSTHAVE_ONLY_RACES or not pool:
                        stats["drop_suspicious_musthave_recruit"] += 1
                        if len(suspicious) < args.show_suspicious:
                            suspicious.append((p.name, last0, first1))
                        continue
                stats["drop_undo_redo_or_replaced"] += 1
                # Show every undo/redo for sanity-check.
                if len(suspicious) < args.show_suspicious:
                    suspicious.append((p.name + " [undo/redo]",
                                       last0, first1))

        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(files)}: dropper_fired="
                  f"{stats['dropper_fired']} "
                  f"genuine={stats['drop_genuine_duplicate']} "
                  f"undo={stats['drop_undo_redo_or_replaced']} "
                  f"suspicious={stats['drop_suspicious_musthave_recruit']}")

    print()
    print("=" * 72)
    print("Trailer-drop audit summary")
    print("=" * 72)
    for k in [
        "single_block",
        "trailer_not_rng_consumer",
        "trailer_completed_no_drop",
        "block_no_action",
        "dropper_fired",
        "dropper_fired_no_block1_action",
        "drop_genuine_duplicate",
        "drop_undo_redo_or_replaced",
        "drop_suspicious_musthave_recruit",
        "parse_error",
    ]:
        n = stats.get(k, 0)
        if n:
            print(f"  {n:6d}  {k}")

    if suspicious:
        print()
        print(f"Suspicious cases (potential false drops):")
        for fname, last0, first1 in suspicious:
            print(f"  {fname}")
            print(f"    dropped (block0 last):  {last0}")
            print(f"    block1 first:           {first1}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
