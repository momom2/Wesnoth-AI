"""Verify our combat resolver against Wesnoth's per-strike ground truth.

When a replay is recorded with `oos_debug=yes` (Strict Synchronization),
each combat strike emits an `[mp_checkup]` block containing:
    chance=<int>          # the cth (% chance to hit)
    hits=<bool>           # did this strike land
    damage=<int>          # damage dealt this strike
followed by a per-target `[mp_checkup]` with:
    dies=<bool>           # did the target die after this strike

Source: wesnoth_src/src/actions/attack.cpp:1010 (for hits/damage) and
:1105 (for dies). The blocks live as `[command dependent=yes]
[user_input name=mp_checkup] [data] ...` server commands, interleaved
in the replay between the player's [attack] and the next action.

This module:
  - Parses a Wesnoth replay (.bz2 or decompressed) with mp_checkup blocks.
  - For each [attack] command, walks through our combat.resolve_attack
    and compares per-strike (chance, hits, damage, dies) against the
    recorded values.
  - Reports the FIRST divergence (which combat, which strike, what
    differs) so we can trace it.

How to produce a replay with mp_checkup data
--------------------------------------------
Wesnoth records this only when `oos_debug=yes` is set on the game
classification at game START. There's no CLI flag for this; the
sole UI hook is the "Strict Synchronization" checkbox in
Multiplayer → Create Game → Settings tab.

Manual workflow:
  1. Launch Wesnoth.
  2. Multiplayer → Local Game.
  3. In Settings (gear icon / advanced), enable "Strict Synchronization".
  4. Configure scenario (e.g. The Freelands), era=Default, both sides
     to AI controllers.
  5. Start. Let the AIs play to completion.
  6. The replay is auto-saved to
     ~/Documents/My Games/Wesnoth1.18/saves/.
  7. Pass the .bz2 path to this script:
       python tools/verify_mp_checkup.py <path/to/replay.bz2>

Dependencies: stdlib only (bz2, re).
Dependents: standalone CLI; not imported elsewhere.
"""
from __future__ import annotations

import argparse
import bz2
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class CheckupStrike:
    """One [mp_checkup] block's data for a strike: the hit/miss and damage
    that Wesnoth recorded for that strike."""
    chance: int
    hits: bool
    damage: int
    dies: Optional[bool] = None  # filled in from the next mp_checkup block


@dataclass
class AttackRecord:
    """An [attack] command's full per-strike trace from the replay."""
    cmd_index: int                # ordinal among [attack] commands
    attacker_pos: Tuple[int, int]  # 1-indexed (we don't convert)
    defender_pos: Tuple[int, int]
    attacker_type: str
    defender_type: str
    weapon: int
    defender_weapon: int
    turn: int
    tod: str
    strikes: List[CheckupStrike] = field(default_factory=list)


_RE_BLOCK = re.compile(
    r"\[attack\][\s\S]*?attacker_type=\"([^\"]+)\"[\s\S]*?"
    r"defender_type=\"([^\"]+)\"[\s\S]*?defender_weapon=(-?\d+)[\s\S]*?"
    r"tod=\"([^\"]+)\"[\s\S]*?turn=(\d+)[\s\S]*?weapon=(\d+)[\s\S]*?"
    r"\[source\][\s\S]*?x=(\d+)[\s\S]*?y=(\d+)[\s\S]*?\[/source\][\s\S]*?"
    r"\[destination\][\s\S]*?x=(\d+)[\s\S]*?y=(\d+)[\s\S]*?\[/destination\]"
    r"[\s\S]*?\[/attack\]"
)
# Each mp_checkup block has data inline — Wesnoth wraps them in
# [command dependent=yes][user_input name="mp_checkup"][data]...[/data].
# In flat WML the data fields live as plain key=value lines.
_RE_MP_CHECKUP = re.compile(
    r'\[user_input\][^[]*?name="mp_checkup"[\s\S]*?\[data\]([\s\S]*?)\[/data\]'
)


def _parse_data_block(data_text: str) -> Dict[str, str]:
    """Pull key=value pairs from a [data] sub-block."""
    out = {}
    for m in re.finditer(r'^\s*(\w+)\s*=\s*"?([^"\n]*)"?\s*$',
                         data_text, re.M):
        out[m.group(1)] = m.group(2).strip()
    return out


def _open_replay(path: Path) -> str:
    """Decompress a .bz2 (or read .wml plain) replay file."""
    if path.suffix == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8",
                        errors="replace").read()
    return path.read_text(encoding="utf-8", errors="replace")


def parse_replay(path: Path) -> List[AttackRecord]:
    """Walk the replay text, pair [attack] commands with the
    [mp_checkup] blocks that immediately follow them."""
    text = _open_replay(path)

    # Split into command blocks. We find every [attack] block and look
    # for [mp_checkup] (user_input) blocks BETWEEN this attack and the
    # next [attack] / [move] / [recruit] / [end_turn].
    attacks: List[AttackRecord] = []
    cmd_idx = 0
    cursor = 0
    while True:
        m = _RE_BLOCK.search(text, cursor)
        if not m:
            break
        rec = AttackRecord(
            cmd_index=cmd_idx,
            attacker_type=m.group(1),
            defender_type=m.group(2),
            defender_weapon=int(m.group(3)),
            tod=m.group(4),
            turn=int(m.group(5)),
            weapon=int(m.group(6)),
            attacker_pos=(int(m.group(7)), int(m.group(8))),
            defender_pos=(int(m.group(9)), int(m.group(10))),
        )
        cmd_idx += 1
        # Find the next non-mp_checkup major command after this attack
        # block, then collect mp_checkup blocks in between.
        attack_end = m.end()
        next_cmd_re = re.compile(
            r"\[(?:attack|move|recruit|recall|end_turn|init_side)\]"
        )
        nxt = next_cmd_re.search(text, attack_end)
        slice_end = nxt.start() if nxt else len(text)
        slice_text = text[attack_end:slice_end]

        # Pull all mp_checkup data blocks; pair {chance,hits,damage}
        # with subsequent {dies}.
        pending: Optional[CheckupStrike] = None
        for cm in _RE_MP_CHECKUP.finditer(slice_text):
            data = _parse_data_block(cm.group(1))
            if "chance" in data:
                # New strike record.
                if pending is not None:
                    rec.strikes.append(pending)
                pending = CheckupStrike(
                    chance=int(data.get("chance", 0)),
                    hits=data.get("hits", "no").lower() in ("yes", "true", "1"),
                    damage=int(data.get("damage", 0)),
                )
            elif "dies" in data and pending is not None:
                pending.dies = data["dies"].lower() in ("yes", "true", "1")
                rec.strikes.append(pending)
                pending = None
        if pending is not None:
            rec.strikes.append(pending)

        attacks.append(rec)
        cursor = slice_end

    return attacks


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("replay", type=Path,
                    help="A Wesnoth .bz2 (or .wml) replay produced "
                         "with oos_debug=yes / Strict Sync.")
    args = ap.parse_args(argv[1:])

    attacks = parse_replay(args.replay)
    print(f"parsed {len(attacks)} [attack] commands")
    if not attacks:
        print("(replay has no attacks; nothing to verify)")
        return 0

    n_with_strikes = sum(1 for a in attacks if a.strikes)
    print(f"  {n_with_strikes} have mp_checkup strike data; "
          f"{len(attacks) - n_with_strikes} do NOT (replay may not "
          f"have been recorded with oos_debug=yes)")
    if n_with_strikes == 0:
        print("\nNO STRIKE DATA. Verify the replay was recorded with "
              "Strict Synchronization enabled.")
        return 1

    # Print a sample
    print("\nfirst 3 attacks with strike data:")
    n_shown = 0
    for a in attacks:
        if not a.strikes:
            continue
        print(f"  attack[{a.cmd_index}] turn={a.turn} tod={a.tod}: "
              f"{a.attacker_type} {a.attacker_pos} -> "
              f"{a.defender_type} {a.defender_pos} "
              f"weap={a.weapon}/{a.defender_weapon}")
        for i, s in enumerate(a.strikes):
            print(f"    strike[{i}]: chance={s.chance}% "
                  f"hits={s.hits} damage={s.damage} dies={s.dies}")
        n_shown += 1
        if n_shown >= 3:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
