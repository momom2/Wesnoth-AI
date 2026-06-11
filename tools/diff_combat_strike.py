"""Diff our combat resolver against Wesnoth's per-strike ground truth.

Inputs:
  - A Wesnoth replay (.gz or .bz2) recorded with oos_debug=yes.
    Each combat strike in such a replay is followed by an
    [mp_checkup] block carrying {chance, damage, hits} and a
    second block with {dies}.
  - The same replay extracted by tools.replay_extract (so we can
    walk our reconstructor through commands in order).

For each [attack] command:
  - Apply our reconstructor up to (but not including) the attack.
  - Run combat.resolve_attack with the recorded seed; instrument
    `_perform_hit` to capture our per-strike (chance, hits, damage).
  - Compare against the recorded strikes from mp_checkup. Report the
    FIRST mismatch (which command, which strike, what differs).

This is the bit-exact oracle: any divergence is a sim bug.

Dependencies: tools.verify_mp_checkup, tools.replay_dataset, combat.
Dependents: standalone CLI.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Project root on sys.path.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

import combat as cb
from tools.replay_dataset import (
    _build_initial_gamestate, _apply_command, _setup_scenario_events,
    _stats_for, _to_combat_unit, _terrain_keys_at, _terrain_def_pct,
    _lawful_bonus_at,
)
from tools.abilities import leadership_bonus, is_backstab_active
from tools.verify_mp_checkup import parse_replay


@dataclass
class StrikeMismatch:
    cmd_index: int
    strike_index: int
    field: str
    ours: object
    wesnoth: object
    detail: str


def _instrument_perform_hit():
    """Wrap combat._perform_hit to capture per-strike (cth, hits,
    damage). Returns a list that gets populated during a resolve_attack
    call; caller clears it before each combat. The wrapper is reset
    by `_uninstrument`."""
    log: List[Tuple[int, bool, int]] = []

    real = cb._perform_hit

    def wrapped(*args, **kwargs):
        s, t, ss, ts = args[0], args[1], args[2], args[3]
        cth = ss.cth
        # Wesnoth's mp_checkup records the UNCLAMPED weapon damage
        # stat (attack.cpp:1004 `damage = attacker.damage_;`), not
        # the actual HP removed. So we capture striker_stats.damage,
        # adjusted for slow.
        weapon_damage = (ss.slow_damage if s.is_slowed else ss.damage)
        target_hp_pre = t.hp
        out = real(*args, **kwargs)
        target_hp_post = t.hp
        # `hits` = the rng-determined hit/miss. We infer it from
        # whether HP changed (true hit) OR from striker_stats's
        # remaining-attacks decrement combined with hp delta. For
        # damage>0 weapons, hits iff hp changed.
        hits = (target_hp_pre != target_hp_post)
        # Edge: if weapon_damage == 0 (e.g., slowed unit with
        # slow_damage 0), the strike might HIT but deal no damage.
        # The combat code returns True without changing hp in this
        # case (line: `if dmg <= 0: return True`). We have no way
        # to distinguish hit-with-0-damage from miss without
        # additional instrumentation, so we approximate with hp
        # delta. None of the default-era weapons have base damage 0.
        log.append((cth, hits, weapon_damage if hits else 0))
        return out

    cb._perform_hit = wrapped
    return log, real


def _uninstrument(real):
    cb._perform_hit = real


def _verify_attack(
    gs, cmd, recorded_strikes,
) -> List[StrikeMismatch]:
    """Run our combat for cmd and compare to recorded strikes."""
    ax, ay, dx, dy = cmd[1], cmd[2], cmd[3], cmd[4]
    a_weapon = cmd[5]
    d_weapon = cmd[6] if len(cmd) > 6 else -1
    seed = cmd[7] if len(cmd) > 7 and cmd[7] else "00000000"

    def at(x, y):
        for u in gs.map.units:
            if u.position.x == x and u.position.y == y:
                return u
        return None

    att = at(ax, ay)
    dfd = at(dx, dy)
    if att is None or dfd is None:
        return [StrikeMismatch(
            cmd_index=-1, strike_index=-1, field="missing_units",
            ours=None, wesnoth=None,
            detail=f"att at ({ax},{ay}): {att}; "
                   f"dfd at ({dx},{dy}): {dfd}",
        )]

    a_def_table = (getattr(att, "_defense_table", None)
                   or _stats_for(att.name).get("defense", {}))
    d_def_table = (getattr(dfd, "_defense_table", None)
                   or _stats_for(dfd.name).get("defense", {}))
    a_def_pct = _terrain_def_pct(gs, ax, ay, a_def_table)
    d_def_pct = _terrain_def_pct(gs, dx, dy, d_def_table)
    att_cu = _to_combat_unit(att, _terrain_keys_at(gs, ax, ay),
                             defense_pct=a_def_pct)
    dfd_cu = _to_combat_unit(dfd, _terrain_keys_at(gs, dx, dy),
                             defense_pct=d_def_pct)
    turn = gs.global_info.turn_number
    a_law = _lawful_bonus_at(gs, ax, ay, turn)
    d_law = _lawful_bonus_at(gs, dx, dy, turn)
    a_lev = int(_stats_for(dfd.name).get("level", 1))
    d_lev = int(_stats_for(att.name).get("level", 1))
    a_lead = leadership_bonus(att, gs.map.units, opponent_level=a_lev)
    d_lead = leadership_bonus(dfd, gs.map.units, opponent_level=d_lev)
    a_bs = is_backstab_active(att, dfd, gs.map.units)
    d_bs = is_backstab_active(dfd, att, gs.map.units)

    log, real = _instrument_perform_hit()
    try:
        cb.resolve_attack(
            att_cu, dfd_cu,
            a_weapon_idx=a_weapon,
            d_weapon_idx=d_weapon if d_weapon >= 0 else None,
            a_lawful_bonus=a_law,
            d_lawful_bonus=d_law,
            a_leadership_bonus=a_lead,
            d_leadership_bonus=d_lead,
            a_backstab_active=a_bs,
            d_backstab_active=d_bs,
            rng=cb.MTRng(seed),
        )
    finally:
        _uninstrument(real)

    mismatches: List[StrikeMismatch] = []
    n = max(len(log), len(recorded_strikes))
    for i in range(n):
        if i >= len(log):
            mismatches.append(StrikeMismatch(
                cmd_index=-1, strike_index=i, field="strike_count",
                ours=len(log), wesnoth=len(recorded_strikes),
                detail=f"Wesnoth has more strikes ({len(recorded_strikes)}) "
                       f"than ours ({len(log)})",
            ))
            break
        if i >= len(recorded_strikes):
            mismatches.append(StrikeMismatch(
                cmd_index=-1, strike_index=i, field="strike_count",
                ours=len(log), wesnoth=len(recorded_strikes),
                detail=f"Ours has more strikes ({len(log)}) "
                       f"than Wesnoth ({len(recorded_strikes)})",
            ))
            break
        our_cth, our_hits, our_dmg = log[i]
        rec = recorded_strikes[i]
        if our_cth != rec.chance:
            mismatches.append(StrikeMismatch(
                cmd_index=-1, strike_index=i, field="chance",
                ours=our_cth, wesnoth=rec.chance,
                detail=f"strike {i}: cth ours={our_cth}% "
                       f"wesnoth={rec.chance}%",
            ))
            break
        if our_hits != rec.hits:
            mismatches.append(StrikeMismatch(
                cmd_index=-1, strike_index=i, field="hits",
                ours=our_hits, wesnoth=rec.hits,
                detail=f"strike {i}: hits ours={our_hits} "
                       f"wesnoth={rec.hits} (cth={rec.chance}%)",
            ))
            break
        if our_hits and our_dmg != rec.damage:
            mismatches.append(StrikeMismatch(
                cmd_index=-1, strike_index=i, field="damage",
                ours=our_dmg, wesnoth=rec.damage,
                detail=f"strike {i}: damage ours={our_dmg} "
                       f"wesnoth={rec.damage} (cth={rec.chance}%, "
                       f"hits={rec.hits})",
            ))
            break
    return mismatches


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("strict_replay", type=Path,
                    help="Original .gz/.bz2 replay with oos_debug=yes.")
    ap.add_argument("extracted", type=Path,
                    help="Extracted .json.gz from replay_extract.")
    ap.add_argument("--max-attacks", type=int, default=None,
                    help="Stop after N attacks (default: all)")
    args = ap.parse_args(argv[1:])

    print(f"parsing strict-sync replay: {args.strict_replay.name}")
    wesnoth_attacks = parse_replay(args.strict_replay)
    print(f"  {len(wesnoth_attacks)} attacks with strike data")

    with gzip.open(args.extracted, "rt", encoding="utf-8") as f:
        data = json.load(f)
    print(f"loaded extracted commands: "
          f"{len(data.get('commands', []))}")

    gs = _build_initial_gamestate(data)
    _setup_scenario_events(gs, data.get("scenario_id", ""))

    cmds = data["commands"]
    attack_idx = 0
    n_checked = 0
    n_clean = 0
    first_mismatch: Optional[StrikeMismatch] = None
    first_mismatch_attack: Optional[int] = None

    for i, cmd in enumerate(cmds):
        if cmd[0] == "attack":
            if (args.max_attacks is not None
                    and n_checked >= args.max_attacks):
                break
            if attack_idx >= len(wesnoth_attacks):
                print(f"  warn: cmd[{i}] is the {attack_idx + 1}th "
                      f"attack but Wesnoth recorded only "
                      f"{len(wesnoth_attacks)}; stopping")
                break
            recorded = wesnoth_attacks[attack_idx]
            mismatches = _verify_attack(gs, cmd, recorded.strikes)
            if mismatches:
                first_mismatch = mismatches[0]
                first_mismatch_attack = attack_idx
                print(f"  FAIL cmd[{i}] (attack #{attack_idx}): "
                      f"{first_mismatch.detail}")
                # Show context.
                ax, ay, dx, dy = cmd[1:5]
                print(f"      {recorded.attacker_type} "
                      f"({ax},{ay}) -> {recorded.defender_type} "
                      f"({dx},{dy}) "
                      f"weap={cmd[5]}/{cmd[6]} seed={cmd[7]}")
                print(f"      turn={recorded.turn} tod={recorded.tod}")
                # Apply the command anyway so we can keep walking.
                # break  # uncomment to stop at first
            else:
                n_clean += 1
            n_checked += 1
            attack_idx += 1
            if first_mismatch is not None:
                break
        _apply_command(gs, cmd)

    print()
    print(f"checked {n_checked} attacks, {n_clean} clean, "
          f"{n_checked - n_clean} with mismatches")
    if first_mismatch is None:
        print("ALL COMBATS BIT-EXACT MATCH WESNOTH")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
