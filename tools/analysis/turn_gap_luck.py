#!/usr/bin/env python3
"""The in-turn combat luck of the turn-level gap's confirmed turns.

The confirmation (docs/turn_gap_ref_prereg_20260921.md) replayed each
position's base turn and best alternative under the screen's turn salt,
so the dice that made the alternative look best on the screen stayed in
it. This tool replays both turns of every confirmed position under that
salt and compares each fight's realized result with its exact outcome
distribution (`combat_outcomes.enumerate_attack_outcomes`), from the
mover's side:

- HP luck: (the mover's fighter's HP above its expectation) minus (the
  enemy fighter's HP above its expectation), summed over the turn's fights;
- kill luck: enemy kills above their probability minus own losses above
  theirs.

A positive difference (alternative minus base) is luck in the
alternative's favour. The confirmation record is
training/metrics/turn_gap_ref_20260921/confirm.json.

    python tools/analysis/turn_gap_luck.py --out RECORD.json [--dataset DIR]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bench_states import DEFAULT_DATASET, DEFAULT_MANIFEST, load_states
from tools.combat_outcomes import enumerate_attack_outcomes
from tools.turn_gap import _action_from_json, _hp_margin, sim_from_state
from wesnoth_ai.classes import Position
from wesnoth_ai.paths import REPO_ROOT

CONFIRM = REPO_ROOT / "training" / "metrics" / "turn_gap_ref_20260921" / "confirm.json"


def _unit_at(gs, x: int, y: int):
    return next((u for u in gs.map.units if u.position.x == x and u.position.y == y), None)


def _fight_luck(sim, cmd, mover: int, apply):
    """Apply one attack command and return its (HP luck, kill luck) for the
    mover, or None when the fight's distribution cannot be enumerated."""
    ax, ay, dx, dy, weapon = cmd[1], cmd[2], cmd[3], cmd[4], cmd[5]
    a, d = _unit_at(sim.gs, ax, ay), _unit_at(sim.gs, dx, dy)
    dist = None
    if a is not None and d is not None:
        try:
            dist = enumerate_attack_outcomes(
                sim.gs, {"type": "attack", "start_hex": Position(ax, ay),
                         "target_hex": Position(dx, dy), "attack_index": weapon},
                advancement_choice="uniform")
        except Exception:  # noqa: BLE001 -- an unenumerable fight is counted, not fatal
            dist = None
    apply(cmd)
    if dist is None:
        return None
    after = {u.id: u for u in sim.gs.map.units}
    a_hp = after[a.id].current_hp if a.id in after else 0
    d_hp = after[d.id].current_hp if d.id in after else 0
    expected_a = sum(p * k[0] for k, p in dist.probs.items())
    expected_d = sum(p * k[1] for k, p in dist.probs.items())
    p_kill = sum(p for k, p in dist.probs.items() if k[1] <= 0)
    p_die = sum(p for k, p in dist.probs.items() if k[0] <= 0)
    sign = 1 if a.side == mover else -1
    hp_luck = sign * ((a_hp - expected_a) - (d_hp - expected_d))
    kill_luck = sign * (((1.0 if d_hp <= 0 else 0.0) - p_kill)
                        - ((1.0 if a_hp <= 0 else 0.0) - p_die))
    return hp_luck, kill_luck


def turn_luck(gs, scenario_id: str, salt: str, actions: list, mover: int):
    """Replay one recorded turn under `salt`; its summed luck, the fights
    counted and enumerated, and the mover's HP margin after it."""
    sim = sim_from_state(gs, scenario_id, 999, salt)
    fights = []
    apply = sim._apply_with_stats

    def hooked(cmd):
        if cmd and cmd[0] == "attack":
            fights.append(_fight_luck(sim, cmd, mover, apply))
        else:
            apply(cmd)

    sim._apply_with_stats = hooked
    side = sim.current_side
    for action in actions:
        if sim.done or sim.current_side != side:
            break
        sim.step(_action_from_json(action))
        if sim.last_step_rejected:
            return None
    known = [f for f in fights if f is not None]
    return {"hp_luck": sum(f[0] for f in known), "kill_luck": sum(f[1] for f in known),
            "fights": len(fights), "fights_enumerated": len(known),
            "hp_margin_after": _hp_margin(sim.gs, mover)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--record", type=Path, default=CONFIRM)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    record = json.loads(args.record.read_text(encoding="utf-8"))
    confirmed = [p for p in record["positions"] if p["screen"].get("verdict") == "hit"]
    states = load_states(DEFAULT_MANIFEST, args.dataset, max(p["index"] for p in confirmed) + 1)
    rows = []
    for p in confirmed:
        gs, scenario_id = states[p["index"]]
        mover = gs.global_info.current_side
        base = turn_luck(gs, scenario_id, p["turn_salt"], p["base"]["actions"], mover)
        alt = turn_luck(gs, scenario_id, p["turn_salt"], p["alternatives"][0]["actions"], mover)
        row = {"position": p["index"], "scenario_id": scenario_id, "gap": p["gap"],
               "base": base, "alternative": alt,
               "hp_luck_difference": (None if base is None or alt is None
                                      else alt["hp_luck"] - base["hp_luck"])}
        rows.append(row)
        diff = row["hp_luck_difference"]
        print(f"position {p['index']:3d} gap {p['gap']:+.3f} HP luck, alternative minus base: "
              f"{'n/a' if diff is None else f'{diff:+.1f}'}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"record": str(args.record.name), "positions": rows}, indent=1),
                        encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
