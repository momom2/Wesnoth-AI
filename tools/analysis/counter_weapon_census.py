#!/usr/bin/env python3
"""Our counter-weapon choice against the one the engine recorded, on
corpus attacks.

Every corpus `[attack]` carries the defender's weapon as the attacking
client's `battle_context::choose_defender_weapon` picked it. For each
recorded attack whose defender has at least two weapons of the
attacker's range, this compares that record with
`combat_outcomes.counter_weapon_choice` twice: as it is, and with the
level-up scoring (`_levelup_average_hp`) replaced by the raw average
HP, the rule before 2026-09-25. A check that cannot tell the two apart
certifies nothing (docs/wesnoth_rules.md "The defender's counter
weapon is chosen on a prediction that counts level-ups").

A disagreement is marked `tie` when every candidate's predicted death
probabilities and average HP agree within 1e-9: the engine's choice
then rests on floating-point residue that our strike DP does not
reproduce bit for bit.

Games are taken in file order among those at least `--min-turn` turns
long (long games carry more experience), from `--offset`, until
`--budget-s` runs out.

    python tools/analysis/counter_weapon_census.py [--corpus DIR]
        [--min-turn 20] [--offset 0] [--budget-s 100] [--out FILE.json]
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools import combat_outcomes as co  # noqa: E402
from tools.replay_dataset import (_apply_command, _build_initial_gamestate,  # noqa: E402
                                  _find_unit_at, _setup_scenario_events, _stats_for,
                                  build_attack_context)

CORPUS = ROOT / "replays_dataset_imitation"
_TURNS = re.compile(r"_Turn_(\d+)_")
_TIE = 1e-9


def _choice(gs, att, dfd, a_weapon: int, *, levelup: bool):
    """`counter_weapon_choice`, with the level-up scoring on or off."""
    real = co._levelup_average_hp
    if not levelup:
        co._levelup_average_hp = lambda cu, opp_cu, avg_hp, *_: avg_hp
    try:
        return co.counter_weapon_choice(gs, att, dfd, a_weapon)
    finally:
        co._levelup_average_hp = real


def _levelup_possible(unit, opp_level: int) -> bool:
    """A kill's XP (never less than the fight's) reaches the cap."""
    return unit.current_exp + co._kill_xp(opp_level) >= unit.max_exp


def _is_tie(gs, att, dfd, a_weapon: int, tables: Dict[int, dict]) -> bool:
    marginals = []
    for i in sorted(tables):
        ctx = build_attack_context(gs, att, dfd, a_weapon, i)
        a_stats, d_stats = co._stats_pair(ctx)
        am, dm = co._engine_marginals(tables[i], a_stats, d_stats, ctx.att_cu, ctx.dfd_cu)
        marginals.append((am.death, am.avg_hp, dm.death, dm.avg_hp))
    first = marginals[0]
    return all(abs(x - y) <= _TIE for m in marginals[1:] for x, y in zip(m, first))


def _games(corpus: Path, min_turn: int, offset: int) -> List[Path]:
    files = sorted(corpus.glob("*.json.gz"))
    long_games = [f for f in files
                  if (m := _TURNS.search(f.name)) and int(m.group(1)) >= min_turn]
    return long_games[offset:]


def census(corpus: Path, min_turn: int, offset: int, budget_s: float) -> dict:
    counts = {"games": 0, "attacks_with_a_choice": 0, "agree": 0, "agree_before": 0,
              "levelup_possible": 0, "levelup_agree": 0, "levelup_agree_before": 0}
    disagreements = []
    start = time.time()
    games = _games(corpus, min_turn, offset)
    for path in games:
        if time.time() - start > budget_s:
            break
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        gs = _build_initial_gamestate(data)
        _setup_scenario_events(gs, data.get("scenario_id", ""))
        counts["games"] += 1
        for k, cmd in enumerate(data.get("commands", [])):
            if cmd and cmd[0] == "attack" and len(cmd) > 7 and cmd[6] >= 0:
                att = _find_unit_at(gs, cmd[1], cmd[2])
                dfd = _find_unit_at(gs, cmd[3], cmd[4])
                if att is not None and dfd is not None and "petrified" not in dfd.statuses:
                    now, tables = _choice(gs, att, dfd, cmd[5], levelup=True)
                    if tables:
                        before, _ = _choice(gs, att, dfd, cmd[5], levelup=False)
                        levels = (_levelup_possible(att, int(_stats_for(dfd.name).get("level", 1)))
                                  or _levelup_possible(dfd, int(_stats_for(att.name).get("level", 1))))
                        counts["attacks_with_a_choice"] += 1
                        counts["agree"] += now == cmd[6]
                        counts["agree_before"] += before == cmd[6]
                        counts["levelup_possible"] += levels
                        counts["levelup_agree"] += levels and now == cmd[6]
                        counts["levelup_agree_before"] += levels and before == cmd[6]
                        if now != cmd[6] or before != cmd[6]:
                            disagreements.append({
                                "game": path.name, "command": k,
                                "attacker": f"{att.name} {att.current_exp}/{att.max_exp} xp",
                                "defender": f"{dfd.name} {dfd.current_exp}/{dfd.max_exp} xp",
                                "recorded": cmd[6], "ours": now, "before": before,
                                "tie": _is_tie(gs, att, dfd, cmd[5], tables)})
            _apply_command(gs, cmd)
    return {"min_turn": min_turn, "offset": offset, "long_games": len(games),
            "elapsed_s": round(time.time() - start, 1), **counts,
            "disagreements": disagreements}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--min-turn", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--budget-s", type=float, default=100.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    result = census(args.corpus, args.min_turn, args.offset, args.budget_s)
    rows = result.pop("disagreements")
    print(json.dumps(result, indent=1))
    for row in rows:
        print(row)
    if args.out is not None:
        args.out.write_text(json.dumps({**result, "disagreements": rows}, indent=1) + "\n",
                            encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
