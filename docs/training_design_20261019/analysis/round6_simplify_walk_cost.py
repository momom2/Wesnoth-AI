"""Round 6, simplify stance: what the luck columns cost and earn in coin-round labels.

Cost: the post-hoc walk of tools/match_luck.py, timed per game in
training/metrics/match_luck_20261019/rows.jsonl (field "secs", one walk per game, 31 jobs on an
EPYC 7D12 slice: box.txt beside it). Against generation's CPU per game: the seed's 17,000 games
an hour on one 4090 box, spread over its effective cores (docs/box_specs.md: at least 32 asked
for; 23 is the smallest slice on record).

Earn: the label noise with the setup baseline alone against setup plus the seven columns, own1
self-play (gen_own1_selfplay, own64_selfplay in round5_provenance_match_variance.out). Equal
information then needs kept_both / kept_setup of the games.
Also: the capped games (not decisive) and their recorded z.
Standard library only.
"""
import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROWS = HERE.parents[2] / "training/metrics/match_luck_20261019/rows.jsonl"
GAMES_PER_HOUR = 17_000
ROUND_PRICE = 8 * 100_000 * 0.42 / 17_000


def main():
    rows_all = [json.loads(line) for line in ROWS.open()]
    secs = [r["secs"] for r in rows_all]
    own = [r for r in rows_all if r["match"] in ("gen_own1_selfplay", "own64_selfplay")]
    capped = [r for r in rows_all if not r["decisive"]]
    print(f"own1 self-play games {len(own)}, capped {sum(not r['decisive'] for r in own)}; "
          f"all matches: {len(capped)} capped of {len(rows_all)}, their z {sorted(set(r['z'] for r in capped))}, "
          f"ended by {sorted(set(r['ended_by'] for r in capped))}")
    mean, med = statistics.mean(secs), statistics.median(secs)
    print(f"walk: {len(secs)} games, mean {mean:.3f} s, median {med:.3f} s per game (one core)")
    for cores in (23, 32, 64):
        gen = cores * 3600 / GAMES_PER_HOUR
        print(f"  {cores} cores: generation {gen:.2f} core-s per game; walk adds {mean / gen:.3f} of it")
    text = (HERE / "round5_provenance_match_variance.out").read_text()
    kept = {m: (float(s), float(b)) for m, s, b in
            re.findall(r"^\s+(\S+)\s+\d+\s+([\d.]+)\s+[\d.]+\s+([\d.]+)$", text, re.M)}
    rows = [kept[m] for m in ("gen_own1_selfplay", "own64_selfplay")]
    setup = sum(r[0] for r in rows) / 2
    both = sum(r[1] for r in rows) / 2
    saved = 1 - both / setup
    print(f"own1 self-play: setup alone keeps {setup:.3f}, setup + columns {both:.3f}; "
          f"games saved at equal information {saved:.3f} = ${saved * ROUND_PRICE:.2f} of a ${ROUND_PRICE:.2f} round")
    print(f"break-even: the walk pays while it costs under {saved:.3f} of generation, i.e. on any box with "
          f"more than {mean * GAMES_PER_HOUR / (3600 * saved):.1f} effective cores")


if __name__ == "__main__":
    main()
