# What the network is shown of a unit and of a fight (2026-11-24)

The audit behind `docs/input_facts_design_20261124.md`, whose "Why" holds
the readings. The user had watched a game of the simple loop's network
and asked whether the network sees a unit's level, resistances, defences
and damage types, and the damage an attack can do, before it attacks.

Nothing here trains or changes a network. The games were played on the
CPU beside the run.

- `measure_games.py`: plays fresh self-play games of one checkpoint at the
  loop's decode (T=1, 64 memory slots, no end-turn offset, fog) and logs,
  for every attack, what each weapon would do against the defender (the
  core's own fight statistics) beside the policy's probabilities; for
  every decision with a recruit legal, the probability of each unit type;
  and at every decision whether the core's zone-of-control flags equal
  "the neighbours of each visible enemy of level 1 or more".
- `summarise.py`: the readings of that log.
- `summary_24_games.txt`: its output on 24 games of round 513's network
  (seeds 20261202 to 20261225), and `games_24.jsonl` the games' results.
  The log itself (7.6 MB) is not kept, and the run has since removed that
  round's checkpoint: the summary is the record.
- `replay_attacks.py`: walks an exported replay on the Rust core and
  prints every attack with each weapon's numbers.
- `watched_game_attacks.txt`: its output on the game the user watched
  (round 503's network, seed 20261124, Drakes against Undead on Elensefar
  Courtyard). Hexes are 1-indexed, as Wesnoth shows them.
- `coverage.py`: which facts of the 158 unit types a default-era game can
  hold have no input column.
- `spot_rows.py`: decodes the unit rows of a real position back into
  words and compares them with the unit database.

To repeat the games on another checkpoint (about 35 seconds a game on the
desktop's CPU; beside a run, at Idle priority on 4 threads):

    python training/metrics/input_audit_20261124/measure_games.py <checkpoint.pt> out.json 20261202 24
    python training/metrics/input_audit_20261124/summarise.py out.json
