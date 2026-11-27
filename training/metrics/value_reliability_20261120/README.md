# The value estimate's reliability (2026-11-20)

The reading registered in `docs/value_reliability_prereg_20261120.md`,
whose "Results" holds the numbers and the verdict.

- `read_value.py`: the reading. It only reads the decision logs of the
  simple loop's games (`wesnoth_runs\simple_loop\round_NNNN\games_*`).
- `rounds_177_208.json`: the registered reading.
- `rounds_001_032.json`: the same reading over the run's first 32
  rounds, beside it.

To read other rounds:

    python training/metrics/value_reliability_20261120/read_value.py --run C:/Users/momom2/wesnoth_runs/simple_loop --rounds 209 240 --out out.json

The follow-ups of 2026-11-21 and 2026-11-23 (the document's two
"Follow-up" sections) read the same logs:

- `where_optimism.py` (`where_*.json`): the estimate less the result by
  place in a turn, kind of decision, side, faction and own turn.
- `hidden_enemy.py`, `hidden_crosstabs.py` (`hidden_177_208.json`,
  `seen_177_208.json`): the same by what the side cannot see, from
  replayed games.
- `early_faction.py` (`early_faction_577_608.json`,
  `early_faction_673_704.json`; 2026-11-27, for
  `docs/audit_fixes_prereg_20261125.md`, "The value estimate since the
  switch"): the estimate less the result at the first decision of a
  side's first ten turns, split by whether the enemy unit types the side
  has seen name the enemy's faction, from replayed games. The windows
  `where_545_576.json` to `where_673_704.json` were read for the same
  section.
- `lambda_clock.py` (`clock_177_208.json`, `clock_393_424.json`): what a
  weight with λ under 1 becomes, fading by the decision or by the turn,
  and how it differs from today's by kind of decision.
