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
