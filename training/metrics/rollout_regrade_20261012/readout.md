# Rollout re-grade readout: Fail

the gain is not above 0 by 2 SE; margin_r2: selection gain -0.002 +- 0.018 over 198 positions; 36 of 198 picks the continue edit, gain without them -0.002 +- 0.021 over 162 positions

records: {"benchmark_candidates": 966, "recorded": 966, "complete": 966, "failed": 0, "continue_edits": 173}
rollouts ended by read: {"2": 0.04192546583850932, "4": 0.08967391304347826}

## adjusted truth

| grader | corrected r | minus HP margin | selection gain | minus HP margin | continue picks | gain without them |
|---|---|---|---|---|---|---|
| hp_margin | 0.380 +- 0.097 | - | -0.010 +- 0.017 | - | 6/198 | -0.011 +- 0.018 |
| margin_r2 | 0.317 +- 0.054 | -0.064 +- 0.090 | -0.002 +- 0.018 | +0.007 +- 0.017 | 36/198 | -0.002 +- 0.021 |
| margin_r4 | 0.302 +- 0.051 | -0.079 +- 0.100 | +0.006 +- 0.017 | +0.015 +- 0.019 | 43/198 | -0.002 +- 0.020 |
| value_r2 | 0.305 +- 0.054 | -0.076 +- 0.083 | +0.027 +- 0.019 | +0.037 +- 0.019 | 38/198 | +0.024 +- 0.021 |
| value_r4 | 0.309 +- 0.052 | -0.072 +- 0.087 | +0.003 +- 0.016 | +0.013 +- 0.018 | 25/198 | -0.003 +- 0.017 |

positions: correlation 199, gains 198

## raw truth

| grader | corrected r | minus HP margin | selection gain | minus HP margin | continue picks | gain without them |
|---|---|---|---|---|---|---|
| hp_margin | 0.461 +- 0.084 | - | +0.011 +- 0.018 | - | 6/198 | +0.012 +- 0.019 |
| margin_r2 | 0.344 +- 0.051 | -0.116 +- 0.082 | +0.006 +- 0.019 | -0.006 +- 0.018 | 36/198 | +0.005 +- 0.022 |
| margin_r4 | 0.330 +- 0.050 | -0.130 +- 0.093 | +0.018 +- 0.018 | +0.006 +- 0.020 | 43/198 | +0.008 +- 0.020 |
| value_r2 | 0.312 +- 0.055 | -0.149 +- 0.077 | +0.034 +- 0.020 | +0.023 +- 0.020 | 38/198 | +0.032 +- 0.022 |
| value_r4 | 0.319 +- 0.051 | -0.141 +- 0.079 | +0.012 +- 0.017 | +0.001 +- 0.019 | 25/198 | +0.007 +- 0.018 |

positions: correlation 199, gains 198
