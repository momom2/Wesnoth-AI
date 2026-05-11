# Cliffness empirical calibration

> **CAVEAT:** This run used a model whose value head was **randomly initialized** (either no checkpoint, or a pre-2026-05-10 checkpoint that predates the C51 head). The histogram below shows the C51 *uniform-prior* baseline, not a calibrated trained-model distribution. Re-run this tool after a fresh checkpoint exists from a C51-aware training run; the meaningful numbers will live in a follow-up section below this one.

Collected 1550 cliffness values from 1550 decision steps across 20 replays.

- **Checkpoint:** `training\checkpoints\supervised_epoch3.pt (pre-C51, partial load)`
- **Atom range:** [-1, +1] over K=51 atoms
- **Theoretical max (point mass at +/-1):** ~1.0
- **Uniform-prior baseline (current default `cliffness_max`):** 0.589

## Percentiles

| Pct | Cliffness |
|----:|----------:|
|   1 |    0.5830 |
|   5 |    0.5852 |
|  10 |    0.5872 |
|  25 |    0.5893 |
|  50 |    0.5913 |
|  75 |    0.5932 |
|  90 |    0.5951 |
|  95 |    0.5962 |
|  99 |    0.5981 |

- **min:** 0.5819
- **mean:** 0.5911
- **std:** 0.0031
- **max:** 0.5994

## Histogram (16 bins, 0 to max)

    [0.000, 0.037)     0  
    [0.037, 0.075)     0  
    [0.075, 0.112)     0  
    [0.112, 0.150)     0  
    [0.150, 0.187)     0  
    [0.187, 0.225)     0  
    [0.225, 0.262)     0  
    [0.262, 0.300)     0  
    [0.300, 0.337)     0  
    [0.337, 0.375)     0  
    [0.375, 0.412)     0  
    [0.412, 0.450)     0  
    [0.450, 0.487)     0  
    [0.487, 0.524)     0  
    [0.524, 0.562)     0  
    [0.562, 0.599)  1550  ########################################

## Calibration recommendation

- Current default `cliffness_max = 0.589` (uniform-prior baseline).
- Empirical p99 = 0.598. If p99 < default, the adaptive-sim-budget rarely saturates and `n_simulations_max` is mostly unreached. Consider lowering `cliffness_max` to the empirical p95 (0.595) so the budget actually varies across positions.
- If p99 > default, the model has more value uncertainty than a uniform prior would predict (suggests the C51 head has learned to spread mass on bimodal positions); keep `cliffness_max` near the uniform baseline or raise it to p99 to avoid clipping.
- For `cliffness_bootstrap_alpha`: with empirical std at 0.003, alpha=1.0 (Bayes-optimal) means a typical leaf's contribution to ancestor Q is downweighted by `1 / (1 + cliffness^2)` ~ 0.74 on average. Start with alpha=1.0 on a short calibration run; if MCTS Q-values drift noisily, lower alpha; if the search ignores cliff signal entirely, raise it.