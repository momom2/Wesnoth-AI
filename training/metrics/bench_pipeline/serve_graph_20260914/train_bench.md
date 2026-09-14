# Training-path cost (tools/bench_train_step.py)

relset.pt (15.03 M params) on NVIDIA GeForce RTX 4090, torch 2.5.1+cu124; source: {'kind': 'bench', 'manifest': '/workspace/Wesnoth-AI/configs/bench_states.json', 'states': 200, 'visits': '32 draws from the prior', 'experiences': 200, 'build_seconds': 33.60156443699816}

## Stage costs, ms per experience (median of 2 steps; 200 unique experiences, cycled)

| precision | B | N | compiled | encode_raw | encode | forward | policy_loss | value_loss | backward | fwd+bwd | clip ms | optimizer ms | snapshot ms | step wall ms/exp | unattributed ms/exp | GPU peak MB | warmup s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bf16 | 16 | 1024 | no | 33.99 | 0.23 | 0.71 | 0.74 | 0.01 | 1.13 | 36.81 | 1.24 | 2.07 | 6.69 | 36.94 | 0.12 | 1338 | 52.9 |

## Parity on one batch of 64 (optimizer stubbed, no clipping; reference fp32 B=1 -- for the compiled rows the eager fp32 B=1 on the same weights; pass = loss within 1%, gradient norm within 5%, cosine >= 0.99)

| config | total loss | loss rel diff | grad norm | norm ratio | grad rel L2 diff | cosine | ok |
|---|---|---|---|---|---|---|---|
| fp32 B=1 | 5.65324 | 0.00e+00 | 2.9005 | 1.0000 | 0.00e+00 | 1.00000 | ref |
| fp32 B=1 rerun | 5.65324 | 0.00e+00 | 2.9005 | 1.0000 | 0.00e+00 | 1.00000 | yes |
| bf16 B=16 | 5.65372 | 8.37e-05 | 2.9056 | 1.0018 | 2.30e-02 | 0.99974 | yes |

## Implied az_loop iteration (24 games, holdout 0.2, 540 experiences per game, step cap 4000, 1 trial(s), 32 sims)

| precision | B | compiled | train-path s | generation s at 833 leaves/s | fraction | generation s at 489 leaves/s | fraction |
|---|---|---|---|---|---|---|---|
| bf16 | 16 | no | 387.5 | 498 | 0.438 | 848 | 0.314 |

Training-path passes per iteration: 9912 forward+backward experiences (step 4000, held-out 2700 x 2, signal 512) and 648 forward-only.
