# Training-path cost (tools/bench_train_step.py)

relset.pt (15.03 M params) on NVIDIA GeForce RTX 4090, torch 2.5.1+cu124; source: {'kind': 'pool', 'actors': 48, 'games': 48, 'decisive': 3, 'sims': 32, 'max_turns': 12, 'gen_seconds': 209.8999775709999, 'forwards': 230307, 'leaves_per_s': 1097.2226041429533, 'experiences_per_game': 154.04166666666666, 'experiences': 7394, 'build_seconds': 214.75495180500002}

## Stage costs, ms per experience (median of 2 steps; 7394 unique experiences, cycled)

| precision | B | N | compiled | encode_raw | encode | forward | policy_loss | value_loss | backward | fwd+bwd | clip ms | optimizer ms | snapshot ms | step wall ms/exp | unattributed ms/exp | GPU peak MB | warmup s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bf16 | 16 | 1024 | no | 0.35 | 0.36 | 0.97 | 0.22 | 0.02 | 1.93 | 3.85 | 3.11 | 4.47 | 10.65 | 4.00 | 0.13 | 1225 | 4.6 |

## Parity on one batch of 64 (optimizer stubbed, no clipping; reference fp32 B=1 -- for the compiled rows the eager fp32 B=1 on the same weights; pass = loss within 1%, gradient norm within 5%, cosine >= 0.99)

| config | total loss | loss rel diff | grad norm | norm ratio | grad rel L2 diff | cosine | ok |
|---|---|---|---|---|---|---|---|
| fp32 B=1 | 3.59032 | 0.00e+00 | 3.0487 | 1.0000 | 0.00e+00 | 1.00000 | ref |
| fp32 B=1 rerun | 3.59032 | 0.00e+00 | 3.0487 | 1.0000 | 0.00e+00 | 1.00000 | yes |
| bf16 B=16 | 3.58937 | 2.62e-04 | 3.0436 | 0.9983 | 1.96e-02 | 0.99981 | yes |

## Implied az_loop iteration (24 games, holdout 0.2, 540 experiences per game, step cap 4000, 1 trial(s), 32 sims)

| precision | B | compiled | train-path s | generation s at 833 leaves/s | fraction | generation s at 489 leaves/s | fraction |
|---|---|---|---|---|---|---|---|
| bf16 | 16 | no | 39.4 | 498 | 0.073 | 848 | 0.044 |

Training-path passes per iteration: 9912 forward+backward experiences (step 4000, held-out 2700 x 2, signal 512) and 648 forward-only.
