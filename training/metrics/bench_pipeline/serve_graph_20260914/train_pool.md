# Training-path cost (tools/bench_train_step.py)

relset.pt (15.03 M params) on NVIDIA GeForce RTX 4090, torch 2.5.1+cu124; source: {'kind': 'pool', 'actors': 16, 'games': 16, 'decisive': 2, 'sims': 32, 'max_turns': 12, 'gen_seconds': 128.67353731600087, 'forwards': 77067, 'leaves_per_s': 598.9343388511674, 'experiences_per_game': 153.1875, 'experiences': 2451, 'build_seconds': 130.3279603589981}

## Stage costs, ms per experience (median of 2 steps; 2451 unique experiences, cycled)

| precision | B | N | compiled | encode_raw | encode | forward | policy_loss | value_loss | backward | fwd+bwd | clip ms | optimizer ms | snapshot ms | step wall ms/exp | unattributed ms/exp | GPU peak MB | warmup s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bf16 | 16 | 1024 | no | 0.47 | 0.17 | 0.44 | 0.10 | 0.01 | 0.94 | 2.14 | 0.94 | 1.94 | 3.44 | 2.22 | 0.08 | 1441 | 2.8 |

## Parity on one batch of 64 (optimizer stubbed, no clipping; reference fp32 B=1 -- for the compiled rows the eager fp32 B=1 on the same weights; pass = loss within 1%, gradient norm within 5%, cosine >= 0.99)

| config | total loss | loss rel diff | grad norm | norm ratio | grad rel L2 diff | cosine | ok |
|---|---|---|---|---|---|---|---|
| fp32 B=1 | 5.19519 | 0.00e+00 | 4.8706 | 1.0000 | 0.00e+00 | 1.00000 | ref |
| fp32 B=1 rerun | 5.19519 | 0.00e+00 | 4.8706 | 1.0000 | 0.00e+00 | 1.00000 | yes |
| bf16 B=16 | 5.19509 | 2.05e-05 | 4.8743 | 1.0008 | 1.71e-02 | 0.99985 | yes |

## Implied az_loop iteration (24 games, holdout 0.2, 540 experiences per game, step cap 4000, 1 trial(s), 32 sims)

| precision | B | compiled | train-path s | generation s at 833 leaves/s | fraction | generation s at 489 leaves/s | fraction |
|---|---|---|---|---|---|---|---|
| bf16 | 16 | no | 22.0 | 498 | 0.042 | 848 | 0.025 |

Training-path passes per iteration: 9912 forward+backward experiences (step 4000, held-out 2700 x 2, signal 512) and 648 forward-only.
