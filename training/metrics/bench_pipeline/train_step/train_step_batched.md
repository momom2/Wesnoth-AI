# Training-path cost (tools/bench_train_step.py)

seed.pt (15.03 M params) on NVIDIA GeForce RTX 4090, torch 2.5.1+cu124; source: {'kind': 'bench', 'manifest': '/workspace/Wesnoth-AI/configs/bench_states.json', 'states': 200, 'visits': '32 draws from the prior', 'experiences': 200, 'build_seconds': 23.025423901621252}

## Stage costs, ms per experience (median of 2 steps; 200 unique experiences, cycled)

| precision | B | N | compiled | encode_raw | encode | forward | policy_loss | value_loss | backward | fwd+bwd | clip ms | optimizer ms | snapshot ms | step wall ms/exp | unattributed ms/exp | GPU peak MB | warmup s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | 1 | 1024 | no | 1.99 | 1.68 | 5.56 | 28.90 | 0.22 | 21.18 | 59.53 | 1.47 | 2.54 | 5.74 | 60.04 | 0.51 | 667 | 61.7 |
| fp32 | 16 | 1024 | no | 1.98 | 0.60 | 5.07 | 22.63 | 0.02 | 14.06 | 44.36 | 1.41 | 2.34 | 5.10 | 44.48 | 0.12 | 6200 | 45.9 |
| bf16 | 1 | 1024 | no | 1.96 | 1.93 | 7.12 | 29.09 | 0.23 | 27.18 | 67.50 | 1.74 | 2.69 | 5.98 | 68.29 | 0.78 | 559 | 69.3 |
| bf16 | 16 | 1024 | no | 2.02 | 0.34 | 1.54 | 26.63 | 0.02 | 4.72 | 35.26 | 1.60 | 2.67 | 6.22 | 35.43 | 0.15 | 3770 | 36.6 |

## Parity on one batch of 64 (optimizer stubbed, no clipping; reference fp32 B=1; pass = loss within 1%, gradient norm within 5%, cosine >= 0.99)

| config | total loss | loss rel diff | grad norm | norm ratio | grad rel L2 diff | cosine | ok |
|---|---|---|---|---|---|---|---|
| fp32 B=1 | 5.81134 | 0.00e+00 | 5.4760 | 1.0000 | 0.00e+00 | 1.00000 | ref |
| fp32 B=1 rerun | 5.81134 | 0.00e+00 | 5.4760 | 1.0000 | 6.40e-08 | 1.00000 | yes |
| fp32 B=16 | 5.81134 | 5.17e-08 | 5.4760 | 1.0000 | 1.46e-04 | 1.00000 | yes |
| bf16 B=1 | 5.81206 | 1.24e-04 | 5.5245 | 1.0089 | 3.77e-02 | 0.99933 | yes |
| bf16 B=16 | 5.81329 | 3.36e-04 | 5.4896 | 1.0025 | 3.32e-02 | 0.99945 | yes |

## Implied az_loop iteration (24 games, holdout 0.2, 540 experiences per game, step cap 4000, 1 trial(s), 32 sims)

| precision | B | compiled | train-path s | generation s at 833 leaves/s | fraction | generation s at 489 leaves/s | fraction |
|---|---|---|---|---|---|---|---|
| fp32 | 1 | no | 596.1 | 498 | 0.545 | 848 | 0.413 |
| fp32 | 16 | no | 444.7 | 498 | 0.472 | 848 | 0.344 |
| bf16 | 1 | no | 676.3 | 498 | 0.576 | 848 | 0.444 |
| bf16 | 16 | no | 352.1 | 498 | 0.414 | 848 | 0.293 |

Training-path passes per iteration: 9912 forward+backward experiences (step 4000, held-out 2700 x 2, signal 512) and 648 forward-only.
