### bench_pipeline seam_fast

| component | ms (median) |
|---|---|
| deepcopy | 0.048 |
| fork | 0.048 |
| encode_raw | 1.383 |
| encode_from_raw | 1.215 |
| legality_masks | 0.769 |
| enumerate_priors | 2.523 |
| sim_step | 0.643 |
| state_key | 0.019 |
| pack_masks | 1.139 |
| unpack_compact | 0.493 |

| tokens (max) | batch | ms/sample | samples/s |
|---|---|---|---|
| 714 | 1 | 5.41 | 185 |
| 714 | 4 | 1.67 | 599 |
| 714 | 16 | 0.62 | 1612 |
| 714 | 64 | 0.58 | 1733 |
| 831 | 1 | 5.40 | 185 |
| 831 | 4 | 1.68 | 595 |
| 831 | 16 | 0.70 | 1429 |
| 831 | 64 | 0.69 | 1444 |
| 1018 | 1 | 5.36 | 186 |
| 1018 | 4 | 1.66 | 604 |
| 1018 | 16 | 0.85 | 1173 |
| 1018 | 64 | 0.84 | 1185 |
| 2200 | 1 | 5.32 | 188 |
| 2200 | 4 | 2.89 | 346 |
| 2200 | 16 | 2.09 | 478 |
| 2200 | 64 | 2.18 | 459 |

| protocol | batch | ms/batch | leaves/s | wire bytes/leaf |
|---|---|---|---|---|
| logits | 16 | 45.5 | 352 | 86738 |
| logits | 64 | 152.4 | 420 | 96300 |
| priors | 16 | 40.7 | 393 | 15022 |
| priors | 64 | 145.0 | 441 | 14613 |
