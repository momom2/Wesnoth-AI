### bench_pipeline fwdbatch_bf16

| component | ms (median) |
|---|---|
| deepcopy | 0.048 |
| fork | 0.048 |
| encode_raw | 1.365 |
| encode_from_raw | 1.222 |
| legality_masks | 0.767 |
| enumerate_priors | 2.519 |
| sim_step | 0.643 |
| state_key | 0.019 |
| pack_masks | 1.125 |
| unpack_compact | 0.495 |

| tokens (max) | batch | ms/sample | samples/s |
|---|---|---|---|
| 714 | 1 | 5.43 | 184 |
| 714 | 4 | 1.67 | 597 |
| 714 | 16 | 0.63 | 1584 |
| 714 | 64 | 0.58 | 1730 |
| 831 | 1 | 5.42 | 185 |
| 831 | 4 | 1.68 | 594 |
| 831 | 16 | 0.70 | 1428 |
| 831 | 64 | 0.69 | 1443 |
| 1018 | 1 | 5.38 | 186 |
| 1018 | 4 | 1.66 | 601 |
| 1018 | 16 | 0.86 | 1170 |
| 1018 | 64 | 0.84 | 1185 |
| 2200 | 1 | 5.34 | 187 |
| 2200 | 4 | 2.89 | 346 |
| 2200 | 16 | 2.09 | 478 |
| 2200 | 64 | 2.17 | 460 |

| protocol | batch | ms/batch | leaves/s | wire bytes/leaf |
|---|---|---|---|---|
| logits | 16 | 38.0 | 421 | 86738 |
| logits | 64 | 152.1 | 421 | 96300 |
| priors | 16 | 50.1 | 320 | 15022 |
| priors | 64 | 235.2 | 272 | 14613 |
