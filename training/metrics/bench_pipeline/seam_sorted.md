### bench_pipeline seam_sorted

| component | ms (median) |
|---|---|
| deepcopy | 0.048 |
| fork | 0.048 |
| encode_raw | 1.346 |
| encode_from_raw | 1.208 |
| legality_masks | 0.771 |
| enumerate_priors | 2.503 |
| sim_step | 0.643 |
| state_key | 0.019 |
| pack_masks | 1.137 |
| unpack_compact | 0.497 |

| tokens (max) | batch | ms/sample | samples/s |
|---|---|---|---|
| 714 | 1 | 5.39 | 186 |
| 714 | 4 | 1.66 | 602 |
| 714 | 16 | 0.65 | 1548 |
| 714 | 64 | 0.59 | 1709 |
| 831 | 1 | 5.37 | 186 |
| 831 | 4 | 1.67 | 599 |
| 831 | 16 | 0.70 | 1427 |
| 831 | 64 | 0.69 | 1445 |
| 1018 | 1 | 5.33 | 188 |
| 1018 | 4 | 1.65 | 607 |
| 1018 | 16 | 0.85 | 1172 |
| 1018 | 64 | 0.84 | 1185 |
| 2200 | 1 | 5.30 | 189 |
| 2200 | 4 | 2.89 | 346 |
| 2200 | 16 | 2.10 | 477 |
| 2200 | 64 | 2.18 | 458 |

| protocol | batch | tokens (mean) | ms/batch | leaves/s | wire bytes/leaf |
|---|---|---|---|---|---|
| logits | 16 | 663 | 16.4 | 976 | 59598 |
| logits | 64 | 827 | 87.4 | 733 | 76087 |
| priors | 16 | 663 | 14.8 | 1080 | 8991 |
| priors | 64 | 827 | 66.3 | 966 | 14677 |
