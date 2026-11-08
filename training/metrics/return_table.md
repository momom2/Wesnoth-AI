# The window's results against the reference (`own2` at argmax, end_turn offset -1.5)

## Argmax: both sides raw:t0+eo-1.5

| Match | Policy | Elo | Decisive | Seed base | Served | Ended |
|---|---|---|---|---|---|---|
| final_match | round 3 | +31.8 ± 3.1 | 12800 of 12800 | 900,000 | fp16 | 2026-11-03 16:47:26 |
| final_match_after_stop | round 16 | +94.1 ± 3.2 | 12800 of 12800 | 900,000 | fp16 | 2026-11-08 01:17:59 |

The final matches share seed base 900,000, so they pair game for game; the confirmations use fresh seeds from 980,000.

## T=1 against the reference at argmax

| Match | Policy | Elo | Decisive | Seed base | Served | Ended |
|---|---|---|---|---|---|---|
| tail_match | round 3 | +134.1 ± 3.3 | 12800 of 12800 | 950,000 | fp16 | 2026-11-03 19:28:07 |

| Claim | Policy | Elo | Decisive |
|---|---|---|---|
| D0 | `own2` at T=1 | -415.2 ± 5.5 | 12800 |
| round 3's claim | round 3 at T=1 | -296.8 ± 4.3 | 12801 |
| round 6's claim | round 6 at T=1 | -204.2 ± 3.6 | 12800 |
| round 9's claim | round 9 at T=1 | -146.5 ± 3.3 | 12801 |
| round 12's claim | round 12 at T=1 | -100.8 ± 3.2 | 12800 |
| round 15's claim | round 15 at T=1 | -43.4 ± 3.1 | 12800 |

## The rounds

| Round | Adopted | Along | Reading | First logged |
|---|---|---|---|---|
| 1 | True |  | the cross-play match reads 0.1612 (SE 0.00987) | 2026-11-02 12:32:42 |
| 2 | True |  | half B reads 0.1378 (SE 0.0174) | 2026-11-02 22:55:17 |
| 3 | True |  | half B reads 0.1087 (SE 0.0117) | 2026-11-03 10:19:11 |
| 4 | True |  | half B reads 0.08956 (SE 0.0193) | 2026-11-04 02:38:41 |
| 5 | True |  | half B reads 0.08753 (SE 0.0186) | 2026-11-04 10:02:30 |
| 6 | True |  | half B reads 0.05949 (SE 0.0147) | 2026-11-04 17:41:23 |
| 7 | True |  | half B reads 0.06221 (SE 0.0217) | 2026-11-05 02:42:24 |
| 8 | True |  | half B reads 0.0491 (SE 0.0223) | 2026-11-05 08:47:17 |
| 9 | True |  | half B reads 0.09949 (SE 0.0232) | 2026-11-05 14:43:34 |
| 10 | True |  | half B reads 0.04076 (SE 0.0239) | 2026-11-06 00:03:25 |
| 11 | True | plain | half B reads 0.0455 (SE 0.0219) | 2026-11-06 06:21:34 |
| 12 | True | plain | half B reads 0.008686 (SE 0.0213) | 2026-11-06 13:32:02 |
| 13 | True | average | half B reads 0.07487 (SE 0.0195) | 2026-11-06 23:29:43 |
| 14 | True | average | half B reads 0.0351 (SE 0.024) | 2026-11-07 05:37:18 |
| 15 | True | average | half B reads 0.07286 (SE 0.0221) | 2026-11-07 11:48:47 |
| 16 | True | average | half B reads 0.02952 (SE 0.0199) | 2026-11-07 21:29:39 |

