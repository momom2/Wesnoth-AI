# Network size for a self-play program: comparison and scaling laws (2026-10-14)

Research by a subagent for the lead, on the user's question: should the network
grow before a self-play program of 1 to 10 million games, and to what size?
Read-only study; no run. Figures marked "est." are estimates, not published.

## Answer

- **About 1 million games (about $100):** no. `own1`'s 15.6M parameters lie
  inside every range the published laws give at this compute (about 8-48M), and
  the anchor closest to our arithmetic intensity gives about 13M.
- **The $1,000 class:** probably grow, to about 40-60M (range 24-130M). At that
  size $1,000 buys about 3-5 million games, not 10 million.
- **Size is second-order until a self-play operator improves the reference.**
  Nothing from self-play has yet, and every law below assumes a working
  learning curve.
- **Growing mid-run is established practice** (KataGo grew four times; OpenAI
  Five grew 43.8M to 156.7M by "surgery"), so the size need not be fixed up
  front.

## Our system (from the code and records)

- `own1`: d_model 384, 8 layers, 12 heads, FFN 1536. 15.30M parameters in the
  model plus 0.26M input embeddings = 15.56M. 14.2M of those are in the trunk
  and 0.49M in the memory.
- **Tokens:** 385 trunk tokens per trained position on average
  (`training/metrics/trainer_speed_2_20261014/arm_batched.steps.jsonl`).
- **FLOPs:** forward about 12.8 GFLOP per position, training about 3x that.
  The GEMM time per window implies about 124 TFLOP/s on the 4090, which is
  consistent.
- **Per game:** 464 decisions a game (generation round 3's records). One game
  generated and trained once is about 2.4e13 FLOP, so 1 million games is about
  2.4e19 FLOP and 10 million about 2.4e20.
- **Imitation:** the corpus is 4.04 million positions an epoch, and `own1`'s
  lineage saw about 18 million human positions plus 2 million of its own.

## Successful game-playing systems

| System | Parameters | Training scale | Imitation start |
|---|---|---|---|
| AlphaGo Zero | 20 blocks about 23M (est.); 40 blocks 46.4M | 4.9M games (3 days); 29M games | no |
| AlphaZero | 20x256 ResNet about 23M (est.) | chess 44M, shogi 24M, Go 21M games | no |
| MuZero | 36.9M | 1M steps x 2,048 (board games) | no |
| KataGo 2019 | grown b6c96 -> b20c256 (1.0 -> 23.6M, est.) | 4.2M games, about 28 V100s for 19 days, about 2e20-2e21 FLOP (est.) | no |
| Leela Chess Zero | BT4 transformer 191M; T78 CNN 195M | community-scale | no |
| AlphaStar | 139M (55M policy) | 971k human replays, then a 44-day league | yes |
| OpenAI Five | 159M, grown from 43.8M | 770 PF-days | no |
| DeepNash (Stratego) | not stated | 5.5e9 games | no |
| Ruoss et al. 2024 (chess, no search) | 9M / 136M / 270M | 10M games, 15.3B labels | distillation |
| microRTS RAISocketAI | 5.0M | behaviour cloning then PPO | yes |
| `own1` | 15.6M, 12.8 GFLOP per forward | 18M human + 2M own positions | yes |

A 10-million-game budget for us is roughly KataGo-2019-class compute, and
KataGo's final net costs about 17 GFLOP per forward against our 12.8.

## Scaling laws

| Source | Setting | Finding |
|---|---|---|
| Jones 2021 | Hex, AlphaZero, MLPs | about 500 Elo per decade of compute on the frontier; 10x train compute replaces about 15x test compute |
| Neumann and Gros 2023 | Connect Four, Pentago | strength scales as N^0.88 (about 106 Elo per doubling of parameters at convergence) and C^0.55 (about 66 Elo per doubling of compute); N_opt ∝ C^0.62 |
| Neumann and Gros 2025 | Oware, Checkers | inverse scaling: larger models did worse, in long games whose end-games dominate |
| Hilton, Tang and Schulman 2023 | single-agent RL | N_opt ∝ C^0.40-0.80, with Dota 2 at 0.76 (N = 2.703e7 · C^0.7617, C in PF-days); RL optima are smaller than generative-model optima at equal compute, and longer horizons push them smaller |
| Tuyls et al. 2024 | NetHack | imitation N_opt ∝ C^0.61 (loss); IMPALA RL N_opt ∝ C^0.43 |
| Pearce et al. 2024 | imitation on human gameplay | at 540 tokens per observation, N_opt ∝ C^0.32: many tokens per observation push the optimum toward smaller models and more data, as with our 385 |
| Ruoss et al. 2024 | chess, no search | 9M -> 136M: +234 Elo; 136M -> 270M: +40; models of 7M and up overfit 10k-game sets but not 100k-game ones, and our corpus is 14k games |

Cost of missing the optimum (from the exponents above): 2x off costs about
16% of compute efficiency, 4x off about half.

## Extrapolation

At $0.50 an hour, $100 is about 200 box-hours or 2.4e19 FLOP, and $1,000 is
2.4e20.

| Anchor | $100 | $1,000 |
|---|---|---|
| Hilton (Dota), in parameters | 10M | 59M |
| Tuyls (NetHack RL), in parameters | 48M | 129M |
| Board-game self-play, 40 decisions per parameter (KataGo, AlphaGo Zero 20 blocks) | 13.5M | 43M |
| Board-game self-play, 130 decisions per parameter (AlphaZero) | 7.5M | 24M |

The board-game anchors share our arithmetic intensity (each parameter applied
to about 361 points against our 385 tokens), so they weigh most. Candidate
shapes at about 40-60M: d640x8 (42.5M) or d768x8 (61M), with 32-dimensional
heads; d512x12 (40M) for depth.

## Cost per million games by size

Generation plus one training pass, in box-hours. "Today" keeps the inference
server's host work; "device-bound" assumes it is ported to Rust.

| Architecture | Parameters | Forward GFLOP | Today | Device-bound |
|---|---|---|---|---|
| d384x8 (current) | 15.5M | 12.8 | 194 | 135 |
| d512x8 | 27M | 21.8 | 253-290 | 194-232 |
| d640x8 | 42.5M | 33.4 | 323-412 | 264-354 |
| d768x8 | 61M | 47.3 | 404-560 | 345-501 |

Generation is about 65% host-bound, so it grows only 1.2-1.9x up to 4x the
FLOPs; training scales with size and dominates once the host work is ported.

## A cheap test before a large run (about $4-8)

**Setup.** An imitation size curve at equal data: d256x6 (5.4M), d384x8 (the
15.5M control) and d640x8 (42.5M). Each trains from scratch with the sequence
trainer and the reference recipe for 2 epochs, 8.07M positions. The control
also gets a cooldown branched at 4M positions.

**Measure.** Holdout cross-entropy, then 800-decisive-game matches:
- each arm against the control;
- the control at 4M positions against the control at 8M.

**Decision.** Grow when Elo per doubling of parameters exceeds Elo per doubling
of positions by two standard errors. Do not grow when 42.5M ties the control or
5.4M ties it. Cross-entropy gains without Elo do not count.

**Once self-play improves the reference at 15.6M:** train a grown d640 shadow
on the same self-play games for about 100k games, and switch when it matches
the 15.6M net on fresh-game holdout loss and in a match. This is KataGo's
rule.

## Main uncertainties

1. **Transfer.** The laws come from deterministic perfect-information games
   with search-strengthened targets, or from single-agent RL; raw-play targets
   and Wesnoth's long horizon push the optimum smaller.
2. **Parameters or FLOPs.** Reading the RL anchors in parameters or in forward
   FLOPs moves them by orders of magnitude.
3. **The imitation start and the memory.** The laws assume training from
   scratch, and none covers a recurrent memory. A larger net loses `own1`'s
   lineage or needs growth, which this project measured as not drop-in.
4. **No working self-play operator yet.**
5. **Inverse scaling in long games.**

## Sources

- Jones 2021, https://arxiv.org/abs/2104.03113
- Neumann and Gros 2023, https://arxiv.org/abs/2210.00849; 2025, https://arxiv.org/abs/2412.11979
- Hilton, Tang and Schulman 2023, https://arxiv.org/abs/2301.13442
- Tuyls et al. 2024, https://arxiv.org/abs/2307.09423
- Pearce et al. 2024, https://arxiv.org/abs/2411.04434
- Ruoss et al. 2024, https://arxiv.org/abs/2402.04494
- AlphaZero, https://arxiv.org/abs/1712.01815
- AlphaGo Zero, https://www.nature.com/articles/nature24270
- MuZero, https://arxiv.org/abs/1911.08265
- KataGo, https://arxiv.org/abs/1902.10565
- OpenAI Five, https://arxiv.org/abs/1912.06680
- AlphaStar, https://www.nature.com/articles/s41586-019-1724-z
- DeepNash, https://arxiv.org/abs/2206.15378
- Epoch AI's model database, https://epoch.ai/data/all_ai_models.csv
