# Critique of round 9's provenance stance

**Verdict: repairable.** The measured recruit-pair gain is real and reproduces. Three problems remain: the link from L to S1 is a guessed relation that carries three decisions; T's outcomes and the pooling rule are misread; and S1's value beyond R can be measured from existing records before $22.73 is spent. One budget ceiling is also wrong. Checks: `analysis/round9_provenance_critique_checks.py`, output beside it.

## 1. Provenance audit

| Number | Tag | Check |
|---|---|---|
| F +22.9 ± 3.2 (fold SE 2.9); T0 +20.6; halves +21.3, +24.8; centred +17.5 | M | Rerunning `round9_provenance_recruit_cells.py` reproduces its output byte for byte; `fold_se.out` agrees; every cell's units belong to its faction (b01). λ = 2,000 is the best of four values (+19.1 to +22.9), so the sealed F carries a small selection. |
| Dispersion +11.2 SD; V_rr 1.59e-3 ± 3.7e-4 | M | Holds. |
| Lower bound 4.7e-5, "18× round 8's bound" | D | The arithmetic holds. The 18× compares a variance of cell means with the variance of the head's span (v), which is a different quantity. |
| σ² 4.05 | D from M | Holds, but the 0.760 share comes from 10,000 `own1` games; `own2`'s share is unknown until the smoke. |
| Match SE 3.07; miss half-width 8.9; SE of J/F 0.19; n_L reference 2.21 | D, M | Hold. |
| P(L detects) 0.96 at ρ 0.5 | D | 0.958 with F known, 0.937 with F's SE (checks §2). |
| Path $83.49 / $60.77; null round 1 $24.42; S1 SE 4.88; S5 line 13.1; line 2.45 | D | Hold. |
| S5's ceiling $0.569/h (0.547, 0.526) | D | **Wrong for the stated policy.** These are uniform-rate ceilings. With S1–S4 at $0.577/h, S5's host must cost at most $0.550/h at h 0, $0.476 at h 0.05 and $0.409 at h 0.10 (checks §1). |
| ρ_S1 = ρ_L, implied by "S1 proposed with ρ = 1", "ρ̂ scales later forecasts" and "no S1" on an L null | none | **Guessed** (§2.1). |
| J_T | U | A prediction is available (§2.2). |

## 2. What would make it fail

**2.1 The link from L to S1 is guessed and carries three decisions.** As written, this is fatal.
- **L and S1 take steps of very different size.** L measures ρ for one rule that switches 2.21 recruit decisions per side-game, mostly in one direction within a game. S1's head switches 19 to 38 decisions, mostly moves.
- **The design's own model makes the extrapolation useless.** It treats second-order loss as quadratic in the step (round 7: half the switches, a quarter of the quadratic loss; the conservative policy iteration bound is also quadratic). Then ρ(n) = 1 − kn, and L's SE of 0.19 at n = 2.21 becomes 1.67 at 19 switches and 3.41 at 38.8. A reading of ρ̂_L = 0.9 maps to +0.14 or to −0.76 (checks §3).
- **The proportional model is an assumption.** Under it, the transfer is assumed, not measured.
- **Switch sizes differ too.** Interactions grow with the product of switch effects. R's are large (about 0.030 z each) and S1's are unknown, so the transfer can fail either way.

*Repair:* L decides R only. S1 keeps its own forecast and miss line, read at S2, with no ρ̂ scaling. A shortfall is answered by a smaller step (E).

**2.2 L's null and T's outcomes are misread.**
- **A null does not show the labels fail.** At SE 3.07 it bounds ρ_R at about 0.54 (2 SE), and it occurs with P 0.41 at ρ 0.3.
- **One joint outcome is contradictory.** If L is null and T positive, the plan says both "no S1" and "π_0 = `own2`+R".
- **T can be predicted.** `own2`'s recruit mix is close to `own1`'s (`own2_recruits.out`). Seal J_T = J_L before either match is read; the difference has SE 4.34. L and T then replicate one link at a pooled SE of 2.17, and their difference reads the change of base. Every cell of the 2×2 then has a stated meaning.
- **T does not test pooling.** That a rule's sign carries over says nothing about whether `own1`'s label magnitudes are unbiased for `own2`. The current design refits on all rounds' labels: stale labels can bias the student, never a claim. Dropping batch 1's labels after a null T loses power and protects nothing.

**2.3 S1's value beyond R can be measured now.** By the stance's own account, nothing outside recruit pairs is certified, and a null round 1 costs $24.42.
- **The measurement.** Batch 1's game records are complete. Replaying them through `own1` gives the frozen token features at every band decision: forward passes on recorded commands, with no generation and no training. The stance's cross-fitted welfare code then reads the 815-column head's out-of-fold gain against the labels already paid for.
- **It is unbiased.** Any function of the pre-coin history keeps the reading unbiased, so the replay's numerics need not match the original play.
- **Cost.** At most about batch 1's $1.75 all-in (derived: the same forward passes without the coin logic; the dump overhead h is unmeasured, as it is for S1). That is about 7% of S1.
- **Limits.** It reads `own1`, not `own2`, which is still a closer proxy than ρ̂ from recruits. Like S1, it needs a box and ruling (e).

**2.4 Smaller points.**
- **λ before sealing F.** Fix it by the design's progressive-loss rule, or seal +19.1 beside +22.9.
- **The attack-pair reading.** Rule (b) with T reads −8.3 ± 2.7 Elo. The stance's explanation, negative typed means, accounts for 0.14 of it: −0.0021 per label against an implied −0.0149 per switch (checks §5). It is not certified over 28 readings, but S1's head will switch attacks, so it should stand as unexplained.
- **The money framing.** The overrun is $1.15 against "about $85". The corrected ceiling matters; the "money is tight" framing does not.

**2.5 Complexity.** R adds a table outside the network on the play path, a hook, a plant and I-R. The measured gain pays for that, and failure costs $0.72. Ruling (h) is a real question, though: R's one-hot unit features were chosen after reading the labels, and they bypass the transformer that the mandate names. The 43 production unit columns raise the same question.

## 3. Novelty

- **Claim 2 is the trust-region ratio:** actual over predicted improvement, used to accept a step and resize the region (Nocedal & Wright 2006, ch. 4, Algorithm 4.1; recalled, since the [fetched PDF](https://pages.mtu.edu/~struther/Courses/OLD/Other/LessOld/5630_11/NocedalAndWright/Chpt4.pdf) did not render). Its response to a low ratio is a smaller step, not a rescaled forecast. [Gilotte et al.](https://arxiv.org/abs/1801.07030) support the stance only in part: they found the classical estimators unsatisfying and proposed variants that matched A/B tests.
- **Claim 1 has a second precedent:** local-randomization regression discontinuity ([Cattaneo, Frandsen & Titiunik 2015](https://www.doi.org/10.1515/JCI-2013-0010)) treats a window around a cutoff as a randomized experiment. The application is new, as stated.

## 4. Strongest part

**The measured F.** It reproduces exactly and survives every check applied: the label without variates (+20.6), honest halves, a between-fold SE (2.9), and centring on the group mean (+17.5). Its +11.2 SD dispersion is far beyond what selection among the roughly 17 partitions examined could produce. L turns it into a sealed first-order forecast, tested against uncoined play for $0.36: the first such test on record.

## 5. Repairs

1. L decides R only; drop the ρ̂ scaling and "no S1".
2. Seal J_T = J_L, with the 2×2 table.
3. Replay batch 1's features before S1 is proposed.
4. Restate S5's ceiling.
5. Fix λ by rule before sealing F.
