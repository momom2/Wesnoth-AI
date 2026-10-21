# Critique of round 10 challenge: labels choose the direction, play chooses the step

**Verdict: repairable.** I reproduced the measured core. Four defects must be fixed before box A″: the known-answer gate contradicts the policy it gates, "exact in-band gain" is not exact for the shift, the null branch contradicts the adoption rule, and the iteration size is carried outside the domain where it was measured. My checks are in `analysis/round10_challenge_critique_checks.py` (*checks*, output beside it).

## 1. Provenance audit

| Number | Class | Check |
|---|---|---|
| Base +22.7 ± 4.4; coin validity | M | Seed §7. Holds. |
| A +21.3, B +28.5 (± 3.1); B − A +7.19 ± 1.71; ridge by loss | M | Rerun: identical (*checks* §1); loss table read. |
| Turn 1 carries +12.5 of R; census 22.89 / 8.89 / 4.29 / 0.983; decode test | M | Reproduced or read; the decode logic holds. |
| Learning curve 0.65 to 1.00 | M | Reproduced. Paired SE, which the stance omits: 30,000 against 54,000 games differ by +2.47 ± 1.25 Elo (*checks* §3). Still rising, so "saturated" is wrong. Measures the first step only (§2.4). |
| W_in(η) | M as computed | Wrong meaning at finite η (§2.2). |
| Prices; SE 347.4/√N; lines 7.2, 3.57, 7.1, 4.34, 8.7; stage costs | M; D | Redone. Null costs are overstated: a null skips the claim ($3.29 a box, $2.50 an iteration). |
| "42–87×" | D, misapplied | It divides by the all-class head's forecast SE. For B (SE 3.1, $1.71) it is **5.2×** (*checks* §4). Carries no decision. |
| Hook window [−52.0, −30.6] | M, D | Arithmetic right, path wrong (§2.1). |
| Line-search Monte Carlo | D | The 0.023 is exact (fresh confirmation); the arms are illustrations. |
| 30,000 batches, two-null stop, η {8, 40}, 100,000 for phase H | C | No consequence derived after the first step (§2.4). |

No guessed number carries a decision outright. Two measured numbers are used outside their domain: the learning curve and W_in(η).

## 2. What would make it fail

**2.1 The known answer cannot pass on a correct implementation.** In §1.1 the network decides the action kind, and the shift re-chooses only among candidates of that class. A shift f = 1[attack] is constant within the attack class, so it changes no decision. It reads 0, which the gate classes as "an unapplied shift", and then "no screen is read". S0(a)'s hook (`shift_attack_prior`, `tools/raw_player.py` on `exp/near-tie-coins`) multiplies every attack's joint prior by e against moves, recruits and end_turn. It changes the kind. To pass, the gate would have to go through a joint-shift path, which is code S0(a) already certified, not the within-class re-choice that the recruit step uses.

*Repair:* restate the hook as a joint shift that certifies the seat coin and where the shift is applied. Certify the re-choice by I-shift's separate recomputation, which is deterministic and the right test for it. Drop the claim that the hook tests the recruit path.

**2.2 W_in(η) is not "exact at any η".** The shift takes the argmax over every recruit candidate (unit × hex). The curve flips only a0 against a1, when η(v₁ − v₀) > G. At η 8 (0.36 nats per sd of p) and η 40 (1.8 nats), a third unit overtakes a0 in three places:
- at different-unit band pairs;
- at the 2.10 same-unit band pairs;
- at the 2.50 band decisions per side-game whose runner-up is not a recruit (8.89 − 4.29 − 2.10).

The coin records log only a0, a1 and the gap (read in `games_b00`), so these flips are unlabelled. The "sealed in-band parts" +24.3 and +28.4 are not a quantity any match reads, and J(40) − J(band-only) mixes past-band with third-candidate flips.

*Repair:* call these "unlabelled flips", have I-shift count third-candidate flips separately, or restrict the shift to the top two.

**2.3 The null branch contradicts the adoption rule.** The own2 screens and their confirmation run in box A″ whatever own1's ρ reads. If ρ is null and the own2 confirmation clears its bound, the table says both "no coin batch is bought" and "adopt". Under the stance's own core, play decides, so the null branch must be conditioned on the own2 result.

**2.4 Iteration size is carried outside its domain.** The 0.91 at 30,000 games was measured on own1's first-step structure, about +28 Elo. Every phase-R batch instead fits the residual left on π_k after an adopted step, and that residual's size is unknown. A null at 30,000 games then cannot separate "no residual" from "too small to resolve at 30,000". The two-null stop ends recruits on that uninformative outcome.

The stance also drops, without a reason, the current design's pooling by progressive loss, which is exact and free. Phase H's 100,000 games have no derived resolution either; the record's one reading outside recruits (moves of different units, +19.0 ± 8.0 at 60,000 games) could size it.

*Repair:* pool by progressive loss. Stop on a bound: the pooled residual welfare's upper 2-SE bound below the confirmation's 3.58-Elo resolution. Do not stop on a count of null iterations.

**2.5 The first claim is in-distribution.** B's labels come from own1 self-play: own1's continuation and own1's recruit mix priced them. Two statements are false for step 1:
- "claims are against own1, which the shifts were not fitted on";
- "its labels always price the current policy" (π_0 is own2).

Counter-recruiting tuned to one opponent's army is the gain most likely to be opponent-specific (the CherryPi evidence the stance itself cites). The per-round check of D₁ against its adoption reading, at a line of 11.2, is the only guard.

**2.6 The cheap iteration depends on a ruling.** The $3.63 recruit iteration exists only if ruling (h) is extended to 257 matchup keys in the player. Without that, phase R becomes the head: a dump with an unmeasured overhead o, and an unmeasured recruit readout. It then costs what phase H costs. The headline "$3.63–$7.62 against $26" is conditional on (h).

Fog costs B little: 96 of 188 human ladder sides pick Random, hiding the enemy faction until first sighting (observation parity, 2026-09-26), but B's gain over A sits mostly after turn 2 (turn 1 +2.06 ± 0.79, turn 2 +1.21 ± 0.55, later +3.92; *checks* §2). The harness never makes a Random pick, so I-key's fallback never runs.

**2.7 Overclaims.** "A certified natural-gradient step on the true objective": the direction is the coined world's gradient at P = 0.25, the step moves an argmax player, and the certificate is a gain over π_k only. If band-only wins, there is no η, so "η̂, 2η̂" and the reset's "logits + ηf" are undefined.

## 3. Novelty against the literature

- **Line search on a measured return:** CPI+ ([Scherrer 2014](https://ar5iv.labs.arxiv.org/html/1405.2878)) line-searches the step toward the greedy policy on a 2^i α grid (its selection criterion is not stated); [CLOP](https://www.remi-coulom.fr/CLOP/) (Coulom 2011) and SPSA tune engines by match outcomes. The assembly with an exact label direction may be new; the parts are not.
- **Logit shift by η × advantage:** the softmax natural-gradient update, as the stance says.
- **Reset target "logits + ηf":** Gumbel MuZero's improved policy, softmax(logits + σ(q)) ([Danihelka et al. 2022](https://mlanthology.org/iclr/2022/danihelka2022iclr-policy)), and MPO's. Claim 4 does not hold.
- **The network findings** (matchup dependence, turn 1, not a decode artifact) are new and reproduce.

## 4. Strongest part

Labels propose and a fresh match decides: the step's size, its second-order effects (R's switches cluster) and its reach past the band are read on the true objective, false adoption stays 0.023 whatever the screens selected, and a null iteration costs $2.50 against $27.22 for the current design's null round. The matchup finding holds: +7.19 ± 1.71 paired, ridge chosen by loss.
