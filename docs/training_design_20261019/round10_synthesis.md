# Round 10 synthesis

## Verdict

The design changed in substance. Its core did not: near-tie coins, exact labels, the closed-form head on the frozen network, adoption only by match, per-round claims. By the round's criteria:

1. **A fixed flaw.** L's sealed prediction, R's first-order F, was not a prediction of R in play.
   - On batch 1's own rows, R's second-order term over its own switches reads −12.3 ± 4.5 Elo. That is the provenance critique's TayPO-2 reading, reproduced exactly here (`analysis/round10_synthesis_checks.out`).
   - The expansion is not convergent at rank 2.
   - An L miss could therefore not separate R's own interactions from the coin-to-uncoined link, which is what L exists to test.

   Capped at each side's first two switches a game, a rule's coin-world value is exact and its interaction reads near zero. L now plays a capped rule against a complete forecast.
2. **A measurement changed a decision.** The design's progressive loss ranks four recruit rules on batch 1. Bd, keyed on unit, unit × enemy faction and the side's earlier recruits of each type, wins at −699.3, against −273.2 for R. Bd replaces R as L's rule and as the recruit block under ruling (h). Capped, it is worth +24.1 ± 3.0 Elo against capped R's +14.9 ± 3.0. With it, L's null bounds the transfer ratio below 0.30 instead of 0.48 (`analysis/round10_synthesis_budget.out` §1).
3. **The adoption stage changed.** S2 then E becomes a step search: three screens of different switch-set sizes, each against its sealed forecast, and the best confirmed on fresh seeds.
   - It costs $1.98 against $1.96 for S2 plus the E reserve.
   - It reads the second-order loss as a function of the step, which the record now shows is real on the one structure found.
   - H1 is removed. T and C run only under ruling (h).
4. **Two more fixed flaws.**
   - No check saw the 772 token columns. I-φ2, an exact identity through the linear actor head and the q·k target logit, now does.
   - The smoke's overhead reading was confounded by a fixed cost per match. o is now read from equal-size runs, and the host test runs 10,000 games.

No new core mechanism was added. The step search applies the challenge stance's principle at the adoption stage, repaired so that every switch it plays is labelled.

## What the round's measurements moved

| Measurement | Decision moved |
|---|---|
| Second-order terms: R −12.3 ± 4.5, Rd −7.8 ± 4.8, B −10.3 ± 4.4, Bd −13.1 ± 4.7; capped interactions −0.3 ± 1.7 (R), +0.6 ± 1.8 (Bd) [*checks*] | L plays a capped rule; S2 searches the step; no full rule's F is sealed |
| Progressive loss picks Bd (−699.3) over B (−557.0), Rd (−403.1) and R (−273.2) at ridge 5,000 [*checks*] | L's rule; the recruit block's columns under (h) |
| Fold schemes: R reads +21.3 (SHA-256 20-fold), +19.9 (crc32 20-fold), +18.0 (leave one match out) | Seals use leave-one-match-out values |
| The collapse: turn-1 distinct types 1.345 (`own1`), 1.356 (`own2`), 3.164 (humans); count slope inside identity z 5.6 [invent] | d enters Bd; T's seal J_T = J_L rests on `own2` collapsing as `own1` does |
| R's recruit split rate is not its cause: split-only rule +1.3 ± 2.9 [challenge] | No marginal recruit decode |
| Walls carry 69 ± 30 s a match [provenance] | o from equal-size runs (SE 0.024); a 10,000-game host test |
| The per-candidate unit block adds 2.36 switches worth +0.2 ± 3.3 [simplify critique] | The recruit block is a pair feature on recruit pairs |
| R's τ curve: the best τ beats 0 by +0.8 ± 1.6; F holds at every ridge [simplify] | τ̃ kept as a guard; one ridge; the typed block goes |
| Claim false misses: lower tail alone 0.023–0.048 [simplify critique] | One-sided claim stop |

Section 7's own measurements moved round 9's decisions (base `own2`, prices, transferred windows), and those stand.

## What changed, and why

### From provenance (critique: repairable)

**Kept:**
- **The exact identity.** It is the performance-difference lemma with the coin policy as base, read by per-decision importance weights. It is the basis of the capped seal.
- **The finding** that 0.71 of R's value sits in turn-1 recruits, and 0.96 of those in a side's first or second switch. This is what makes a capped rule worth playing.
- **The equal-size overhead protocol,** and the 10,000-game host test.
- **Per-match dump processing,** 2.40 GB at a time.

**Repaired, from the critique:**
- **The null row.** It claimed own interactions could not produce an L null. They read −12.3 ± 4.5, so the claim is dropped. Capping makes the null row name one cause, the coined continuation, and the lower-density options now target it.
- **The first-coin coefficient** is 0.5, not 0.75: −2.3 ± 2.8.
- **The turn-1 state shift** is measured by exact reweighting (−1.2 ± 1.3) in place of a dose-response model.
- **The seal uses leave-one-match-out rules,** not the higher 20-fold reading.
- **B1's price depends on its host:** $2.59 at $0.42 an hour, $3.56 at $0.577.

**Not kept:**
- **Ruling (i), the futility stop.** It has no trigger without H1, and its power cost cannot be derived.
- **The within-rank contrasts as randomized.** They condition on a post-treatment count.

### From challenge (critique: repairable)

**Kept:**
- **"Labels choose the direction, play chooses the step"** as the adoption rule, and fresh-seed confirmation of a screen winner (false adoption 0.023 whatever was screened).
- **The matchup finding:** B − R +7.19 ± 1.71 on 20 folds, +8.6 ± 2.1 leave one match out.
- **The turn-1 concentration,** and the refutation of a decode artifact.

**Repaired:**
- **The step varies the switch set within the network's top two,** not a class-wide logit shift. Past-band and third-candidate flips are unlabelled (critique §2.2), so they have no forecast.
- **The known answer.** The hook's within-class form changes no decision (§2.1). The capped screen's complete forecast and L now anchor the step search.
- **Pooling** stays by progressive loss (§2.4).

**Not kept:**
- **30,000-game recruit iterations.** Their size was measured on the first step only. They are deferred until per-block learning curves exist.
- **The two-null stop.**
- **The η grid.**
- **The "42–87×".** It is 5.2× for B.

### From invent (critique: repairable)

**Kept:**
- **The diagnosis.** It holds on both networks against the human contrast, with a count slope inside unit identity (z 5.6). It explains part of R's negative interactions: R's own play lowers d at its later switches by 1.39.
- **The count column d.** With it, Bd beats B, and Rd beats R by +5.1 ± 1.7.
- **Composition statistics** as a report on every head seat.

**Not kept:**
- **Herded argmax in the regimen.** It is deferred to ruling (i), for four reasons:
  - Inside the band, HA switches on n0 − n1 > ΔQ, not on d. ΔQ is unmeasured, so "HA acts like the count rule" does not hold (critique §2.1).
  - As specified, its T → 0 limit is not the reference decode (§2.2).
  - W under coins and vetoes is unspecified (§2.3).
  - It is a hand-designed decode, of the class of the −1.5 offset that the user wants replaced by learning.

  If (i) is yes, H runs in box A with the critique's repairs.
- **κ_comp as a gate.** Its target is not derived.

### From simplify (critique: repairable)

**Kept:**
- **R0's differential test,** repaired: the pair feature on recruit pairs, b ≡ 0, readout_b's β and per-fold sums.
- **Removing H1, E as a stage, and the typed block.**
- **One ridge.**
- **The R² report,** extended to Bd's values and d.
- **The one-sided claim stop.**
- **I-φ2,** from the critique.
- **Sandwich variance.**

**Not kept:**
- **Removing T and C unconditionally.** Under (h), T is the cheapest bankable gain on record.
- **Moving the reset and Branch S out of the plan.** The third slot is the user's choice between them and a refit.

## New in this synthesis

`analysis/round10_synthesis_checks.py` (58 s):
- the progressive loss of four recruit rules;
- each rule's leave-one-match-out F;
- the capped exact value, the second-order term by rank, paired differences, and capped R by faction.

R's readings reproduce the provenance stance and its critique to the digit (+18.0, +14.9, −0.3, −12.3).

`analysis/round10_synthesis_budget.py`:
- L's lines at three seals;
- the step search's lines, with a Monte Carlo of selection and adoption;
- the path's cost with and without (h), by o, for both third-slot options.

The Monte Carlo's truths are illustrations; only its 0.023 is exact.

## Open questions

1. **Rulings:**
   - (a) unchanged;
   - (e) now covers the capped rule;
   - (h) revised to Bd's 301 columns, as a capped rule in π_0 and as head columns;
   - (i) new: herded argmax;
   - whether the $5 window counted;
   - after round 2, the reset or a third refit.
2. **Do coin labels price uncoined play at a step with no measured interaction?** L answers this on `own1`, and S2's capped screen on the new base.
3. **How does the second-order loss grow with the switch count?** On recruits it is −8 to −13 Elo at about 2.2 switches; for the head at 19–38 it is unknown. S2's three screens answer this.
4. **Does the frozen token span carry the matchup and the composition linearly?** The smoke's R² reports it, and so do the recruit-block learning curves in S1.
5. **Is there near-tie value beyond recruits on `own2`?** S1 answers this.
6. **Is a composition change outside the band worth playing?** H answers this only if (i) is yes; otherwise the reset does.
7. **Unexplained:**
   - the fold-scheme gap (+3.3 ± 1.2);
   - the attack-pair reading under rule (b) (−8.3 ± 2.7);
   - why only the parameter-free count rule shows a small second-order term (−3.4 ± 4.8).

## Research targets for round 11

- **First-order surrogates against actual improvement as the step grows.** Find empirical studies and theory, such as CPI's α² term or TRPO's KL bound. Look for contradicting cases where large steps transferred. These would predict S2's curve from L and the capped screen.
- **Exact or low-variance evaluation of policies that deviate from the behaviour policy at most k times an episode.** Could control variates or doubly robust forms built from the labels extend the exact reach from k = 2 to 3–4, where the variance now grows as 4^k?
- **Probing late transformer tokens for opponent identity and own-army counts.** This predicts the smoke's R². Look for the counter-case where entity attributes fade by layer 8.
- **Purification against mixing in learned game policies.** [Ganzfried, Sandholm & Waugh 2012](https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2012/papers/3E_4.pdf) found purification often helps. Is there evidence on when realising a learned mixture beats its mode? This bears on ruling (i) and on the reset's target.
- **Measured, not literature:** host throughput by host type (a 10,000-game coin match on the candidate).
- **Carried:**
  - distillation closure on an exchanged-pair target;
  - rating-conditioned networks (Branch S);
  - [Andrews, Kitagawa & McCloskey 2024](https://nber.org/papers/w25456) for sizing the $1,000 case from adopted rounds;
  - twins' ρ, only if round 3 needs it.

## Unknowns that remain

| Unknown | Measured where | What relies on it |
|---|---|---|
| The link at the capped step on `own1` | L ($0.98) | Whether a round is bought |
| Capped Bd on `own2` | T, iff (h) ($0.98) | π_0 |
| Bd's seal with every game's enemy faction | R0 ($0) | L's line |
| `own2`'s gates, census, E[T²], dump timer, R² | The smoke ($0.09) | Gates; a report |
| o; staging minutes; hook overhead; processor time | Batch 1's matches 1–2; box A; the screens | Rounds that fit; the third slot |
| v per type, τ̃, forecasts on `own2` | S1 | The round's gain |
| Second-order loss against the step; the link on the new base | S2's step search | Adoption; round 2's grid |
| A host's cost per coin game | 10,000-game timing | The third slot with the reset |
| Whether the $5 window counted | The user | The third slot |
| ĉ, ρ̂, J_S4 | S4 | The next base |
| Branch S's readings | B1–B3 | Base choice |
| Herded argmax in play | H, iff (i) | The recruit decode |
| Compounding, exploitability, human level | Claims; none; none | Nothing |
