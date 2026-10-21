# Round 9 synthesis

## Verdict

The design changed in substance this round, though not at its core. The coin label, the closed-form head on the frozen network and adoption only by match are round 5's. No core mechanism was added. What changed, by the criteria the round is judged on:

1. **Measurements replaced unknowns, and decisions moved with them.**
   - **The base is `own2`** (S0(b), +22.7 ± 4.4 Elo).
   - **The first near-tie spread on record:** recruit choices between two different units. A 43-value rule R collects it at +21.3 ± 3.1 Elo first order, out of sample, with the ridge chosen by the design's progressive loss (`analysis/round9_synthesis_checks.out` §1).
   - **Measured prices put round 3 outside the budget** on the measured host: $4.53 short at zero dump overhead (§6).
2. **Stage order changed.** Before any round is bought, box A ($2.18) tests the link from labels to uncoined play on R:
   - L plays R on `own1`;
   - T plays R on `own2`.

   After S1's first batch, H1 ($0.98) tests the link at the head's own switch count.

   Round 8's design bought a $15–24 round first and found out at S2 whether labels price play.
3. **A flaw fixed.** Every stance predicted the final claim as 22.7 plus the adoption readings, and so did the current design. Those readings are selected: three near-floor steps bias the prediction by +5.85 Elo and push the miss rate to 0.126 against 0.046 (§5). Per-round claims, each checked against its predecessor, restore 0.042–0.053.
4. **Simplified**, taking simplify's architecture:
   - the baselines are fitted once and frozen;
   - the server dumps features and an offline processor computes everything else;
   - pooling across rounds is chosen by the progressive loss rather than assumed.

Against the user's standard: L's prediction is sealed from 60,000 measured games and every outcome of L has an action; H1's reading against T is sharp (SE 2.53). The unknown that remains is the value of the band beyond recruits, and nothing before H1 relies on it.

## What section 7 moved

| Measurement | Decision moved |
|---|---|
| S0(b): `own2` +22.7 ± 4.4 | π_0 is `own2`; batch 1's `own1` labels serve R, the frozen baselines and L. Claims stay against `own1` (two networks) |
| S0(a): −41.3 ± 4.4 Elo against −76 ± 17 from decode-knob records; the label's 2Δ −0.242 ± 0.023 z, the match mean half of it | The match path and the label orientation are right. Windows transferred from other networks become reports; the dice control's window stays a gate, its between-pair spread being measured |
| Batch 1's pooled mean +0.0003 ± 0.0007; 16 typed means, none past the family-wise line | No stage keyed on a pair type's mean; coin play costs +2.0 ± 4.7 Elo |
| Batch 1's records, read by this round's stances | R, L and T; outside recruits nothing is certified |
| Prices: coin game all-in $2.849e-5, 35% above round 8's model | Round 3 does not fit on the measured host; hosts are chosen by measured cost per game |
| Derived SEs held (pooled SE 0.00070 against 0.00070; census 77.6 against 76.1); the plant SE ran about 15% wider | The derivation chain is trusted; plant lines are recomputed from logs |

## What changed, and why

### From provenance (critique: repairable)

**Kept:**
- **The recruit finding and its checks:**
  - excess dispersion +11.2 SD;
  - the label without variates;
  - honest halves;
  - the between-fold SE;
  - centring on the group mean (+17.5 ± 3.2).

  The critique reproduced it byte for byte.
- **L as the first box.**
- **The R plant and I-R.**
- **The corrected S5 host ceiling.**
- **The finding that knob windows from other networks cannot gate.**

**Repaired:**
- **λ.** It was chosen among four values after seeing welfare (+19.1 to +22.9). This synthesis chose it by the progressive loss over the seven matches in game order: 5,000 labels, F = +21.3 ± 3.1, 2.19 switches per side-game. L's prediction is sealed at that value.
- **L decides R only.** L's ratio does not scale S1's forecasts. The critique's quadratic model maps L's SE of 0.19 to 1.67–3.41 at S1's step.
- **T** is sealed at J_T = J_L. It is a test of R across bases, and not a test of pooling, which is now decided by the progressive loss.
- **L's size: 37,800 games, not 12,800.** At 12,800 a null bounds the transfer ratio only below 0.58, which still leaves a round worth buying. At 37,800 the bound is 0.34, for $0.62 more (§2).
- **The contradictory joint outcome (L null, T positive) is resolved:** a null L buys no round, whatever T reads.

**Not kept:**
- **The 43 unit columns as an unconditional part of φ.** They wait on ruling (h). Without (h), the smoke's R² of R's values on the dumped x, and batch 1's recruit readout, report whether the token span carries R.
- **The replay of batch 1** that the critique proposed (see simplify).

### From simplify (critique: repairable)

**Kept:**
- **Frozen b and β.** They close the leakage path for at most 1.6% more games.
- **The architecture:** the dump plus an offline processor, the merged feature plant, and the host chosen by cost per game.
- **C1's idea,** now L.

**Repaired, from the critique:**
- **No κ̂ multiplies S2's prediction.**
- **L's null branch** debugs first and allows a recruit-specific second-order reading. R's switches cluster: 0.58 follow an earlier switch by the same side (`round9_simplify_critique_clusters.out`).
- **Per-round claims are restored.** Without them, a final miss cannot be assigned to a round.
- **I-φ** compares the dump's log-probability block with the coin record's logged gap, by separate code. That is the only check that sees a mis-joined x.
- **I-V per type is restored.**
- **The step plant's predicted fit change** is computed in the processor's single pass, and the box needs 24 GB free.

**Not kept:**
- **Removing E.** E is now also the response to a failed H1.
- **Replaying batch 1's 60,000 games through `own1` to dump features before S1,** which both critiques proposed. Three reasons:
  - **Its cost is unmeasured** between $1.56, at generation's rate, and $10.77, at the measured 396 positions a second of the existing replay path, `tools/own_targets.py` (§4).
  - **It cannot change the decision to buy S1:** a head's welfare at 60,000 games has SE 12.6 Elo, and the spread estimator's SE is 13 times S1's.
  - **H1 measures the at-scale link more cheaply.** It plays the first student after S1's first batch, which S1 buys anyway, for $0.98 with SE 2.53, using S1's own code. A match carries 78 times the labels' information per game about a fixed rule's value (`round9_simplify_budget.out`), so once a candidate exists, playing it is the cheap evaluator.

### From challenge (critique: repairable)

**Not kept as the core.** Three reasons:
- **The factorial needs proposals,** and the record has none beyond recruits.
- **Coins on both seats halve a switch block's effect.**
- **Column 1 aliases the side.**

This synthesis adds a fourth: for per-faction attribution, a stratified one-rule match (L) reads each faction's contribution at SE 0.73 Elo, against a Hadamard block's 1.79 raw or 1.49 with variates at the same 37,800 games (§2). Within one seat at most one faction block fires, so the factorial's extra columns buy nothing here.

**Kept:**
- **Fixed multi-decision rules are cheaper to read in play than by welfare** (the uncoined ratio 2.87 n). That is H1's rationale.
- **Per-faction readings,** inside L.
- **I-side.**
- **The fire-count gate** in I-R.
- **The winner's-curse finding,** which drives the per-round checks.

**Dropped:**
- **The "coherence" number,** −0.0126 z per changed decision. The hooked seat's lost moves reappear on the opponent.
- **The offset block.**
- **SPSA-style steps.**

### From invent (critique: repairable)

**Deferred.** The proof that a sibling-centred label stays exact is sound, and the pilot is shaped right. But the twins only cut cost. Rounds 1 and 2 fit without them (up to o 0.56). Round 3's $4.53 gap needs r ≤ 0.90, about $4.79 saved over three rounds (`round9_invent_critique_checks.out` §3). The pilot certifies that with P 0.80 at ρ 0.2 and 0.04 at ρ 0.15 (`round9_invent_twin_power.out` §2). The record leans toward low ρ: Q8, and a first disagreement at a turn-1 recruit in 0.667 of pairs. A host 7% cheaper per coin game closes the gap with no simulator change.

If twins are ever run, the critique's repairs apply:
- plants into every twin's z, since a self-included sibling mean passes the stated plants and reads as a twin gain;
- ρ̂ from pairwise covariances;
- I2 on the twin label;
- sizing on the seed-clustered variance of the head's statistics, since r̂ is not the head's noise ratio;
- ordinal dice keys gated off the match and search paths.

**Record corrections:**
- r ≥ 0.9975 at K 4;
- the novelty claims (ii) RLOO and (iv) do not hold;
- the budget counted L+T where the regimen ran one arm.

### New in this synthesis

`analysis/round9_synthesis_checks.py`, output beside it:
1. **R's ridge by the progressive loss:** 5,000 labels; F +21.3 ± 3.1 Elo, with per-faction contributions.
2. **L's lines at both sizes;** the stratified per-faction SE.
3. **H1's SEs:** 2.53 against T, P 0.023 for a false fire, and the first student's forecast SE of 6.8–9.8.
4. **The replay's cost range** and what it could resolve.
5. **Selection in claims:**
   - the cumulative prediction's bias (+5.85, miss rate 0.126);
   - the conditional MLE's failure near the floor (sd 7.6, bias −3.6), checked against [Zhong & Prentice 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2536726/);
   - per-round checks at 0.042–0.053.
6. **The budget at measured prices:**
   - two rounds fit iff o ≤ 0.56;
   - the whole path is $4.53 short at o = 0;
   - it fits iff a host's all-in cost per coin game is ≤ $2.66e-5, or ≤ $2.77e-5 if the $5 window did not count;
   - a null round 1 after box A costs $27.22.

## Open questions

1. **Rulings:**
   - (a) unchanged;
   - (b) unchanged;
   - (c) revised to per-round claims;
   - (e) now covers R;
   - (h) new: may R stay in the player, and may the head carry unit columns;
   - whether the $5 window counts against the $85.
2. **Do first-order coin labels price uncoined play?**
   - at R's step: L;
   - across bases: T;
   - at the head's step: H1 and S2.
3. **Does the frozen token span carry unit identity linearly?** The smoke's R² and batch 1's recruit readout answer this. If (h) is no and the span does not carry it, the recruit structure reaches the player only through the reset.
4. **Is there near-tie value beyond recruits?** H1 gives a first reading, S1's v̂ a shape-free one.
5. **The attack-pair reading** (−8.3 ± 2.7, rule (b)) is unexplained, and the head will switch attack pairs. It is reported per batch.
6. **`own2`'s dump overhead o,** and a host whose cost per coin game brings round 3 into reach.
7. **Distillation closure** on an exchanged-pair target (S4). Branch S's questions are as in round 8.

## Research targets for round 10

- **Selection-adjusted estimates for chained adoptions.** The truncated-normal conditional MLE fails near the floor. [Andrews, Kitagawa & McCloskey, "Inference on Winners"](https://doi.org/10.1093/qje/qjad043) (QJE 2024; recalled, not read this round) gives median-unbiased estimators and hybrid intervals. They could size the $1,000 case from adopted rounds without the winner's curse. [Lee & Shen 2018](https://dblp.dagstuhl.de/rec/conf/kdd/LeeS18.html) treats the same sum problem in online experiments; read the full text, since only its abstract was seen.
- **Second-order loss as a function of the step,** now that L (2.2 switches) and H1 (the head's count) give two points. Is there a bound or estimator in measurable quantities whose two-point fit would predict S2? Candidates: the conservative-policy-iteration bound ([Kakade & Langford 2002](https://dl.acm.org/doi/10.5555/645531.656005)), or the trust-region ratio's step-size updates. Look for contradicting evidence: cases where small-step transfer did not predict large-step transfer.
- **Linear decodability of input-entity identity from late transformer layers.** It predicts the smoke's R² before it is read. Probing literature on residual-stream preservation of input token identity; look for the counter-case where identity is overwritten by layer 8.
- **Throughput by host type for CPU-bound simulation with a GPU server** (the EPYC 7702 against cheaper hosts). Measured, not literature: the first 2,000 games on a candidate.
- **Carried:**
  - distilling a rule-modified argmax target (closure when 5–15% of decisions change);
  - rating-conditioned networks' top setting (Branch S);
  - coupled rollouts after divergence with endogenous chance (twins' ρ), only if round 3 needs them.

## Unknowns that remain

| Unknown | Measured where | What relies on it |
|---|---|---|
| The link at R's step on `own1`; R on `own2` | L, T (box A, $2.18) | Whether a round is bought; π_0 |
| `own2`'s census, E[T²], o, processor time; R² of R on x | S1's smoke | Rounds that fit; how unit identity enters |
| The head in play at one batch; the recruit readout on `own2` | H1, batch 1 | The step form of S1's later students |
| Spread v per type, τ̃, q, Ĝ_pf on `own2` | S1 | The round's gain; the reading of a null |
| The link at the adopted step | S2 against Ĝ_pf, with L and H1 beside it | Adoption; E |
| Whether the $5 window counts | The user | Round 3 |
| A host's cost per coin game below $2.66e-5 | A 2,000-game timing on the candidate | Round 3 |
| ĉ, ρ̂, J_S4 | S4 | Round 3's generator |
| Twins' ρ | T0, only if round 3 needs it | Batch sizes in round 3 |
| Branch S's s use, transfer, M2 | B1–B3 | The base for later rounds |
| Compounding, exploitability, human level | Each claim; none; none | Nothing |
