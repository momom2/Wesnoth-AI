# Round 9 critique of the simplify stance

**Verdict: repairable.** No guessed number carries an adoption, a gate or a budget line. One guessed transfer carries a prediction, one rejection answers the wrong question, and one removal breaks attribution. New script: `analysis/round9_simplify_critique_clusters.py` (`.out` beside it).

## 1. Provenance audit

I reran all six of the stance's scripts. Each reproduces its `.out` byte for byte.

| Number | Class | Check |
|---|---|---|
| `own2` +22.7 ± 4.4 | M | `own2_base_20261020/reading.json`: 22.72, SE 4.35, 6,400 decisive. Holds |
| S0(a): −41.3 ± 4.4 Elo, −0.242 ± 0.023 z | M | `readout_a_offline.txt`. Holds. The match and the label read the same games, so S0(a) never tested labels against uncoined play |
| Census 77.6; pooled mean +0.0003 ± 0.0007 | M | `readout_b_offline.txt`. Holds |
| Typed-rule P 0.32 | D | Reproduces, under a null of independent normal types |
| V_rr 1.59e-3 ± 0.37e-3; cross-fitted +23.1 ± 3.4; C1's +22.0 ± 4.7 (reverse split +16.7 ± 4.5) | M | Reproduces. It is corroborated independently by `round9_provenance_recruit_cells.out`: a rule fitted on T reads +20.6 ± 3.4 with T0, so the luck columns do not create it; opposite orientations agree (Vampire Bat > Ghost +0.126, Ghost > Vampire Bat −0.115); and the whole-band partition reads z 4.41 (`round9_provenance_cells.out`) |
| E[T²] 4.01, within-game ratio 0.999, residual 0.699, on `own2` 0.692 (SE ≈ 0.04) | M | Reproduces |
| 22,209 / 20,254 / 6,521 games an hour | M | `matches.txt` (8,105 s for 50,000 games), `walls.txt` (about 155 s of processing per match), `match.walls` (3,533 s). Hold |
| Forecast SE 4.88; miss lines 10.4 and 10.1; null line 3.6, P 0.0001; claim SEs; $23.98, $50.38, $74.36; o ≤ 0.117; 24 GB | D | Redone; all hold |
| $82.34 left | D, on an assumption | It assumes the $5 window came out of the $85. Ask the user. Only the third round depends on it |
| Rule shares; SE(v̂) 1.39e-6 | D | Reproduce. They carry only "keep τ̃" (protected by the unbiased welfare curve) and "no pilot" (§2.2) |
| κ̂Ĝ_pf as S2's prediction | **guessed transfer** | §2.1 |
| 8 × 100,000 games a round | C | Its cost is derived; its benefit outside recruits is not, so the stage order should not wait on it (§2.2) |

## 2. What would make it fail

### 2.1 κ̂ is measured where the head will not act

C1 measures transfer on one family (recruits), on `own1`, at 2.28 switches per side-game. S2 applies κ̂ to a head on `own2` making 19–38 switches across all families. My script counts what C1's frozen rule does on b04 (`round9_simplify_critique_clusters.out`):
- 58.3% of its switches follow a rule switch by the same side in an earlier side-turn, and 6.9% follow one in the same side-turn.
- In the label world, only 47% of the cross-turn followers came after an earlier switch that was actually played. Under the rule, every earlier switch is played.
- After a played switch, another rule decision in the same turn occurs 0.56× as often. The switch rate in C1 will therefore sit a few percent below 2.28.
- The labels can identify the interaction, since H_j contains D_i. On b04 it reads −0.005 ± 0.026 per follower switch, unresolved against a mean effect per switch of about 0.028 (22.0 / 347.4 / 2.28).

A C1 shortfall can be a recruit-specific second-order effect (diminishing returns on one unit type), which says nothing about move pairs.

**Repairs:**
- S2 predicts Ĝ_pf ± 10.4, with κ̂ reported beside it, not multiplied in.
- C1's ≤ 3.6 branch reads: I-match and the label path are debugged first; if both are clean, a second-order effect in recruits is a live reading, not proof of a bug.

### 2.2 The pilot rejection answers the wrong question

SE(v̂) ∝ 1/N means a pilot cannot bound v near 0. It can confirm a spread of the recruit family's size. The stance's own main unknown is whether the tokens carry unit identity linearly, and its recruit readout waits until batch 8. Three repairs, cheapest first:
1. **At R1, $0:** regress the 44-value rule's fitted values on the smoke's dumped recruit-pair x. The target is deterministic, so no labels are needed. The held-out R² says whether the head's form can express the only measured signal before $24 is committed.
2. **In S1, $0:** read the recruit readout after batch 2, with a stated action. The cross-fitted table's SE is about 3.4 × √(60,000/100,000) ≈ 2.6 Elo on one held-out batch [D from `recruit_spread`; assumes `own2`'s recruit-band density, which R1 measures]. If the head is short of the table by 2 SE while I-φ is clean, stop and change φ, which saves up to six batches ($17).
3. **Measure from what exists.** Replay batch 1's 60,000 recorded `own1` games through `own1` to dump x, then fit the head on the labels already paid for. That yields v̂, the recruit readout, τ̃ and a sealed forecast before any `own2` coin game. Cost about $1.9 [D: 471 decisions a game and 2,440 replay decisions a second, the inputs of `round7_provenance_budget.py`; that rate is for an earlier network, so time the first 1,000 games]. Replay fidelity is checked by comparing the log-probability block with each decision's logged gap. The stance's §6 says the full round is the cheapest measurement of the other 94%; this replay costs less.

### 2.3 Dropping the per-round claims breaks attribution

Adoption is read only against the incumbent. A final claim below its line (11.2–12.3) then cannot be assigned to a round, and rounds 2–3 (about $48) can compound on a gain that holds only against `own2`. The stance's own response is "intransitivity … reported", which is no action. Keeping one claim per round costs $1.13, about 5% of a round, to keep "a miss points at the stage that failed".

### 2.4 Checks that could pass on wrong code

- **A mis-joined x:** the merged plant injects h of the same mis-joined x, so it recovers h exactly. Only I-φ can catch the mis-join, and only if it compares the dump's log-probability block with the gap the coin record logged for that decision (written by separate code). The stance's "the dump reproduces the logits" must say so.
- **A defect confined to recruits:** with E and the per-type I-V gone, the pooled I-V sees such a defect only at 0.0021. All of the measured signal lives in recruits. Keep the T-against-T0 reading for whichever family carries the gain; the provenance script already computes it.
- **The step plant's predicted fit change,** (G + λΛ)⁻¹ Σ x h(x), must be computed in the processor's single pass, because x is deleted afterwards. The stance does not specify this.

### 2.5 Smaller items

- **Costs:** offline processing (24 GB and a 772-wide Gram matrix per batch) is unknown, yet the budget multiplies o into generation only. Measure it inside o at R1, and require 24 GB of free disk on the host.
- **C1's split:** b00–b03 → b04–b06 reads +22.0, the reverse +16.7, each honest for its own rule. Record that the split was fixed before both were seen; otherwise the prediction is the larger of two noisy readings.
- **Literature:** the stance claims no method, accurately. Gilotte et al. is verified ([ar5iv](https://ar5iv.labs.arxiv.org/html/1801.07030)): 39 A/B tests, best offline–online correlation 0.49, which supports running C1 first. The design's least-squares head on frozen deep features has an uncited precedent, LS-DQN ([Levine et al. 2017](https://arxiv.org/abs/1705.07461)).

## 3. Strongest part

C1 on the recruit signal.
- The signal is real: z 4.3; it survives without the luck columns; opposite orientations agree; it holds out of sample.
- C1 is the design's first test of its central premise, that coin labels predict gains in uncoined play. The prediction sits at z 4.7, the test costs $0.98, and it runs before $24 is spent.
- It would also be the first self-play-derived candidate measured against `own1` with a gain predicted in advance.

Freezing b and β also closes the leakage path, for at most 1.6% more games.
