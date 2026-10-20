# Critique of round 7, invent stance (antithetic chance siblings, known-switch calibration)

**Verdict: repairable.** Antithetic chance siblings (ACS) are exact by construction and admitted only on a measured reading, but two spec gaps could bias them. The known-switch calibration (KSC) has readings that can fire together, decides nothing, and spends the design's overhead allowance. Both of the stance's scripts rerun to identical output. My checks are in `analysis/round7_invent_critique_checks.py` (`.out` beside it).

## 1. Provenance audit

| Number | Class | Check |
|---|---|---|
| 109.7 attacks, 449.1 decisions, 11.86 support | M | Confirmed, `round5_synthesis_outcomes.out`. The support comes from a non-random subsample. |
| 0.688; 0.152; 0.042 | M | Confirmed, `round6_provenance_cells.out`, `round6_invent_conditioning.out`. |
| 0.602 batch line; $9.81; $2.92e-5 | D | Confirmed, `round6_synthesis_power.out` §5–6. Misused (§2.3). |
| m = 1 + f·109.7/449.1 | D | Assumes cost scales with forwards; unmeasured. It carries the admission line r·m ≤ 0.602. |
| κ 0.735; shares 0.644 / 0.839; lines 0.099–0.278 | D | Reproduced. They rest on a hand fight model and a hand price. They carry "E_full < 0.099 closes child prices". |
| 0.511; $0.099; $0.050; fit penalty 0.001 | D | Reproduced. The penalty is 0.010 at own1's 1,600 games, which is harmless. |
| \|S₁\| 4.8–31.8 | M (bounds) | Reproduced. The upper bound fails (§2.2). |
| H's cost; cap SE ≤ 11.5 | D | Reproduced. The SE holds only if \|S₁\| ≤ 31.8. |
| −78 ± 16 Elo | M via D | Confirmed: a linear fit on unselected knob games, offsets inside 0–1.5. |
| M₁ SE 1.79; window 36–39; $3.59 | D | Confirmed. |

No outright guess carries a decision. The faults are modelled numbers used where S0 measures the real ones, and two accounting errors.

## 2. What would make it fail

### 2.1 ACS

**The math holds.**
- L_k is h's odd part under the reflection Φ.
- Φ preserves the law of an i.i.d. draw stream whatever the number of draws used, so E[L_k | past] = 0 for any h. Orthogonality to every coin innovation follows by the tower property.
- Fog, function approximation and self-play cannot bias it.
- I redid the capture. When z loads on h, ACS removes Var(h)(1 − ρ)/2. Otherwise it removes only the projection of z's odd part onto h's odd part. S0's r_ACS is read against z, so admission is safe.

**Where it can fail: both siblings must be priced by one function.**
- **Terminal children.** Nearly every decisive game ends with a leader killed in a fight. That fight's realized child has no next forward, and the memo leaves h undefined there. A ±1 price on one sibling and a forward on the other biases the largest fight of nearly every game.
- **Memory.** The reflected side request must read the memory written at the attack decision. I-A3 checks only the realized child.
- **Why I-A4 misses it.** Its mean over about 2 M fights passes when the mismatch is confined to about one fight per game.
- **Repair.** Run the path with Φ = identity, which must give L_k = 0 to numerics in every fight, terminal ones included.

**Order and lines.**
- S0(c) times the side request "before any generation code", which is impossible.
- Admission must be confirmed on the measured m.
- Close the class on the measured r_ACS·m, not on the modelled 0.099.
- The selection proxy, a hand price, belongs on ruling (b)'s list.

### 2.2 KSC

1. **H's readings are not exclusive.** One outcome can trigger both "outside −78 ± 36: label path broken" and "beyond the M₁ line with invariants passing: interactions". M₁, at ±1.79, makes the records line redundant. A single t-point cannot attribute its own miss.
2. **H names no failure that P-coin, I1, I2, I5, I9 and I-A4 miss.** With a logit-linear response, the midpoint gap at −78 Elo is 0.31 Elo (part 4). H mostly reads whether J is non-smooth in t, on a set the student never uses.
3. **T's inference fails.**
   - S_T is hand-chosen attack promotions with negative effect; the student's set is selected for positive effect.
   - F_D carries the background and opponent coin terms the memo cites against the trapezoid.
   - The ratio M_T/F_D is unbounded when |F_D| is within 2–3 SE of 0.
4. **No KSC outcome changes adoption, and §6 disclaims transfer.** $3.44 and two hooks buy a label on a forecast. The hooks are one-sided coins on a non-band set and a hand-set student. KSC fails the seed's complexity rule.
5. **The upper bound on |S₁| does not hold.** A further 13.3 readings per side-game have no attack in the top eight with the eighth action within 1 nat (part 1). The census in S0 sizes H first.

### 2.3 Costs

- **The $9.81 is the coin-overhead allowance, not spare money.** After $3.59 of additions, the three-round line falls from $2.92e-5 to $2.77e-5, and overhead room from 0.163 to 0.103 (part 2). S0(c)+ still quotes $2.92e-5.
- **"Saves $2.51 a round" needs batches resized to 100,000/m games.** With fixed 100,000-game batches, r·m = 0.602 saves $1.23, $0.37 or $1.35 at m = 1.073, 1.122 or 1.244 (part 3).

### 2.4 Novelty

- **KSC.** [Farias, Li, Peng & Zheng 2022](https://arxiv.org/abs/2206.02371): Q-differences under the 50/50 mixture (the DQ estimator) equal the first-order truncation around P_½, with bias second order in the intervention (Theorem 1, read on ar5iv). With the design's §3.5, KSC's mathematics is covered; only its use as a known answer is new.
- **ACS.**
  - (1 − ρ)/2 is the variance of the odd part, and ρ ≤ 0 for monotone h is the classical antithetic result (Hammersley & Morton, not re-read; [*A First Course in Monte Carlo Methods*](https://arxiv.org/abs/2405.16359), seen in summaries only).
  - [AIVAT](https://ar5iv.labs.arxiv.org/html/1612.06915) needs exact chance-node expectations, and its authors note that chance branching limited their experiments.
  - A single reflected sibling with the realized sibling priced free was not found in a short search.

## 3. Strongest part

ACS's construction. The correction is exactly centred for any price, so a weak value head, fog or a moving policy costs variance, never bias. It is admitted only on a kept ratio measured on records for $0.15. It reopens round 5's corrector class at one forward per selected fight, against 11.86 for enumeration. The memo states plainly that the record leans against E.

## 4. Repairs

1. **Drop T, and either drop KSC or keep H against M₁ only.**
   - A miss means "curvature or an error the invariants miss".
   - Add the two-point arm if attribution is wanted.
   - Recompute the budget line.
2. **ACS.**
   - Define h at terminal and single-action children.
   - Add the Φ = identity plant.
   - Admit on the measured m, and close the class on measured readings.
   - Fix the batch accounting.
