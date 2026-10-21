# Round 9 challenge: factorial self-play — matches label, coins search

Tags as in `current_design.md`: [M] measured, [D] derived, [U] unknown, [C] choice. Scripts in `analysis/`, outputs beside them: `round9_challenge_hook_flips.py`, `round9_challenge_factorial.py` (cited F§n), and from this stance's earlier pass `round9_challenge_coherence.py`, `round9_challenge_es_toy.py`.

## Summary

The current design turns coin labels into a step through a closed-form head, then tests the step with one match. Here authority is reversed. **Every generation game is a mirrored factorial over K candidate policy blocks:** seat A plays each block at one level, seat B at the other, signs from a Hadamard design. With the frozen luck and cell variates, the results read all K blocks' play effects at once, each at least as precisely as its own PURE match. Near-tie coins ride in the same games but only *propose* blocks; a block is adopted only on its measured play effect and a fresh confirmation match. A round costs $3.83 against $23.98 [D, F§5]: six steps where the current design takes one, none resting on a first-order label.

## 0. Why change the core

1. **A step needs search and evaluation; the current design pays for search where the record shows little.**
   - Of a $23.98 round, $22.79 buys 800,000 coin games for a 788-dimensional head; adoption costs $0.98 [D, `round9_simplify_budget.out`].
   - The band's mean advantage is 0 (+0.0003 ± 0.0007); nine observable partitions show no spread at a cell-mean sd of about 0.005 [M, seed §7; `round9_provenance_cells.out`].
   - The one structure found, recruit choice by unit identity, is worth +22.9 ± 3.2 Elo first order, from 60,000 games. A rule chosen on 30,000 of them reads +21.3 ± 4.5 and +24.8 ± 4.5 on the other 30,000 [M, `round9_provenance_recruit_cells.out` §3, §5]: a small batch finds it.
2. **First-order labels do not price what a coherent change does in play.** S0(a)'s per-seat attack hook, the one coherent change measured in play, barely changes the attack count (+1.05 ± 0.79 per side-game) but removes 9.38 ± 1.80 moves: units attack before moving. Each changed decision costs −0.0126 ± 0.0028 z, against −0.0032 ± 0.0031 per in-band attack-promoting coin switch; difference −0.0094 ± 0.0041 (z −2.3) [M, `round9_challenge_hook_flips.out`]. Past the band, a coherent change costs four times per decision what the band's labels say; only play prices it. This is the current design's weakest point 3, measured once.
3. **Coherence favours play measurements for blocks that move several decisions.** For equal SE on a block's effect, coin welfare needs n·V_T/(2σ_R²) = 2.87 n times the games of a seat-level contrast, n the decisions it switches per side-game [D, F§3]: 6.3× for the recruit rule R. Stockfish's tuner, mirrored θ+ against θ− games, 30,000–100,000 a match, reports +40–70 Elo; its author found single piece-square entries "ineffective" and tuned shared knobs ([chessprogramming.org](https://www.chessprogramming.org/Stockfish's_Tuning_Method), read): coherence again.
4. **A factorial reads K blocks for the price of one match** (Hadamard signs, orthogonal main effects): 0.92 Elo each at 100,000 games, against 1.10 for one dedicated raw-z match of that size [D, F§2].

## 1. The algorithm

### 1.1 What is learned

- **The policy** π_k = (θ_k, B_k): the weights plus a set of adopted blocks.
- **A block** maps a decision's candidates to logit changes, keyed only on what the network already scores: the pair's kinds, the candidates' unit identities (recruit unit, actor's and target's unit types), the side-turn, the margin G. A *switch block* plays the runner-up in a set of cells when G < 0.35; a *bias block* adds s·v(identity) to every matching candidate's logit, reaching past the band.
- **The network picks every action** at argmax and re-decides after every roll.
- **Learned values:** cells and values come from coin labels, adoption from play. Distillation absorbs blocks into θ when the optimizer ruling allows (§2, D); until then they are decode rules, and the band moves as blocks are adopted (the top two are taken after the blocks).

### 1.2 Generation: one game, two experiments

- **Factorial:** game g takes row g mod 2^m of a Sylvester Hadamard matrix (2^m > K), columns 1..K: ε_g ∈ {−1, +1}^K. Seat A plays block k where ε_gk = +1, seat B where ε_gk = −1, so each block is on for exactly one seat in every game. Signs keyed by seed and game index, logged (I-design).
- **Coins:** band coins on both seats, as built (w 0.35, P 0.25 [M, passed]). The configuration is fixed at game start and so is part of each coin's history: every label stays exactly unbiased for its decision's advantage under the configured policy. Proposal labels come only from seats whose blocks touching that cell are at the level this round's step adopts (usually half the seats).

### 1.3 The estimator

    δ̂_k = (1/N) Σ_g R_g ε_gk,   R_g = z_g − b(cell_g) − βᵀS_g

b and β are frozen before the batch (fitted on `own1`'s 60,000 coin games; the simplify stance's coefficients).

**Exactness.** ε is independent of everything, b is fixed, and each luck term has mean zero given its pre-fight state whatever the policy, so E[R_g ε_gk] = E[z_g ε_gk]: block k's main effect (Walsh coefficient) over the design.

**Meaning.** With additive seat contributions, one-seat effects e_k and within-seat pair interactions a_jk,

    E[z | ε] = Σ_k e_k ε_k + Σ_{j<k} a_jk (ε_j + ε_k)/2,

so δ_k = e_k + ½ Σ_j a_jk [D]: pair interactions alias into main effects at half weight (Monte Carlo +0.0351 against +0.0350 [D, F§6]).

**Variance.** Var(R)/N, the other blocks included: 2.90 / 1.49 / 0.92 Elo at 10,000 / 37,800 / 100,000 games at σ_R² 0.699 [M]; simulated SE 0.00494 against 0.00495 [D, F§6].

**Invariant found by the Monte Carlo:** cross-fitting halves must be whole Hadamard cycles; a parity split holds columns constant and halved the prediction until fixed [D, F§6].

### 1.4 Proposals (the search)

A pre-registered registry of partitions keyed on §1.1's identities. Per round: the cross-fitted welfare of "switch where the cell mean > 0" per cell (20 folds, unbiased for the fold rules, as the provenance stance computed on batch 1); positive cells grouped by (partition, faction) into blocks, kept if the predicted effect exceeds 3 SEs of the next factorial, 2.8 Elo at 100,000 games [D]. A bias block extends a kept block's values to every candidate at one scale s [C]; whether identity preferences extend past the band is unknown, and its sign is the reading. **Optional head block:** the current design's ridge head on a feature dump, entered as one block and priced in play.

### 1.5 The step and its certificate

1. Select blocks with δ̂ > 2 SE on half A (whole cycles); predict Σ δ̂ on half B, unbiased for the step's additive effect (Monte Carlo: +0.00044 ± 0.00055 [D, F§6]). Interactions with unselected blocks bias it by −½ Σ a_jk [D]; the confirmation reads that.
2. **Confirmation:** π_k plus the selected blocks against π_k, 37,800 pool games on fresh seeds, raw z, SE 1.79 Elo; adopt iff the lower 2-SE bound is above 0 (false adoption ≤ 0.023 per proposed step; all-null simulation 0 of 400). Miss line 2√(k·1.30² + 1.79²) Elo; simulated misses 0.040 against a nominal 0.046 [D, F§6].
3. The levels are the step: no step size is extrapolated. A step sized from a gradient and a curvature trace loses when curvature along the gradient dominates (−0.0044 to −0.75 z realised, `round9_challenge_es_toy.out` B).

### 1.6 Why more learning signal per dollar

- **Steps per dollar** 6.3× ($23.98 / $3.83) [D].
- **Evaluation per game** 6.3× coin welfare for an aggregated rule, 0.7–1.6× for faction-level blocks [D, F§3], always in play; K readings per batch.
- **Search kept:** coins run in every game (coin games cost 1.10× pool games, all post-passes included [M, `round9_simplify_labels.out`]); the head block keeps the current design's search as one candidate.
- **Less unchecked machinery:** no 24 GB dump by default, no threshold pre-test, no sealed forecast, no reset before compounding.

### 1.7 Invariants and planted checks

**Invariants:** I-design (logged ε equals the Hadamard row of the game index; halves and batches are whole cycles); I-block (every decision where an active block changes the argmax logs it, played its choice, and recomputes; a unit test with a planted block table); I-swap (a seat swap flips every δ̂); I-coin, I-log, I-V and the dice control as built and passed [M].

**Planted checks, one per code path, read at 3 SE:**

| Code path | Plant | Prediction | Failure reading |
|---|---|---|---|
| Factorial read (parse, join, orientation) | Separate copy z′ = z + κ Σ_k u_k ε_k, parsed by separate code | δ̂′ − δ̂ = κu exactly under whole cycles (F§6: recovered 0.0200 on the planted blocks, 0.0000 elsewhere) | Any deviation is a code error |
| Factorial in play (real known answer) | Attack hook +1 as a block in F1a | [−57.1, −25.5] Elo [D from M, F§4] | A swap reads about +41; a broken hook about 0 |
| Proposals | Coin plant (passed, 0.994/0.995 [M]); planted cell copy z″ = z + Σ s(D − P)κ·1[cell = c*] | c*'s cross-fitted mean recovers κ (slope 1), and the registry ranks it with welfare κ·n_c* | Slope off 1 |
| Distillation | The own-trajectory planted relabel (as in Branch S's B1) | Closure above 0 by 4 SE | Closure ≈ 0 |

## 2. Regimen

Costs [D, F§5] are at the measured host's $0.577 an hour.

| Stage | Cost | Readout and prediction | Decision per outcome |
|---|---|---|---|
| **F0** records | $0 | Block rules from batch 1: rule (a) per mover faction, hashed; offline planted-copy identity | A plant fails: fix before any box |
| **F1a** smoke, `own1` | $0.26 (10,000 games) | Hook block in [−57.1, −25.5]; the six recruit blocks read but not gated; I-design, I-block, planted copy | Outside the window: the factorial path is broken; debug |
| **F1b** `own1` | $0.98 (37,800 games) | Six faction blocks against their first-order predictions [M]: Undead +9.2 ± 1.7, Northerners +5.5 ± 1.3, Drakes +3.6 ± 1.4, Knalgan +3.4 ± 1.6, Loyalists +3.0 ± 1.0, Rebels −1.7 ± 1.2. Miss lines ±5.4 to ±6.8 each; the sum against +23.1 ± 3.4, miss line ±10.0 | **Within the lines:** labels carry to play per block. **All six ≈ 0** (4.6 SE below the prediction): labels do not carry to play even where strongest; switch blocks retire and the escalation menu applies. **Sign flips** locate where they do not |
| **R1** `own2` | $3.83 | Factorial with band coins, 100,000 games: the six blocks (does a rule learned on `own1` hold on `own2`?), one recruit bias block, and an end_turn offset block at −1.5 ± 0.5; each read to 0.92 Elo. Confirmation of the selected step. No magnitude is predicted for `own2` | Adopt by the confirmation's bound. A null: `own1`'s rule does not carry to `own2`, and R2 proposes from `own2`'s own coins, already logged in R1 |
| **Claim** | $1.13 | π_k against `own1`, two networks, 12,800 games; predicted 22.7 + Σ confirmations, SE √(4.4² + Σ 1.79²); miss line 2√(SE² + 3.07²) | Below the line: intransitivity or a two-network defect; reported |
| **R2…Rk** | $3.83 each | Proposals from the previous round's coins; factorial; step; confirmation | No proposal ≥ 2.8 Elo, or no adoption: the structure visible at 100,000 games is exhausted for this π_k; escalate (below) |
| **D** distillation | $3.90 | When the optimizer ruling allows: π_k distilled by the own-trajectory recipe onto its block-applied target; closure held out, then N against π_k (two networks) | Closure not above 0 by 2 SE: absorption broken. Else N, without blocks, is the next base |

**Escalation after a null:** a head block on a feature dump; band 0.7 for proposals (125 band decisions per side-game, `current_design.md` §4); D, which moves the band; Branch S's rating input as a one-network block (s_top against s_mid), once trained.

**Budget** [D, F§5]: $82.34 left. With F1, three claims and three distillations ($16.35), 17 rounds fit ($81.47); 20 without distillation. A null first round costs $5.07 (F1, R1), against $24.27 for the current design's at measured prices (`round9_simplify.md`). If ruling (b) is refused, raw z replaces R: SEs grow 1.20×.

**Toward and past human strength.** Confirmations certify each round; claims chain them against `own1`. Nothing links `own1` to human ratings, and nothing rests on it. No gain per later round is predicted [U]; rounds stop on their own null rule. The $1,000 case is argued only from three adopted rounds' measured gains, at about $3.83 a round.

## 3. Provenance

| Number | Value | Tag | Source or derivation | Used for |
|---|---|---|---|---|
| Hooked seat; moves, attacks changed | −41.1 ± 4.3 Elo; −9.38 ± 1.80, +1.05 ± 0.79 per side-game | M | `round9_challenge_hook_flips.out` (6,400 S0(a) games, randomized seat) | §0.2 |
| Per changed decision: hook vs band | −0.0126 ± 0.0028 against −0.0032 ± 0.0031 z; difference −0.0094 ± 0.0041 | M/D | same; the band figure from `round9_challenge_coherence.out` §4 (−4.6 ± 4.4 Elo over 4.12 switches) | §0.2 |
| S0(a) hook | −41.3 ± 4.4 Elo | M | seed §7, `readout_a_offline.txt` | F1a window |
| σ_R² frozen β + shrunk cell | 0.699 ± 0.003 (`own2` uncoined 0.692) | M | `round9_simplify_residual.out` | Every factorial SE |
| V_T frozen baselines | 4.01 | M | `round9_simplify_labels.out` | Coherence ratio |
| Coherence ratio | 2.87 n; R 6.34; faction blocks 0.73–1.55 | D | V_T/(2σ_R²)·n, n from `round9_provenance_recruit_cells.out` §3, §6 (switch share 0.602 × decisions) | §0.3, §1.6 |
| Block SE | 2.90 / 1.49 / 0.92 Elo at 10,000 / 37,800 / 100,000 | D | √(0.699/N)·347.4 | All stages |
| Recruit structure | R +22.9 ± 3.2 (2.21 switches per side-game); rule (a) +23.1 ± 3.4; halves +21.3 ± 4.5, +24.8 ± 4.5 | M | `round9_provenance_recruit_cells.out` §3, §5 | §0.1, proposal batch size |
| Per-faction predictions | Undead +9.2 ± 1.7 … Rebels −1.7 ± 1.2 | M | same, §6 | F1b |
| F1 windows | Hook [−57.1, −25.5]; faction miss lines ±5.4–6.8; sum ±10.0; all-null 4.6 SE | D | F§4 | F1 gates |
| Band census; mean label; coarse nulls | 77.6 per side-game; +0.0003 ± 0.0007; nine partitions, no V above 1.2 SE | M | seed §7; `round9_provenance_cells.out` | §0.1 |
| Prices | Coin games all-in $2.849e-5 (1.10x pool); pool $2.598e-5; two-network $8.848e-5; $0.577/h | M | `round9_simplify_labels.out` [prices] | All costs |
| Distillation | 4.8 box-hours | D from M | $2.1 at about $0.44/h, `docs/memory_own_trajectories_prereg_20261011.md`, via `round9_provenance_budget.py` | D |
| Round; claim; D; smoke; F1b | $3.83; $1.13; $3.90; $0.26; $0.98 | D | F§5 | Regimen |
| Rounds that fit | 17 (with 3 D, 3 claims), 20 without D | D | ($82.34 − $16.35)/$3.83 | Budget |
| Current round; its null round 1 | $23.98; $24.27 | D | `round9_simplify_budget.out`; `round9_simplify.md` | Comparison |
| Budget left | $82.34 | D | seed §7 | Budget |
| Monte Carlo | Bias ≤ 0.0004; SE 0.00494 against 0.00495; aliasing +a/2; planted recovery exact; split-half prediction +0.0004 ± 0.0006; miss rate 0.040; null adoption 0/400 | D | F§6 (binary results, known-law luck variate) | §1.3, §1.5 |
| ES step overshoot | Realised −0.0044 to −0.75 z when curvature along the gradient dominates | D | `round9_challenge_es_toy.out` B | Finite steps, §1.5 |
| `own2` over `own1` | +22.7 ± 4.4 | M | `own2_base_20261020` reading | Claim prediction |
| Band at 0.7 | 125 per side-game | Carried | `current_design.md` §4 (S3-null option (i)); nothing rests on it before an escalation is chosen | Escalation |
| Stockfish SPSA | +40–70 Elo; 30,000–100,000 games a match; fine knobs ineffective | Literature, read | chessprogramming.org | §0.3, §4 |
| Proposal supply after recruits | — | U | Each round's registry; nothing relies on it: a null round triggers escalation | Rounds |
| Block effects on `own2`; aliasing size | — | U | R1; the confirmation reads the step whole | Adoption |
| Bias blocks past the band | — | U | R1's bias block; its sign is the reading | Proposal forms |
| Distillation closure on block targets | — | U | D, held out, before its match | Absorption |
| Gain per later round; human level | — | U | Not predicted; nothing rests on it | — |
| K, 100,000-game rounds, 37,800-game confirmation, 3-SE proposal floor, s, ±0.5 offset levels | — | C | Consequences derived above | Design |

## 4. What is new, and the nearest work

Searched 2026-10-21; finding nothing does not prove absence.

**Nearest work.**
- **SPSA** ([Spall 1992](https://doi.org/10.1109/9.119632), recalled) in games. [Kocsis & Szepesvári 2006](https://mlanthology.org/mlj/2006/kocsis2006mlj-universal) (abstract, read): plain SPSA "made no progress with the resources allocated" in poker; with common random numbers, antithetic variables and RPROP it was competitive with TD-learning. That is contradicting evidence for plain perturbation search. Stockfish's mirrored θ± tuner is the supporting evidence, and it documents non-convergence: unimportant variables random-walk and erode strength unless tuning stops.
- **Guided ES** ([Maheswaranathan et al. 2019](https://arxiv.org/abs/1806.10230)) stretches the search along surrogate gradients; coin labels are the surrogate here.
- **Factorial screening** (Fisher; Plackett–Burman) and response-surface steps (Box & Wilson 1951), recalled. Reused parts: chance-law variates ([AIVAT](https://arxiv.org/abs/1612.06915)), the current design's coins and EWM.

**New, as far as found:**
1. Mirrored-seat Hadamard factorials over *learned* policy blocks in self-play, all K main effects read per batch with exact chance-law variates; SPSA tunes hand-written parameters one random direction per game pair.
2. One game, two experiments: block factorials adopt, near-tie coins propose.
3. Screening, split-half prediction and a fresh confirmation in place of SPSA's random walk, which removes the non-convergence Stockfish documents.
4. The coherence ratio n·V_T/(2σ_R²) assigning each question to coins or to play, and the measured four-fold cost per decision of a coherent change past the band.

## 5. Weakest point

**Proposal supply.** The policy moves only through blocks small coin batches can see: about 2.8 Elo or more at 100,000 games. The record holds one such structure (recruit identity); nine coarse partitions show nothing. Diffuse structure, thin over many decisions and linear in the token features, is what the current design's 800,000-game head is built to resolve; here it is seen only through the head block, at the current design's price. If the spread outside recruits is small, this regimen wins by round cost; if large and diffuse, the current design's S1 wins and the head block becomes this regimen's main proposal. Nothing on disk says which.

**Lesser:** within-seat interactions alias at half weight, so attribution between interacting blocks is lost (the confirmation reads the step whole); decode blocks accumulate as code paths until D; the offset block learns a level but does not remove the offset (ruling (d)).

## 6. Rulings and decision record

- **(a), narrowed:** coin labels only propose; every adopted change is a measured play effect certified by a fresh match.
- **(b):** the frozen luck and cell variates in factorial reads (raw z costs 1.20× in SE).
- **(h)** (the provenance stance's): blocks keyed on candidates' unit identities.
- **The optimizer ruling** holds only D; rounds train nothing.

**Rejected:** continuous SPSA updates (no adoption certificate; the field reports a random walk); ES steps sized from a gradient and a curvature trace (they overshoot, `round9_challenge_es_toy.out`).
