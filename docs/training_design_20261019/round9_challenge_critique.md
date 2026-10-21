# Round 9 challenge critique: factorial self-play

Checks: `analysis/round9_challenge_critique_checks.py`, output beside it (C§n). The stance's hook and factorial scripts rerun to identical outputs.

## Verdict: repairable

The algebra holds: mirrored seats cancel the product terms of pair interactions, which alias only into main effects. Four defects change numbers that stages rest on:
- column 1 aliases the side;
- coins halve the effects R1 reads;
- the claim's prediction is built from selected readings;
- the switch-block path has no known answer.

Repaired, the factorial is an evaluation stage for the current design, not a new core: the record shows no proposals beyond recruits.

## 1. Provenance audit

| Number | Class | Check |
|---|---|---|
| Hook −41.3 ± 4.4 Elo; −0.118 z | M | Seed §7; the script's rerun gives −41.1 ± 4.3 |
| Hooked seat: −9.38 ± 1.80 moves, +1.05 ± 0.79 attacks | M as counts | Rerun |
| −0.0126 z per changed decision; "four times" | Tagged M/D; not identified | §2.2 |
| In-band −0.0032 ± 0.0031 z per switch | D | −4.6 ± 4.4 Elo over 4.12 switches; redone, matches `coherence.out` §4 |
| σ_R² 0.699; V_T 4.01 | M | `simplify_residual.out` pooled 0.6988; `simplify_labels.out` 4.011 |
| Recruit R +22.9 ± 3.2; rule (a) +23.1 ± 3.4; halves +21.3, +24.8; faction values | M | `provenance_recruit_cells.out` §3, §5–7 |
| Block SE 2.90 / 1.49 / 0.92 Elo | D | Arithmetic correct; the wrong estimand for R1 (§2.1b) |
| Coherence ratio 2.87 n | D | Correct uncoined; 0.72 n coined |
| F1 windows (±15.8; ±5.4–6.8; 4.6 SE) | D | Redone |
| Prices; round $3.83; claim $1.13; D $3.90; 17 rounds | M/D | Arithmetic correct |
| "Steps per dollar 6.3×" | Tagged D; rests on proposal supply [U] | §2.3 |
| Claim prediction 22.7 + Σ confirmations | D, biased | §2.4 |
| Monte Carlo (bias, SE, null 0/400, miss 0.040) | D | Reruns; models neither coins nor side alternation |
| K, 100,000, 37,800, 3-SE floor, s, ±0.5 | C | — |
| Stockfish: +40–70 Elo, 30,000–100,000 games, random walk | Literature | Read on [chessprogramming.org](https://www.chessprogramming.org/Stockfish's_Tuning_Method) |

No guessed number sets a budget or kill line; the guessed premise is the case for replacing the core (§2.3).

## 2. What would make it fail

### 2.1 Two errors in what R1 estimates

**a. Column 1 is the side.**
- `run_elo_batch` puts side_a = 1 on even slots (`slot_for`); Sylvester column 1 is (−1)^g, and I-design keys rows on the game index. Block 1 is the first-mover contrast: +9.3 ± 1.4 Elo in batch 1, +10.3 pooled over the walk's 23 matches, no excess spread (Q 17.4 on 22 df) [C§A].
- **Raw z** (the stated fallback if ruling (b) is refused): block 1 is biased by +9.3 Elo, 8.5 of its SEs at 100,000 games. In F1b, whichever faction sits in column 1 reads outside its ±5.4–6.8 line.
- **With R**, the map × side baseline leaves +1.4 ± 2.0 Elo on held-out games. Because b is frozen, that same bias returns every round.
- §1.3's "ε is independent of everything" is false; the planted copy, I-swap and the hook window (±15.8) cannot see it.
- **Repair:** use row ⌊i/2⌋ mod 2^m, so each row is played from both sides.

**b. Coins dilute switch blocks.**
- Switch blocks act at G < 0.35, where R1 coins both seats at P 0.25: the block-on seat plays the block's choice with probability 0.75, the block-off seat with 0.25. R1 reads half the uncoined effect [C§E].
- The 0.92-Elo SE becomes 1.84 Elo on the quantity the confirmation measures.
- The split-half prediction is half the confirmation's expected reading: a recruit-sized step (about 23 Elo, predicted about 11.5) lands outside a six-block miss line of 7.3 Elo.
- The coherence ratio falls to 0.72 n: 1.58 for R and 0.18–0.39 for the faction blocks. At those ratios, coins beat the factorial.
- "One game, two experiments" does not survive. Factorial reads must run uncoined, as F1b already does, or be predicted at the diluted scale.

### 2.2 The coherence measurement does not measure coherence

- The hooked seat's 9.38 lost moves reappear on the opponent: +9.87 ± 1.87, summing to +0.50 ± 2.59 [C§B]. Recruits move the same way (−0.43 against +0.43).
- These counts are a downstream shift of material and tempo, not decisions the hook changed, so −0.0126 per decision has no identified denominator.
- The hook also flips decisions up to 1 nat from the top, while the band's are under 0.35. Margin explains a larger cost per decision as well as coherence does.
- §0.2 and novelty claim 4 rest on this number. F1b asks the same question properly: drop the claim and keep F1b.

### 2.3 Proposal supply decides the comparison

- 6.3× is a ratio of round costs; steps per dollar requires proposals, and the record has none outside recruits: non-recruit types +0.90 SD excess dispersion, attack pairs −0.03 SD, nine coarse partitions within 1.2 SE (`round9_provenance_unit_cells.out`, `round9_provenance_cells.out`).
- R2 proposes from R1's eligible seats, about half of 100,000 games: fewer than the 60,000 that found only recruits. The expected path is F1, R1, then the head block at the current design's price; 17 rounds describe a path the record does not predict.
- The stage order is safe (null rules stop it); the case for replacing the core is not made.

### 2.4 The claim's prediction is selected

- A step is adopted iff its reading exceeds 3.58 Elo, and the claim's prediction sums those readings. At a true step of 2.8 Elo this biases the prediction by +1.96 Elo per adopted step, +5.9 over three. The claim then falls outside its miss line with probability 0.119 against a nominal 0.046 [C§C].
- The stance would read such a miss as intransitivity. **Repair:** predict from the half-B estimates, whose noise is independent of the confirmation's. The current design's claim (`round9_simplify_budget.out`) has the same flaw.

### 2.5 Planted checks that pass on broken code

- The planted copy is an identity on the logged ε: it passes under a wrong seat, the side alias and coin dilution.
- The only known answer in play is a bias block. The switch-block path (identity keying, band after blocks) has a unit test only, and I-block passes on an empty log, so F1b's "all six ≈ 0" cannot be told from "blocks never fire".
- **Repair:** gate each block's switch count per side-game on batch 1's prediction (0.602 × 0.42–0.90); run the hook once through the switch-block path.

### 2.6 Power, rulings, complexity

- The 37,800-game confirmation adopts a 2.8-Elo step with probability 0.33 [C§D]: the factorial's 0.92 Elo is wasted behind a 1.79 gate.
- Until D, which waits on the optimizer ruling, blocks are decode-time tables keyed on unit types that override the network. Ruling (h) covers the keys, not a player that is network plus tables, against "deep learning on the project's transformer".
- The offset block tunes a hand-set knob the record reads flat, with no prediction: drop it.
- Cross-seat interactions alias into main effects at (ε_j − ε_k)/2, and §1.3 omits them.

### 2.7 Novelty

Mirrored θ+/θ− games are Stockfish's SPSA; Hadamard-cycled ± perturbations are [Bhatnagar, Fu, Marcus & Wang 2003](https://www.jhuapl.edu/spsa/PDF-SPSA/bhatnagar_ACM.pdf) (abstract read); discrete levels [Wang & Spall 2011](https://www.jhuapl.edu/SPSA/ACC11_discreteWangSpall.pdf) (abstract only); screening then confirmation is standard (recalled). New: chance-law variates in the read, and coin labels as its proposals.

## 3. Strongest part

**F1b.** For $0.98, six faction recruit blocks on `own1`, each against a measured first-order prediction with a derived miss line; the sum +23.1 ± 3.4 against ±10.0, "all null" 4.6 SE away. It measures the current design's weakest point, labels carried to play ("detected, not sized"), on the strongest structure on record, before any round is paid. The uncoined ratio 2.87 n is correct: fixed multi-decision rules are cheaper to read in play than by welfare. Adopt F1b into the current design ahead of S1: uncoined, with the side fix and the fire-count gates.
