# Round 6 challenge: the dice teach, the coins audit

**Stance.** The current design learns from its coins and subtracts the dice as noise. At argmax the dice are almost all of a result's variance, and each fight is a randomized experiment with an exactly enumerated law. This proposal learns from the dice: a closed-form regression of the result on each fight's outcome deviations identifies, without confounding, what each outcome is worth in the context of the two fighters' frozen tokens. The exact law of each candidate attack turns that into a score for attack decisions. Coins come second, to audit and calibrate one score at exactly the decisions a rule would change. The same function corrects the luck of every label and match.

Tags: [M] measured (file, script), [D] derived (inputs named), [U] unknown (when it is measured), [C] a choice whose consequences are derived. Scripts are in `analysis/`, named `round6_challenge_*`.

## 0. Why change the core

1. **Where the result's information lies.**
   - In `own1` self-play the seven fixed luck columns explain 0.161 of Var(z) out of sample (0.153 pooled over 23 matches) [M, `round6_challenge_dice_share.out` §1]. That is a lower bound on the dice's share: at argmax, all that the setup (0.110 in-sample) leaves is chance [M, `round5_provenance_match_variance.out`; seed §3].
   - All the current design's coins together cause V_coin = Σ p(1−p)Δ² of the variance, to first order: 1.1e-4 to 5.1e-4 at its own floors (sd 0.0020–0.0038, μ 0 or −0.0018). The dice columns explain 320 to 1,400 times more, 55 times at sd 0.01 [D, same script §3].
2. **The network cannot compute what the dice decide.** A unit token carries HP, moves, XP, cost, leader flag, alignment and type, but no hit chance or kill probability [M: `rust/wesnoth_core/src/encode.rs` unit columns; `round5_invent.md` §1 item 4]. Fights are often undecided: 50.5% have P(defender dies) between 0.05 and 0.95 [M, same script §4, a subsample biased toward fights whose counter weapon was a choice]. At a near-tie between attacks the network cannot use the law; the simulator can.
3. **Reach.** An attack is the top action at 19.7% of decisions. Inside the current band (w = 0.35), 8.0 to 24.3 decisions per `own1` side-game have an attack in the pair (10.5–31.8% of the band; the record does not name the runner-up, hence bounds); within 1.5 nats, 21.5–58.5 [M, `round6_challenge_band_attacks.out`, `parity3` margins in `own1`'s decision unit].
4. **Evidence against.** The one-ply HP-margin look-ahead, a fixed consequence valuation over every decision kind, read −0.0021 ± 0.0033 per flip; exposure −0.0008 ± 0.0016 [M, `round5_challenge_lookahead.out`]. Weights varying with faction and map explain no more than fixed ones: −0.0042 (−0.0076 to −0.0014) [M, dice_share §2]. Unit-level context, which ψ uses, is not on disk.

## 1. The algorithm

### 1.1 What is learned, and from which labels

A game has 109.7 fights on average [M, `round5_synthesis_outcomes.out`]. Before fight e the law p_e(o) is exact, and o_e is drawn from it independently of the past. φ(o) ∈ R⁷ is the outcome as the rules state it: each fighter's HP lost as a fraction of its maximum, its death, its advancement, and whether the game ends. t̃_e ∈ R⁶⁵ is a constant plus 32-dimensional PCA projections (fitted on the first 1,000 games [C]) of the attacker's actor token and the defender's hex token, both from the attack decision's own forward pass. s_e is the attacker's sign in side A's result.

**The luck vector.** For each fight,

    g_e = s_e · (φ(o_e) − Σ_o p_e(o) φ(o)) ⊗ t̃_e  ∈ R⁴⁵⁵,   with E[g_e | F_e⁻] = 0.

**The consequence model.** It is fitted by ridge regression:

    z = c(setup) + Σ_e ⟨W, g_e⟩ + ε,   ψ_W(o; t̃) = ⟨W, φ(o) ⊗ t̃⟩.

The fit is closed-form from per-game sufficient statistics (Σ_e g_e, 455 floats a game), cross-fitted by batch, λ by progressive validation [C]. No network is trained and no forward added; laws cost 0.06–0.11 ms of CPU per fight [M, seed §2].

### 1.2 Identification, bias and variance

- **Identification.** Martingale differences are uncorrelated (E[g_e g_e′ᵀ] = 0 for e ≠ e′, E[g_e c] = 0), so the normal equations are Σ_e E[g_e g_eᵀ] W* = Σ_e E[g_e z], and by the tower property E[g_e z] = E[g_e Q_e(o_e)], with Q_e(o) = E[z | F_e⁻, o_e = o] the post-fight value under the policy that played. W* is the best linear predictor of the post-fight value's within-fight variation in the span of φ ⊗ t̃, identified by randomization as a coin's effect is, whatever else enters z. Unlike Texel tuning's observational regression ([chessprogramming](https://www.chessprogramming.org/Texel%27s_Tuning_Method)), the regressor is randomized.
- **Precision.** Excess variance r·d/N = 0.0041 of Var(z) at 100,000 games, 2.5% of what the fixed columns explain [D, `round6_challenge_power.out` §1, r = 0.897 the setup-only residual].
- **As a corrector.** For any W fixed before a game, L = Σ_e ⟨W, g_e⟩ has mean 0, so corrected matches and coin labels stay unbiased (the luck walk's proof, `tools/match_luck.py`). Its kept ratio r_ψ is read on held-out games for free.
- **Blind spot.** ψ does not see where the attacker stands after the fight; different defenders are compared only through W's dependence on their tokens.

### 1.3 The decision rule, under the "network picks actions" ruling

An attack candidate a scores X(a) = Σ_o p_a^h(o) ψ_W(o; t̃(a)), with the law computed in the mover's observed world (unseen units removed), so the player reads only its own observation; non-attacks score 0, φ's "nothing happens" level. At a band pair (a0, a1) with an attack in it, arm **real** plays argmax X and arm **anti** argmin X; each is one decision, after which the policy re-decides after every roll, and elsewhere the base policy plays.

By the performance-difference lemma each arm's gain is the sum of Δ over its switches ([Kakade & Langford 2002](https://mlanthology.org/icml/2002/kakade2002icml-approximately/)), and the two switch sets are disjoint. The readout is â = R/n⁺ − A/n⁻ (gains R, A; switches per side-game n±, from the logs). With no information both per-switch means equal μ, the attack pairs' mean advantage, so â has mean exactly 0 whatever μ and however unevenly the switches split; (R + A)/(n⁺ + n⁻) estimates μ. â is also read per pair type (attack against attack, attack against other), since a ψ that only sorts pairs by type would move â through type-level means.

### 1.4 Coins second: targeted, at exactly the switches

If D2 reads alignment, coins calibrate and extend the rule: p = 0.25 only at attack pairs within 1.5 nats where ΔX = X(a1) − X(a0) > 0 [C]. The propensity reads only pre-coin quantities, so the current design's label T_i stays exactly unbiased, with ψ as its control variate. Fit E[Δ | G-bin, ΔX] = μ_bin + κ·ΔX and switch where μ̂_bin + κ̂ΔX > 0: seven numbers to learn, not 776 directions. The forecast is model-free (held-out labels summed at the switches), SE 3.1–5.9 Elo at 200,000 coin games [D, power §4], against the current design's 4.7 Elo for $19.8.

### 1.5 Why more signal per dollar

1. **Information:** a training channel with 320–1,400 times the coins' result variance at the current design's floors (§0).
2. **The first decision read costs $3.42** (D1 + D2) [D, power §5], against $9.9 or $19.8 [M, `current_design.md`].
3. **Robust to ruling (a):** D1, D2, adoption and distillation use no coin regression; the dice regression is the luck walk's estimator.
4. **No added forwards:** a learned corrector at fight children costs 2.0–3.9 times a game's forwards [M/D, `round5_synthesis_outcomes.out`]; ψ costs CPU only.
5. **Failure leaves a tool:** a null D2 leaves ψ as a corrector of measured quality for the coin ridge.

### 1.6 Invariants and plants (tests from this spec)

- **J1** On an enumerable duel, the mean of g_e over draws is 0 by χ²; a corrector reading the next fight's outcome fails the same test (negative control).
- **J2** On synthetic z = Σ⟨W*, g_e⟩ + noise, the fit recovers W* within its SE. **J3** The sufficient-statistics fit equals a per-game fit on 1,000 games.
- **J4** With nothing hidden, the observed-world law equals the true one; perturbing a hidden unit never changes X.
- **J5** Logged switches equal argmax and argmin X recomputed offline, also under a planted W; anti's set is real's complement among pairs with ΔX ≠ 0. **J6** ψ-corrected matches preserve the mean (Σt² against χ²).
- **P-dice**, booked into z so that it passes through the regression and the corrector: δ·s_e·(1[defender dies] − P_law), s_e = ±1 by the defender hex's parity, that column added to the design. Predicted coefficient 1 ± 0.136 (2 SE) at δ = 0.02, 100,000 games [D, power §2; the Bernoulli variance comes from the biased subsample]. P-coin and P-distill are the current design's (§3.6), in D3 and D5.

## 2. The regimen

Prices: a generation game $2.47e-5; a match game $1.49e-4 [D, `current_design.md` §4].

| Stage | Cost | Readout | Decision per outcome |
|---|---|---|---|
| **D0** records | $0, done | §0's numbers | They set D1's size (§1.2) and D2's resolution |
| **D1** dice fit | $2.47 + logging overhead [U] | 100,000 `own1` argmax self-play games: P-dice; r_ψ held out against 0.732 (setup + columns) and 0.897 (setup), no prediction (no record holds token-contextual luck); `own1`'s attack-pair census (8.0–24.3 if its margins match `parity3`'s); X's blind reliability; J1–J6 | P-dice outside: regression path broken, stop. r_ψ below the columns: ψ replaces them as corrector everywhere, settling ruling (b). Reliability under 0.8 [C]: 100,000 more games |
| **D2** decision read | $0.95 | real and anti against `own1`, 3,200 corrected games each, side-swapped pairs: â, μ̂, each arm's gain (SE 5.14 Elo, 5.82 without columns) [D, power §3]. No prediction of â: nothing on disk measures it | **â > 2 SE** (line 0.0034–0.0105 per switch, by census): decision information at attack pairs; D3. **Null:** alignment below 0.0017–0.0052 per switch; evaluator closed at $3.42; the current design's S1 runs with the better corrector. **â < −2 SE:** the exchange misleads at near-ties (position dominates); closed |
| **D3** targeted coins | $4.94 | 200,000 coin games (P-coin in batch 1; ψ refitted on their dice). μ̂_bin, κ̂, model-free forecast (SE 3.1–5.9 Elo) | κ̂ not above 0 by 2 SE: keep D2's band rule, its forecast the real arm's measured gain. Otherwise the calibrated rule, its forecast sealed |
| **D4** adoption | $2.03 | Corrected 12,800 games, fresh seeds, against the incumbent (SE 2.57 Elo), plus PURE 800 against `own1`. Predicted: the sealed forecast ± 2 combined SE | The current design's adoption rule; a miss below: second-order loss, a mixture match ($1.9) |
| **D5** distillation | $2.70 | P-distill, then the own-trajectory recipe ($2.1 [M]) on the adopted player's games; π_1 matched against the adopted player | Within 2 SE: π_1 is the base and its near-ties the next band. Below: the switched player plays if ruling (e) allows, else π_1, gap reported |
| **Round k ≥ 2** | $10.62 | D3 on π_k (ψ refit on those games' dice), a D2-type check, D4, D5 | Continue while D4 adopts. A forecast upper bound under 5.3 Elo: the attack band is exhausted |

**Budget** [D, power §5]: round 1 $13.09; rounds 2–4 $31.86 more, $45 in all. The remaining $40 covers the current design's coin ridge on the non-attack band with ψ as corrector ($21.8), the extension below, and $8 for logging overhead and a rematch. A null D2 stops the evaluator at $3.42, and the current design's plan ($76) still fits.

**Extension past attacks (after two adoptions).** A move is scored by the ψ-value of the attacks the opponent's policy would most likely make on the moved unit, in the observed world with exact laws. It needs opponent-policy forwards per candidate, so its price is [U] until D1 times the server; it gets its own real-and-anti read.

**The $1,000 case**, argued only after two adoptions within forecast: ψ as a deep head trained on the dice identity at 10⁶–10⁷ games ($25–250 of generation [D]); the law as observation tokens in a new observation epoch (an imitation pass cost $6.22 [M, `docs/imitation_anneal_prereg_20261003.md`]), so distillation can close on law-based choices; many rounds. Nothing measured links `own1` to human ratings, and nothing rests on it.

## 3. Provenance table

| Number | Value | Used for | Tag | Source or derivation |
|---|---|---|---|---|
| Dice share, fixed columns, `own1` self-play | 0.161 (pooled 0.153) | Core argument, D1 sizing | M | `round6_challenge_dice_share.py` §1, leave-one-match-out on `match_luck_20261019/rows.jsonl` |
| Context by faction and map | −0.0042 (−0.0076 to −0.0014) added share | Evidence against | M | same script §2, ridge 100, game bootstrap |
| Setup share, in-sample | 0.110 | "rest is chance" | M | `round5_provenance_match_variance.out` |
| V_coin at the floors | 1.1e-4 to 5.1e-4; ratio 319–1,412 (55 at sd 0.01) | Core argument | D | 2 × 76.1 × 0.1875 × (v² + μ²); `round6_challenge_dice_share.py` §3 |
| Band per side-game | 76.1 | V_coin | D | current design §2 (0.339 × 224.6) |
| Fights per game | 109.7 | Plant SE, CPU | M | `round5_synthesis_outcomes.out` |
| Kill uncertainty | 50.5% of fights within 0.05–0.95; Bernoulli variance 0.088 | Mechanism, P-dice SE | M (biased subsample) | `round6_challenge_dice_share.py` §4, 6,130 counter-weapon tables |
| No hit chance in the observation | — | Mechanism | M | `encode.rs` unit columns; `round5_invent.md` §1.4 |
| Attack top share | 0.197 of decisions | Reach | M | `round6_challenge_band_attacks.out` |
| Attack pairs in band | 8.0–24.3 per side-game (w 0.35); 21.5–58.5 (w 1.5) | D2 resolution, D3 volume | M bounds, scaled | same script; × 224.6 (`round5_simplify_critique_decision_units.out`) |
| Look-ahead per flip | −0.0021 ± 0.0033; −0.0008 ± 0.0016 | Evidence against | M | `round5_challenge_lookahead.out` |
| d | 455 = 7 × 65 | Fit size | C | §1.1 |
| Excess variance at 100k games | 0.0041 (2.5% of 0.161) | D1 size | D | r·d/N, r = 0.897; `round6_challenge_power.out` §1 |
| P-dice band | 1 ± 0.136 at δ 0.02 | Plant | D | power §2: SE √(0.897/(N·109.7·0.5·0.088)) |
| D2 SE per arm | 5.14 Elo (r 0.700) / 5.82 (r 0.897) | D2 lines | D | 347.4·√(r/3,200); r from `round5_provenance_critique_pairs.out`, `match_variance.out` |
| D2 per-switch line | 0.0034–0.0105 (0.0039–0.0118) | Meaning of D2's outcomes | D | 2·√2·SE/(347.4·m/2), m from the census bounds |
| D3 forecast SE | 3.1–5.9 Elo at 200k games | D3, D4 prediction | D | 347.4·√(L·σ²/(2N)), σ² = 0.732/0.1875; power §4 |
| Prices | $2.47e-5 per game; $1.49e-4 per match game | All costs | D | current design §4 |
| Round 1; first read; later round | $13.09; $3.42; $10.62 | Budget | D | power §5 |
| Distillation; P-distill | $2.1; $0.6 | D5 | M; D | `memory_own_trajectories` record; current design S4 |
| Adoption match | SE 2.57 Elo, $1.9 | D4 | D | current design §2 |
| Corrector cost at fight children | m 2.0–3.9 | Why ψ is free | M/D | `round5_synthesis_outcomes.out` |
| Imitation pass | $6.22 | $1,000 case | M | `docs/imitation_anneal_prereg_20261003.md` |
| Current design's first reads | $9.9 / $19.8 | Comparison | M | `current_design.md` §4 |
| r_ψ | — | Corrector choice; ruling (b) | U | D1, held out; nothing before D1 relies on it |
| `own1`'s attack-pair census | — | D2 line | U | D1 logging; bounds above |
| Alignment â; μ at attack pairs | — | Everything after D2 | U | D2; no stage before it depends on it |
| X's reliability | — | D1 size | U | D1, blind posterior variance |
| Logging overhead | — | D1 dollars | U | D1, timed |
| κ, μ_bin | — | D3 rule | U | D3 |
| Distillation closure on law-based switches | — | D5 | U | D5 match |
| Move-extension price | — | Extension | U | D1 server timing |
| Compounding, exploitability, human level | — | Nothing | U | — |
| p 0.25, 1.5-nat reach, 32-dim projection, 200k coin games | — | Design | C | Consequences derived above |

## 4. What is new, and the nearest work

Searched 2026-10-20; finding nothing does not prove absence.

1. **Self-play's chance outcomes as a randomized experiment that trains an evaluator.** Chance-node control variates serve evaluation and variance reduction: AIVAT ([Burch et al.](https://arxiv.org/abs/1612.06915)), Baseline ([Davidson, Archibald & Bowling 2013](https://webdocs.cs.ualberta.ca/~bowling/publications/b2hd-13aamas-baseline.html)), MCTS ([Veness, Lanctot & Bowling 2011](https://proceedings.neurips.cc/paper_files/paper/2011/file/d736bb10d83a904aefc1d6ce93dc54b8-Paper.pdf)); of AIVAT, VR-MCCFR's authors note it "was only used for evaluation of strategies" ([Schmid et al. 2019](https://arxiv.org/abs/1809.03057)). Exogenous reward is removed for variance ([Dietterich et al. 2018](https://proceedings.mlr.press/v80/dietterich18a.html)); hindsight baselines exclude action information ([Mesnard et al. 2021](https://proceedings.mlr.press/v139/mesnard21a.html)); luck is separated from control in return-conditioned learning ([Yang et al.](https://arxiv.org/abs/2210.13435)). Values learned through dice by TD ([Tesauro 1995](https://dl.acm.org/doi/10.1145/203330.203343)) or with chance codes ([Stochastic MuZero](https://mlanthology.org/iclr/2022/antonoglou2022iclr-planning)) are bootstrapped, not identified by randomization. I found no work that regresses results on chance-outcome deviations to fit an evaluator and scores decisions with it through exact laws.
2. **One function as exact luck corrector and decision evaluator**, its corrector quality measured on every game for free.
3. **Real-and-anti certification**, mean exactly 0 under no information whatever μ, with no fitted placebo (nearest: branch L's placebo arms).
4. **Coins targeted by a pre-coin score** at exactly the decisions a rule would change, past the band: a micro-randomized trial with state-dependent propensities ([Boruvka et al. 2018](https://doi.org/10.1080/01621459.2017.1305274)).

Not claimed: expectimax over chance, corrected matches, the coin label. Contradicting evidence comes from the records (§0 item 4), not from literature I found.

## 5. Weakest point, and rulings

**Weakest point: the bridge from consequence to decision.** ψ prices the fight, not where the attacker stands afterwards. The nearest record, a fixed consequence valuation one ply ahead over all decision kinds, was neutral, and game-level context added nothing. If unit-level context and the law add nothing either, D2 reads null: the route has cost $3.42, names a floor of 0.0017–0.0052 per switch at attack pairs, and returns ψ as a corrector of measured quality. Lesser points: the reach is 8–24 decisions per side-game against the band's 76 until D3 widens it; distillation must close on choices whose reason the network cannot see.

**Rulings for the user:**
- **(e)** The exact law of a candidate attack, computed from the mover's observation (what the attack dialog shows a human), weighted by a learned ψ, at play time. If no, the law-using player is a training-time teacher and the distilled network plays.
- **(f)** ψ reads the fight's outcome as the rules state it (HP lost, death, advancement, game end), with 455 weights learned from the network's tokens and identified by randomized outcomes. Round 4 rejected a result-fitted valuation over a hand-chosen basis; if this falls under it, own1's value head at fight children replaces ψ, at a forward multiple of 2.0–3.9 [M/D].
- **(c)**, **(d)** as in the current design; (a) is not needed before D3.

## 6. Decision record

Rejected: luck weights by faction and map (measured to add nothing); a fitted placebo for D2 (anti has mean 0 under no information); band-wide coins in round 1 (776 directions where one score needs calibration). Deferred: the move extension, until two adoptions; a deep ψ and law tokens, to the $1,000 case; the non-attack coin ridge, after D2 whatever it shows.
