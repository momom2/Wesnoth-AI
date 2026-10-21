# Round 10, invent stance: herded argmax, the player plays its own recruit mixture

Tags: **[M]** measured (record and script named), **[D]** derived (inputs named), **[U]** unknown (measured before anything relies on it, or nothing relies on it), **[C]** a choice whose consequences are derived. Scripts are in `analysis/`, each with its `.out`:
- `round10_invent_composition.py` (*comp*) and `round10_invent_composition_own2.py` (*comp2*);
- `round10_invent_recruit_extract.py` with `round10_invent_recruit_rules.py` (*rules*, §n);
- `round10_invent_feedback.py` (*feedback*);
- `round10_invent_shares.py` (*shares*);
- `round10_invent_herding_checks.py` (*herd*).

## Summary

1. **A measured pathology.** At argmax, `own1` and `own2` recruit one unit type over and over. In turn 1 they field 1.35 distinct types, against 3.16 for the humans they imitate [M, *comp*, *comp2*]. Batch 1's labels price it: at recruit near-ties, the type the side has fewer of is better, increasingly so as the count gap grows. In side-turn 1, at a gap of 2 or more, the gain is +0.070 ± 0.010 z per decision [M, *rules* §1].
2. **The core stays as designed:** near-tie coins, exact labels, the closed-form head, adoption by match, per-round claims and the reset.
3. **New component: herded argmax (HA).** At a recruit decision the player picks the type whose realised count lags furthest behind the network's own cumulative probability for it. HA has no fitted value and no knob, and it keeps play deterministic, so the coin labels stay exact. Inside the band it acts like C, "recruit the type you have fewer of", which collects +18.4 ± 3.2 Elo first order. The 43-value R collects the same: C − R is −1.4 ± 3.1 [M, *rules* §2].
4. **One $0.98 match, H, joins box A:** `own2`+HA against `own2`, read against T at SE 2.53. Its out-of-band part has no forecast, and each outcome sets π_0 or retires HA. Section 7 moves three decisions:
   - the base is `own2`, whose collapse equals `own1`'s;
   - the measured prices leave the whole path $5.51 short with H;
   - S0(a)'s −41 Elo shows that coherent changes of many decisions need play, not labels.
5. **Weakest point.** HA herds toward the humans' mixture, and nothing shows that mixture suits `own2`. Loyalists collapse most and gain least in band.

## 1. What the records say

**The collapse** [M; *comp*: 6,402 S0(a) sides on the unhooked seat, 18,586 human sides on the same 21 scenarios; *comp2*: 6,402 `own2` sides of S0(b)]:

| Per side-game | `own1` | `own2` | humans | humans rated ≥ +146.5 |
|---|---|---|---|---|
| Turn-1 distinct types | 1.345 (0.007) | 1.356 (0.007) | 3.164 (0.009) | 3.131 (0.026) |
| Turn 1 a single type, given ≥ 3 recruits | 0.659 | 0.648 | 0.059 | 0.025 |
| A turn-1 recruit repeats the previous one | 0.916 | 0.913 | 0.313 | 0.248 |
| Whole-game Herfindahl index of type shares | 0.514 (0.003) | 0.514 (0.003) | 0.274 (0.001) | 0.285 (0.003) |

`own2`'s Loyalists recruit 0.794 Spearmen, against the humans' 0.302. At the human shares, 8.4 of `own2`'s recruits per side-game would change [D, *shares*]. That is a scale only: it assumes the network's cumulative probabilities equal the human shares, which is unverified.

**Batch 1's labels** [M, *rules*: 514,281 different-type recruit near-ties, 20 game folds, game-clustered SEs]:
- **By count gap.** Let d = n0 − n1, the earlier recruits of the argmax's type less the runner-up's. The mean label is:
  - −0.0179 ± 0.0063 at d ≤ −2;
  - −0.0049 ± 0.0068 at d = 0;
  - +0.0313 ± 0.0082 at d = 3–4.

  In side-turn 1 it reaches +0.0700 ± 0.0103 at d ≥ 2.
- **Inside unit identity.** With the 43 unit columns in the fit, the slope on d is +0.00267 ± 0.00048 per unit (z 5.6) (§7).
- **Cross-fitted rules** (§2–3):
  - C, switch iff n1 < n0: +18.4 ± 3.2 Elo per side-game at 2.26 switches;
  - R: +19.9 ± 3.1 on these folds;
  - R plus d, against R: +4.1 ± 1.7;
  - where both C and R switch: +16.2 ± 2.3.
- **By margin.** For d ≥ 1 the label is flat in the gap G within the band: slope −0.028 ± 0.040 (§6). Nothing is extrapolated past it.
- **R's count feedback.** In R's own play, d at its switches is 1.39 lower than in the coin world: about −2.85 Elo (slope SE 0.51) [D, *feedback*], a second-order term well inside L's 7.2 miss half-width.
- **Against the mechanism.** C's in-band gain by faction does not follow the size of the collapse: Loyalists +1.1 ± 1.0, Undead +5.5 ± 1.4 (§4).

## 2. The component: herded argmax

### 2.1 Definition

For a side with recruit types O, the decoder keeps W, zero at game start. At a decision whose argmax action is a recruit:

    q_t(o) = actor mass of recruit slot o, renormalised over legal slots
    W(o) += q_t(o) for legal o;   o* = argmax_legal W (ties: larger q_t);   W(o*) −= 1

The played action is the argmax hex for type o*.
- **Cost.** q_t comes from the compact priors `tools/raw_player.py` already receives, at no extra forward.
- **Untouched:** whether to recruit, end_turn, moves and attacks.
- **Refusals.** A recruit refused for lack of a vacant castle hex undoes its update, as the memory's `drop_last_pending` does. A relocated recruit keeps its type, so its update stands.
- **Tempered HA** herds toward q^{1/T}; T → 0 is argmax. At T 0.05 it departs at 0.017 of decisions with a top-two ratio ≥ 1.5 and at 0.42 of those below 1.05 [D, *herd* §4]. Only T = 1 is proposed.

### 2.2 Invariants

- **Bound.** |n_t(o) − Q_t(o)| < |O| − 1, with Q the cumulative q.
  - *Proof, without masks:* after the addition Σ W = 1, so the chosen W is at least 1/|O| and stays above −1 after the subtraction. An unchosen W only rises. So every W > −1, and since Σ W = 0, max W < |O| − 1.
  - Synthetic streams reach 1.10, or 1.12 with random masks [D, *herd* §2–3].
  - Fixed weights {5, 1, 1}/7 reproduce nginx's smooth weighted round-robin sequence, a a b a c a a (§1).
- **Coin exactness is kept.** W is a function of the history, and the design's proof of E[T_i | H_i] = Q̃(H_i, a1) − Q̃(H_i, a0) needs only a decode that is a fixed function of history. The continuation includes HA.
- **I-herd.** Separate code recomputes, from the logged q_t, that every HA decision played argmax W, and logs max |n − Q| per game. A q numerics flip can flip a W tie, the same class as argmax's flips (1.76% of decisions, 0.53 of those with gaps below 0.0136 [M, `round5_simplify_critique_extra.out` §1]).

### 2.3 Why it should work

1. **Imitation learns a mixture.** Humans mix (turn-1 repeat 0.313), and the network's band holds 4.29 different-type recruit near-ties per side-game.
2. **Argmax takes the conditional mode every time.** The repeat rate of 0.916 says the mode does not move as the army fills.
3. **The labels show diminishing returns per type for the network's own play.** They rise with d, inside identity too.
4. **In band, HA departs where d > 0.** At a near-tie, W(a1) − W(a0) is about Q(a1) − Q(a0) − (n1 − n0), and Q's gap stays small while the preference stays near-tied. HA's departures there are C's switch set: +18.4 ± 3.2 first order.
5. **Out of band, nothing prices HA,** and S0(a) shows coherent changes can cost. H measures.

Greedy language decoding loops the same way ([Holtzman et al. 2020](https://arxiv.org/abs/1904.09751), recalled).

## 3. The algorithm

**Learned:**
- the head's ŵ, from coin labels by the design's closed-form ridge, λ chosen by the progressive loss, switching above the tested τ̃;
- the reset's distilled network.

HA learns nothing: it decodes what the network learned. b and β stay frozen.

**Labels.** The design's T_i = s_i ω_i (z − b − βᵀS), exact for any history-measurable decode. If H adopts HA, π_0 = `own2`+HA and two things are added:
- **Audit pairs.** Wherever HA's type differs from argmax's, the coin pair is (a0 = HA's choice, a1 = argmax's), at P = 0.25 [C, the built propensity], in band or out. Elsewhere the band rule is as built. The labels price every HA departure under HA's continuation.
- **Head columns.** Δ_W = W(type a1) − W(type a0) on recruit pairs, plus a typed constant for audit pairs. The head can veto HA where argmax was better: HA proposes, coins price, the head selects, the match certifies. Δ_W's proxy d adds +4.1 ± 1.7 Elo over R, with no fitted per-unit table [M].

**Update, adoption and claims** follow `current_design.md` §3.3–3.6. Every comparison is of single decisions, each followed by play that re-decides after every roll.

**Signal per dollar.**
- **Finding HA cost no box:** batch 1's labels, already paid for, gave the mechanism.
- **A fixed rule is cheapest to read by play:** a match carries 78 times the labels' information per game about one [D from M, `round9_simplify_budget.out`]. H costs $0.98; a head that must find structure costs a $26 round.
- **Audit pairs aim the coins** exactly where HA acts.
- **The reset gets a sharp known answer.** If π_k includes HA, the distilled N, played without HA, must reproduce π_k's composition. Define κ_comp = (0.514 − HHI_N)/(0.514 − HHI_πk), read on 2,000 self-play games of N ($0.05).
  - Its SE is about 0.0038·√2/(0.514 − HHI_πk): about 0.03 at a 0.2 shift [D, the HHI SE scaled to 4,000 sides].
  - A broken distillation (κ_comp ≈ 0) then sits more than 30 SE from a working one (≈ 1). The design's ĉ asks only for "above 0 by 2 SE".

## 4. Regimen

Costs are at the measured host [D, `round9_synthesis_checks.out` §6]. Rows not shown are as designed.

| Stage | Cost | Readout and prediction | Decision per outcome |
|---|---|---|---|
| **R0** offline | $0 | The design's R0, plus HA unit tests, each able to fail: the {5,1,1} sequence; the bound under masks; T → 0 equals argmax; refusal and relocation. Plus a CI replay: on a few recorded `own2` games, HA at T 0.001 reproduces every recruit whose top-two type ratio is ≥ 1.5 | Red: fix before any box |
| **Box A: L, T** | $2.18 | As designed. R's count feedback predicts about −2.85 Elo of second order [D]; C's per-faction readings are reported beside R's | As designed. A low L is first compared with the feedback term |
| **Box A: H** | $0.98 | `own2`+HA on one seat (per-game seat coin, I-side) against `own2`, uncoined, 37,800 decisive games, SE 1.79; J_H − J_T, SE 2.53. Shadow log at every recruit decision of both seats: q_t, HA's and argmax's types, the gap. Reported: departures per side-game in and out of band [U]; the HA seat's turn-1 distinct types and HHI against `own2`'s 1.356/0.514 and the humans' 3.164/0.274; I-herd. J_H is not predicted | **J_H − J_T > 5.1:** composition past the band is worth playing; π_0 = `own2`+HA under ruling (i), and the claim C runs. **Within ±5.1:** the band holds the lever; π_0 as designed, or `own2`+HA if (h) refuses R and J_H > 3.6. **< −5.1:** HA is retired. **Census ≈ 0:** q too peaked for HA to act |
| **C** | $1.13 | Iff π_0 ≠ `own2` | As designed |
| **S1 smoke** | $0.08 + setup | With HA in π_0: I-herd, the audit-pair census, the coin plant on audit pairs (slope 1 per mover) | A gate fails: debug |
| **S1, H1, S2, S3** | As designed | Audit pairs and Δ_W enter every student; their typed mean and v̂ are reported | As designed |
| **S4 reset** | $3.90 + $0.05 | As designed, plus κ_comp, predicted ≈ 1, SE about 0.03 | κ_comp not above 0 by 4 SE: distillation broken, whatever ĉ reads. Short of 1 by 2 SE: N keeps HA in its decode |

H's reading does not depend on L: HA is certified by play alone, so a null L leaves H's decision standing.

**Planted checks per code path:**
- audit-pair labels: the design's coin plant, read per pair type;
- the HA decode: R0's exact sequences and I-herd (HA trains nothing);
- distillation: κ_comp.

**Budget** [D, `round9_synthesis_checks.out` §6 plus H]:
- Box A costs $3.16.
- Through round 2 with C and the E reserves: $57.55, leaving $24.79 of $82.34. Two rounds fit iff o ≤ 0.54.
- The whole path is $5.51 short at o = 0. Round 3 fits iff a host's all-in cost per coin game is at most about $2.62e-5, which is 0.920 times the measured: the 0.934 line moves by $0.98 over the $68.6 of coin batches it scales.

**Toward human strength.** The record's largest gains corrected argmax's reading of an imitation policy (end_turn mass: +229, +193); HA corrects a second such error. Coin rounds, resets and the $1,000 case follow as designed. Nothing measured links `own2` to human ratings.

## 5. What is new, and the nearest work

Searched 2026-10-21 (WebSearch, standard and extended). Finding nothing does not prove absence.
- **Herded argmax.** A deterministic decode of a learned game policy that realises the policy's own mixture over a recurrent option set, with bounded discrepancy and argmax as its T → 0 limit. Not found in game AI or RL. Nearest:
  - herding, which turns moments into deterministic pseudo-samples ([Welling 2009](https://mlanthology.org/icml/2009/welling2009icml-herding); [Chen, Welling & Smola 2010](https://proceedings.mlr.press/r8/chen10a.html));
  - nginx's smooth weighted round-robin, the same recursion with fixed weights ([commit](https://nginx.googlesource.com/nginx/+/52327e0627f49dbda1e8db695e63a4b0af4448b1));
  - penalised sampling, a hand-set factor on tokens already produced ([Keskar et al. 2019](https://arxiv.org/pdf/1909.05858));
  - AlphaStar's statistic z, a sampled human build order conditioning a trained policy ([summary](https://cs.paperswithcode.com/method/alphastar), secondary only);
  - low-discrepancy action selection, which explores by the least-tried actions ([Carden et al.](https://scholars.georgiasouthern.edu/en/publications/low-discrepancy-action-selection-in-markov-decision-processes-3/), abstract only).

  HA needs no training, sampled plan or knob, and stays deterministic.
- **Audit pairs** randomise per step between two policies, as in [Farias et al. 2022](https://arxiv.org/abs/2206.02371). Not claimed as new.
- **The composition closure** (a behaviour statistic as the reset's known answer) is a modest tool, and the collapse is a fact about the network, not a method.

## 6. Weakest point

H's decisive part has no forecast. HA herds toward the humans' mixture, and nothing shows it suits `own2`: Loyalists collapse most (Herfindahl 0.706) and gain least in band; S0(a) shows a coherent change can cost −41 Elo; and if q already conditions on the army, herding counts that twice. A null H costs $0.98 and leaves C ≈ R. HA also reaches only recruits, about 23 decisions per side-game.

## 7. Rulings and decision record

- **(i), new:** may the player decode recruits by herding its own probabilities? No fitted values, no game knowledge. Δ_W and the audit pairs ride on it.
- **(a)–(h) and the budget question:** unchanged.

**Rejected:**
- **A tempered-HA sweep.** No reading predicts any T, and the best of several by match carries the measured winner's curse (+1.95 Elo per near-floor step).
- **Sampling recruits at T = 1.** Its discrepancy grows with the recruits, and it gives up determinism.
- **A frequency penalty.** It is a hand-set knob.
- **Herding end_turn.** It ends turns earlier, against the record's largest lever, and the stop margin is flat at `own1`.
- **Switching L to C.** R is sealed and C ≈ R.

**Deferred:** other option sets. None recurs as recruits do.

**Record note:** *comp* ran 3.4 minutes, over the one-minute guideline; the others ran in 6–28 s.

## Provenance

| Number | Value | Tag | Source or derivation | Used for |
|---|---|---|---|---|
| `own1` turn-1 distinct types; single type (≥ 3); repeat; whole-game HHI | 1.345 (0.007); 0.659; 0.916; 0.514 (0.003) | M | *comp*, 6,402 unhooked S0(a) sides | Mechanism |
| Humans, same (all; rated ≥ +146.5) | 3.164, 0.059, 0.313, 0.274; 3.131, 0.025, 0.248, 0.285 | M | *comp*, 18,586 sides; tiers from `player_ratings/ratings.json` | Mechanism |
| `own2`, same | 1.356, 0.648, 0.913, 0.514 | M | *comp2*, 6,402 S0(b) sides | HA tested on `own2` only |
| Composition distance | TV 0.19–0.49 by faction; 8.4 recruits per side-game | D (assumption unverified) | *shares*: `own2`'s seat; every 4th human file | Scale only |
| Label by d | −0.0179 (0.0063) at ≤ −2; −0.0049 (0.0068) at 0; +0.0313 (0.0082) at 3–4; side-turn 1, ≥ 2: +0.0700 (0.0103) | M | *rules* §1 | §2.3 |
| Count slope inside identity | +0.00267 (0.00048) | M | *rules* §7, clustered sandwich | §2.3; *feedback* |
| C; R; C − R; R plus d − R | +18.4 (3.2); +19.9 (3.1); −1.4 (3.1); +4.1 (1.7) Elo | M | *rules* §2, 20 folds by game hash | HA's in-band part; Δ_W |
| R ∩ C | 1.17 switches, +16.2 (2.3) Elo | M | *rules* §3 | §1 |
| C by faction | Loyalists +1.1 (1.0) … Undead +5.5 (1.4) | M | *rules* §4 | §6 |
| In-band margin slope, d ≥ 1 | −0.028 (0.040) | M | *rules* §6 | Nothing extrapolated |
| R's count feedback | d −1.39 at R's switches; −2.85 Elo (slope SE 0.51) | D from M | *feedback*: shift × slope; limits stated there | L's reading |
| HA bound; synthetic maxima | < \|O\| − 1; 1.10, 1.12 with masks | D | §2.2 proof; *herd* §2–3 | I-herd |
| nginx sequence; tempered departures | a a b a c a a; 0.017 (ratio ≥ 1.5), 0.42 (< 1.05) at T 0.05 | D | *herd* §1, §4 | R0 tests |
| Base | `own2` +22.7 ± 4.4 | M | seed §7 | π_0 |
| R's sealed F | +21.3 ± 3.1 | M | `round9_synthesis_checks.out` §1 | L |
| Recruit near-ties | 4.29 different-type per side-game | M | *rules*; seed §7 | §2.3 |
| S0(a) hook | −41.3 ± 4.4 Elo; −9.38 moves, +1.05 attacks per side-game | M | seed §7; `round9_challenge_hook_flips.out` | §1, §6 |
| Match SEs; lines | 1.79 at 37,800; difference 2.53; 5.1 = 2 × 2.53; 3.6 = 2 × 1.79; false fire 0.023 | D | 347.4/√N | H |
| Match information ratio | 78× labels per game, fixed rule | D from M | `round9_simplify.md` (158.1 / 2), `round9_simplify_budget.out` | §3 |
| Budget | H $0.98; box A $3.16; $57.55 through round 2, $24.79 left; o ≤ 0.54; $5.51 short; host ≤ $2.62e-5 | D from M | `round9_synthesis_checks.out` §6 plus $0.98 (o slope $45.6 per unit; $68.6 of coin batches) | Regimen |
| HHI SE at 4,000 sides | 0.0038 | D | 0.003 × √(6,402/4,000) | κ_comp |
| Numerics flips | 1.76% of decisions; rate 0.53 at gaps < 0.0136 | M (carried) | `round5_simplify_critique_extra.out` §1, via `round7_provenance.md` | I-herd |
| Audit P 0.25; T = 1 only; 2-SE lines | — | C | §3, §7 | — |
| HA's census on `own2`; q against human shares | — | U | H's shadow log | H's reading |
| J_H, J_H − J_T | — | U | H | π_0 |
| Audit pairs' spread; Δ_W on `own2` | — | U | S1 | The head |
| κ_comp | — | U | S4 | Reset |
| Whether the human mixture suits `own2`; human level | — | U | Nothing relies on them | — |
