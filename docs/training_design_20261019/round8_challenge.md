# Round 8 challenge: the skill axis

**Stance.** The current design takes its direction of improvement from blind exploration: a coin plays the runner-up at near-ties, and a linear head looks for where that helped. No record shows that band's mean advantage is positive, and the near-tie readings on disk lean negative. This proposal takes its direction from information the project owns and does not use: the corpus's player ratings.
- **Conditioning:** a network conditioned on the player's rating learns how stronger humans play differently.
- **Guidance:** classifier-free guidance follows that direction past the top rating.
- **Matches** between two settings of one network read the axis for $0.72.
- **Self-play coins**, the current design's exact label, then decide decision by decision how far past the human range to go.

Tags: **[M]** measured, **[D]** derived, **[U]** unknown (with when it is measured), **[C]** a choice whose consequences are derived. The scripts are in `analysis/`, outputs beside them: `round8_challenge_skill.py` (cited §n), `round8_challenge_stop_census.py` and `round8_challenge_power.py` (cited P§n).

## 0. Why change the core

1. **The corpus holds a measured skill axis.** Bradley-Terry ratings were fitted on 2026-09-05 (`tools/player_ratings.py`, `training/metrics/player_ratings/`).
   - **Spread:** 142 human regulars have at least 30 games each. Their ratings have sd 144.6 Elo, with the 5th, 50th and 95th percentiles at −178, +54 and +253 (§1).
   - **Out of sample:** refit on 80% of the games, the ratings cut the other 20%'s log loss from 0.693 to 0.605 nats. The higher-rated side wins 0.675 of games at gaps of 100–200 Elo and 0.792 at gaps of 200–400 (§3).
   - **Status:** the seed lists rating-conditioned imitation as alive but "unverified" for player names. The names exist for all 17,124 games.
   - **Never tested:** its only attempt failed before training. The 2026-09-05 escrow logs (`box_49875606/box_escrow_20260906.tgz`, `toprated/*.log`) show the fit stopping on a missing scipy, and the matches on a missing `arm.pt` [M].
2. **`own1` imitates the middle of the axis.** The corpus trains on winners' decisions only.
   - Weighted by winner actions, the winner's rating averages +84 Elo.
   - Top-quartile regulars (≥ +147) supply 0.140 of the winner actions; non-regulars, whose ratings say little, supply 0.642 (§2).
3. **The axis points at the record's largest lever.** In 2,000 games between two human regulars, compared within each game (same map, factions and luck), the higher-rated player takes +0.436 ± 0.051 more actions per side-turn over its side-turns 2–4.
   - In the games it went on to lose, it still takes +0.189 ± 0.090 more.
   - The gap grows by +0.269 ± 0.047 actions per 100 Elo of rating gap (§4).
   - Acting more is what the hated −1.5 offset buys (+229 Elo).
   - This is an association: early village income can differ.
4. **The current band barely reaches the stop decision.**
   - When end_turn is on top, the gap to the best act is below w = 0.35 at only 0.140 of decisions (95% 0.113–0.168). The median gap is 0.986 nats (`round8_challenge_stop_census.out`, `parity3`'s replay; `own1`'s census is [U]).
   - Round 7 called the "continue" edits covered by the (end, best act) pair type. For 86% of these decisions they are not.
5. **The route can be closed cheaply.** Whether the network expresses the axis costs $3.32 to learn [D, P§7]. The current design's first reading that can close its route, the interim match, costs $6.64 [D].

Not shown: how much of the human skill gap the network captures, and whether guidance past the top adds strength. S2 measures both before anything relies on them.

## 1. The algorithm

### 1.1 What is learned

**N(a | h, s)** is `own1` plus a rating input s: one extra column of the global token's input projection (`global_proj`, `wesnoth_ai/encoder.py`), initialised to zero, so that N equals `own1` for every s at the start (invariant I-zero).
- **Human data:** both sides of every non-holdout corpus game, with the side's posterior-mean rating from `ratings.json`. Sides played by the built-in AI give no targets.
- **Own trajectories:** `own1`'s own game-sides with their recorded memoryless targets, under a reserved s off the human axis. This keeps round 1's correction of the memory.
- **Update:** maximum likelihood with the existing sequence trainer, own-trajectory mix. 4 M positions [C] at 217 a second [M]: $2.33 [D, P§2].
- At each s, N estimates how players of that rating act from their own observation. There is no strategy fusion and no hidden-state input.

### 1.2 Guidance

At decode, N(·|s_top)'s joint log-prior over the legal candidates is replaced by

    ℓ_γ(a) = ℓ_top(a) + γ·(ℓ_top(a) − ℓ_mid(a)),

followed by argmax and then the reference end_turn offset. Here s_top = +253 (the regulars' 95th percentile) and s_mid = +84 (the level `own1` imitates) [C, from M]. This is classifier-free guidance ([Ho & Salimans 2022](https://arxiv.org/abs/2207.12598)) on a skill condition.

Properties [D]:
- γ = 0 reproduces N(s_top), and the legality mask does not depend on s or γ.
- **The head's structural end_turn bias enters once.** ℓ_γ = (1 + γ)ℓ_top − γℓ_mid, so a bias b(a) shared by both readings appears in ℓ_γ once, not (1 + γ) times.
- **Guidance approximates a setting past the top.** If the logits are linear in s, guidance at γ approximates the setting s_top + 169γ: exactly for one softmax, and up to the factors' normalisers for the factored head.
- **First-order prediction.** Under Bradley-Terry transitivity, **M3(γ) = M2 + γ·M1**. Its SE is 3.55 Elo at γ = 1 and 4.71 at γ = 2, with miss lines of 9.4 and 11.3 Elo (P§4). A miss means the axis bends.
- **Price.** The guided player needs two forwards per decision. Server-bound, the 39.3-ms cycle becomes 72.5 ms: ×1.845 (P§1).

### 1.3 Distillation (S3)

The adopted setting is distilled by the own-trajectory recipe ($2.1 [M]). That recipe is also the measured remedy for memory confusion, which the human fine-tune may bring back.
- **Data:** the setting plays itself.
- **Target:** its memoryless top-16 distribution, with end_turn lowered by its play offset.
- **Before any match:** the closure ĉ and the share ρ̂ of reproduced choices are read on held-out positions.
- **Play:** at the residual offset o(1 − ĉ) [D, `round7_synthesis_power.py` §4], with one forward and s kept.

### 1.4 Self-play past the human range (S4)

On the distilled base N*, a coin at p = 0.25 [C] sits wherever N*'s argmax differs from its guided argmax at γ_next = γ* + 1 [C].
- **Unchanged from the current design:** the label T_i = s_i ω_i R (cell and luck columns frozen forward), exactly unbiased for Q̃(a1) − Q̃(a0), with its plants and invariants.
- **What changes is the band:**
  - **Directed:** the alternative is one step further along the skill direction, not the runner-up.
  - **First moment measured first:** before any coin, a replay census gives n_dis disagreements per side-game [U; a replay like S0(a)'s, $0.04 per setting], and one "always follow γ_next" match against N* ($2.13) gives the band's mean μ_g = (Elo/347.4)/n_dis. Batches are sized from it: resolving μ_g at 2 SE takes 4·3.67/μ_g² labels (P§6).
  - **Reaches confident decisions,** including stop decisions where the axis says act.
- **The gate:** the current ridge head with its typed terms, plus one column, the pair's guidance contrast ℓ_top − ℓ_mid.
- **Each adopted round is distilled again.** The policy moves past the human range only where exact self-play labels and a match say it pays.

### 1.5 Why more signal per dollar

1. **The corpus signal is already paid for.** Skill information costs one fine-tune, not games. The record's gains came from training changes ($2.1 for +88.6 ± 12.7 [M]).
2. **Matches between settings are coherent experiments.** Every decision moves along one direction, so information grows with the square of the decisions moved (round 5's argument). M1 resolves the axis to ±1.79 Elo for $0.72, detecting a transfer fraction f = M1/169 ≥ 0.021 (P§3).
3. **S4's mean is measured before coins.** The near-tie band's typed means lean negative: −0.0018 (−0.0061 to +0.0017) per act at the end/act margin, and ≤ −0.0048 per attack-promoting switch (`round8_provenance_attack_type.out`).

### 1.6 Invariants and plants

**Invariants:**
- **I-zero:** at initialisation, N(·|s) reproduces `own1` at every s (bf16 tolerance).
- **I-guide:** γ = 0 gives N(s_top); s_top = s_mid gives N(s) for every γ.
- **I-mask:** the masks are identical across s and γ.
- **I-seat:** the per-seat hook (S0(b)'s) logs each seat's s; a swapped hook reads −M1.
- **I-rate:** each position's s equals `ratings.json` for that side's id; holdout games and AI sides feed no target.

**Plant: conditioning and guidance path,** before the skill run. 0.5 M positions, $0.27 [D].
- **The modification:** a random 20% of games get the token s = +500. In them, every recruit of one faction's most frequent type U is relabelled V, another type that faction can recruit [C].
- **The prediction,** per held-out U-recruit decision, derived from the same network's s_mid reading: P(V | s_plant) moves toward p_mid(U) + p_mid(V) and P(U | s_plant) toward 0. Guidance at γ = 1 doubles that log-odds shift.
- **The gate:** the closure must be above 0 by 4 SE; its value is reported. A rating input that never reaches the logits reads 0. A wrong-signed guidance term reads a shift away from V.

**Known answer on real data:** M1's sign is predicted positive, because the higher-rated side wins in the corpus (§3). A sign error or a swapped hook reads negative.

**S4** reuses the current design's coin plant on T′ − T, I2, and the attack known answer read through the label code.

## 2. Regimen

| Stage | Cost [D, P§7] | Readout and prediction | Decision per outcome |
|---|---|---|---|
| **S0** | $0 | Done on records (§0) | — |
| **S1** | $2.59 | Plant, then the skill run; I-zero, I-rate, I-mask; holdout loss as a crash barrier only | Plant not above 0 by 4 SE: the path is broken; debug |
| **S2a: M1** | $0.72 | N(s_top) against N(s_mid), one network in the pool, 37,800 games, SE 1.79. Predicted sign: positive. | **Not above 0 by 2 SE:** no axis at this run size. The route closes at $3.32, and the current design runs on `own1`. **Otherwise:** f̂ = M1/169. |
| **S2b: M2, M3** | $5.76 | N(s_top), then guided at γ = 1 and γ = 2, each against `own1` (12,800 two-network games, SE 3.07); decisions per side-turn logged. Prediction: M3 = M2 + γ·M1. | **Best arm's lower bound above 0:** adopt it; this is the PURE claim (false adoption 0.067 over three arms). **A miss:** the axis bends; the best measured γ stands. **No arm above 0 while M1 > 0:** that arm at 0 slots against `own1` at 0 slots ($1.50). If it is above 0 by 2 SE, the memory is the cost and S3 applies. Otherwise the route closes at $10.57. |
| **S3** | $5.73 | ĉ and ρ̂ held out. N* against its guided parent (non-inferiority) and against `own1` (the claim), SE 3.07 each. | ĉ not above 0 by 2 SE: distillation is broken. N* not below its parent by 2 SE: it becomes the base (a 2.8-Elo deficit passes with P 0.18). |
| **S4** | $31.21 (8 batches) | Census, the always-follow match, μ_g, batches sized from it; the current plants; a sealed forecast; adoption against N*; the claim against `own1` | Adopt by lower bound. A null exhausts the band at γ_next. After an adoption: S3, then S4 again. |

**Budget [D, P§7]:**
- **All stages:** $46.02, leaving $38.98. On a 1.41× slower host: $62.96.
- **Exits:** $3.32 at an empty axis; $9.08–10.57 with no arm above `own1`; $14.80 without S4.
- **The remainder** can run the current design's near-tie rounds on whichever base wins. The coins are independent.

**Beyond the human range.** Past s_top + 169γ, each step is certified by self-play labels and a match, not by human data. Nothing measured links `own1` to human ratings, and nothing rests on that link: every claim is read against `own1`.

**The $1,000 case** comes only after two adopted S4 rounds. It holds a permuted-rating placebo (attribution); a second axis read from N*'s own games, labelled by match-measured strength mapped to s through M1 and M2; and more batches.

## 3. What is new, and the nearest work

Searched 2026-10-20; not finding a method does not prove it is absent.

**Nearest work:**
- **Skill-conditioned imitation:**
  - [Maia-2](https://arxiv.org/abs/2409.20553) (chess, skill-aware attention).
  - KataGo's rank-conditioned human network ([extra networks](https://katagotraining.org/extra_networks/)).
  - [UniMaia](https://arxiv.org/abs/2605.27767), which steers a frozen chess policy by player strength. It reports that "for Elo values far below the training distribution, the policy partially reverts toward stronger play" (HTML §7, read). That is evidence that out-of-range conditioning is unreliable.
- **AlphaStar** used MMR to filter and fine-tune its data, not as an input (search summary; the Nature paper not re-read).
- **Guidance and contrastive decoding:**
  - [Decision Diffuser](https://arxiv.org/abs/2211.15657) applies classifier-free guidance to returns.
  - In language models: [Li et al. 2023](https://arxiv.org/abs/2210.15097) and [Sanchez et al. 2023](https://arxiv.org/abs/2306.17806) (neither re-read).
- **Imitation beyond its experts:** [Transcendence](https://arxiv.org/abs/2406.11741) finds that low-temperature imitation can beat its data, but not at its 1500 cap. That contradicts extrapolating from the top of an axis.
- **Weight extrapolation:** [ExPO](https://arxiv.org/abs/2404.16792) (unverified).

**New, as far as I found:**
- guidance along a learned human-skill axis in a game policy, past the top rating, each step's size certified by match;
- the axis's sign as a known answer for the conditioning path on real games;
- self-play coin bands directed by that axis, with the band's mean measured by a match before any coin;
- the within-game measurement that stronger regulars act more per turn, which ties the axis to the end_turn lever.

## 4. Weakest point

**The user's prior: "imitation has diminishing returns and will not pass human strength"** (2026-10-19).
- **S1 and S2 can only lift `own1` toward the top of the human axis.**
- **Past the top, the route rests on two things.** The first is extrapolation, which UniMaia's reversion and Transcendence's 1500 result argue against. The second is S4, which directs the current design's estimator and does not replace it.
- **If `own1` already plays like the top,** M2 reads 0 even when M1 > 0, and the route is worth only S4's directed band.
- **Timing.** S1 is a training run, so it waits for the optimizer discussion (ruling 2026-10-15). The current design's closed-form head might not wait, depending on ruling (e).

**Lesser points:**
- 0.642 of the winner actions come from weakly rated players.
- Ratings are confounded with map and faction choice: the matches certify strength, not attribution.
- The tempo link is an association.
- The stop census is `parity3`'s.

## 5. Rulings for the user

- **(f)** Is rating-conditioned training on both sides of the corpus "imitation's remaining gains, parked" (2026-10-04)? It uses information that more epochs do not.
- **(g)** One zero-initialised input column on the global token: an architecture change that starts as `own1`.
- **(h)** Guided play at two forwards a decision in matches, until S3 distils it.
- **(a)–(e)** as in the current design; S4 needs (a) and (b).

## 6. Decision record

**Rejected:**
- **Top-rated-subset imitation (test 3's design):** it keeps 0.140 of winner actions and discards the axis's contrast.
- **Outcome-conditioned self-play policies.** By Bayes, π(a | h, win)/π(a | h) = P(win | h, a)/P(win | h): exactly the coin label's information. Conditioning later actions on lucky results fails in stochastic games ([Paster et al. 2022](https://arxiv.org/abs/2205.15967)).
- **Forks, with or without common random numbers:** dominated by coins at equal stakes, and common random numbers were measured dead.
- **Arbitration over lineage readings as a core:** its floor is "adopt `own2`" (+16.5 ± 12.3 [M]). That is a $1.50 confirmation match worth running on its own.
- **Targets from the AI's decisions:** that is scripted play.

**Deferred:**
- the permuted-rating placebo ($2.33 plus matches);
- a self-play read of the axis;
- weight extrapolation (the `parity3` → `own1` delta is the memory fix, not skill).

## Provenance

| Number | Value | Tag | Source or derivation | Used for |
|---|---|---|---|---|
| Human regulars; rating sd; percentiles 5/25/50/75/95 | 142; 144.6 Elo; −178/−39/+54/+147/+253 | M | `round8_challenge_skill.out` §1 (`ratings.json`, ≥ 30 games) | s_top, s_mid, the axis |
| Games with 0/1/2 regulars | 6,717/5,076/5,176 of 16,969 joined | M | §1 | Coverage |
| Built-in AI | −177 ± 13 over 1,399 games | M | §1 | AI sides excluded |
| Winner rating, action-weighted mean | +84.1 Elo (quartiles 27/79/145) | M | §2 | s_mid |
| Share of winner actions: regulars; top quartile | 0.358; 0.140 | M | §2 | §0.2, the subset rejected |
| Out-of-sample log loss | 0.6051 against 0.6931 (3,405 test games; 266 exact ties excluded from the bins) | M | §3 (MAP refit, prior sd 200; in-sample refit correlates 1.0 with the tool's ratings) | The axis is real |
| Higher-rated side wins, by gap | 0.531/0.604/0.675/0.792/0.928 (0–50 … 400+) | M | §3 | M1's sign |
| Within-game tempo, all turns | +0.789 ± 0.050 actions per side-turn (2,000 games) | M | §4 (command streams) | §0.3 |
| Within-game tempo, own side-turns 2–4 | +0.436 ± 0.051; +0.189 ± 0.090 in games the higher-rated side lost; slope +0.269 ± 0.047 per 100 Elo | M | §4 | §0.3; tempo logged in S2 |
| Test 3 never ran | fit: missing scipy; matches: no `arm.pt` | M | `box_escrow_20260906.tgz`, `toprated/*.log` | §0.1 |
| Stop decisions in the band | 0.140 at w 0.35 (0.113–0.168); 0.509 at w 1.0; median gap 0.986 nats; 40.5 end-top decisions per `parity3` side-game | M | `round8_challenge_stop_census.out` | §0.4 |
| Pool price; two-network price | $1.912e-5; $1.17e-4 (eval median, upper bound) | M | `round7_provenance_price_tail.out`; `round6_provenance_prices.out` | All costs |
| Worker cycle; server wait | 39.3 ms; 33.2 ms | M | `round7_provenance_price_tail.out` | Two-forward factor |
| Two-forward factor; guided pool game; guided against `own1` | 1.845; $3.53e-5; $1.66e-4 | D | (39.3 + 33.2)/39.3, server-bound; P§1 | M3, S4 |
| Trainer rate; own-trajectory run | 217 positions/s; $2.1 | M | `docs/memory_own_trajectories_prereg_20261011.md` "Measured" | S1, S3 |
| Plant run; skill run | $0.27 (0.5 M positions); $2.33 (4 M positions plus 25-min rebuild) | D | P§2; the rebuild is that prereg's estimate | S1 |
| M1 | 37,800 games, SE 1.79 Elo, $0.72 | D | 347.4/√n; P§3 | S2a |
| M2, M3 | 12,800 games, SE 3.07; $1.50, $2.13 | D | P§3 | S2b |
| Human expected score at Δs 169 | 0.726 | D | logistic, P§3 | Context for f |
| Detection floor of f | 0.021 at 2 SE | D | 2·1.79/169 | S2a's null |
| M3 prediction | M2 + γ·M1; SE 3.55 (γ 1), 4.71 (γ 2); miss lines 9.4, 11.3 | D | logit-linear in s, BT transitivity; P§4 | S2b |
| False adoption, three arms | 0.067 | D | 1 − (1 − 0.0228)³ | S2b |
| S3 acceptance at −2.8 Elo | P 0.18 | D | Φ(−2.8/3.07) | S3 |
| Residual offset | o(1 − ĉ) | D | `round7_synthesis_power.py` §4 | S3 |
| S4 batch | $3.53 per 100,000 games | D | P§6 | S4 |
| Labels to resolve μ_g | 4·3.67/μ_g² | D | σ² 3.67 [`current_design.md` §2] | Sizing S4 |
| Budget; exits; slow host | $46.02; $3.32/$9.08–10.57/$14.80; $62.96 | D | P§7 (the 0-slot diagnostic $1.50) | Regimen |
| Own-trajectory gains | +88.6 ± 12.7; +37.4 ± 12.3 at 0 slots; agreement 0.862 | M | `own_confirm_20261011`, the prereg above | §1.5 |
| `own2` against `own1`; `own1` 64 against 0 slots | +16.5 ± 12.3; −21.7 ± 12.3 | M | `memory_own_round2_20261011/r1*.fit.json`; `own_confirm_20261011/own64_vs_own0.fit.json` | Decision record |
| Near-tie means | −0.0018 (−0.0061 to +0.0017) per act; ≤ −0.0048 per attack-promoting switch | M | `round5_provenance_stop_acts.py`; `round8_provenance_attack_type.out` | §1.5 |
| s_top, s_mid; γ ∈ {1, 2}; 4 M positions; plant share 0.2 and s +500; p 0.25; γ_next | — | C | consequences above | Design |
| Skill transfer f | — | U | M1 (S2a); nothing before it relies on it | Route open or closed |
| Curvature of the axis | — | U | M3 against its prediction | γ choice |
| Memory confusion at s_top | — | U | Shows in M2/M3 (conservative); removed by S3's measured remedy | S3 |
| ĉ, ρ̂ | — | U | S3, held out, before its match | Residual offset |
| n_dis; μ_g | — | U | Replay census; the always-follow match, before S4 | S4 sizing |
| Pool two-network price | — | U | Timed at M2; bounded by $1.17e-4 | Costs |
| `own1`'s human-scale level; exploitability | — | U | Not measured; nothing rests on them | — |
