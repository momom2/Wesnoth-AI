# Round 7 challenge: plans, not ties

**Stance.** The current design learns where the network's top two actions nearly tie; each coin moves one decision, and the gain is bounded by a spread no record measures. The one record that prices alternative play puts large stakes in the choice of a turn's plan, outside the band. Here the unit of exploration is a **learned intent**: vectors pinned into spare memory slots of the frozen `own1`, steering a whole side-turn coherently while the network still picks every action and re-decides after every roll. A coin at each side-turn start picks the intent with a logged propensity, giving an exactly unbiased label of every intent's advantage; a low-dimensional selector learns which intent wins where; a PURE paired match certifies. Nothing in the network is retrained before the first certified gain.

Tags: **[M]** measured (file, script), **[D]** derived (inputs named), **[U]** unknown (when measured), **[C]** a choice whose consequences are derived. New scripts in `analysis/`, outputs beside them: `round7_challenge_turn_stakes.py`, `round7_challenge_power.py`.

## 0. Why change the core

1. **The stakes are in plans.** `round7_challenge_turn_stakes.py` reads the turn-value bench: 199 human positions; the replay's own turn and about four alternatives sampled at temperature 1; 28 playouts each to the end.
   - It keeps only candidates that made no attack, so their turn rolled no combat dice (332 of 966).
   - Their between-candidate variance of the true mean result is **0.0193 (95% 0.0077 to 0.0328), sd 0.139**, luck-corrected [M].
   - Against the human's own attack-free turn, an alternative scores −0.029 (−0.076 to +0.015) on average, with sd 0.178 [M].
   - So sampled alternatives are close to the human turn on average and differ widely from it in both directions.
2. **Per decision.** These turns take 7.03 decisions on average (90th percentile 12) [M, same script §2].
   - If a turn's value were a sum of independent per-decision effects, each deviation would carry at least 0.040 to 0.052 [D].
   - If it is not such a sum, the stakes sit in coherent combinations of decisions.
   - Either way, they are 11 to 33 times the spread the current design is sized to detect (sd 0.0016 to 0.0036) [D]. Its band reaches neither: confident decisions are never coined, and one coin never moves a plan.
3. **Caveats.** Human positions, with `obs8` sampling and playing on at temperature 0.5, not `own1` at argmax. 0.707 of these turns recruit, so trait draws sit in the spread [M]. Picking each position's best alternative on half its playouts gained only +0.020 ± 0.027 on the other half [M, `turn_value_20260925/validation_summary.md`], a per-position pick from about 14 playouts, not a pooled model. `own1`'s near-tie spread is unmeasured, so a measured quantity meets an unmeasured one; §2 orders the work to suit.
4. **Pointers.** Acting more is the largest lever on record (+229 Elo); the better turn takes more decisions in 6 of 7 turn-gap positions (CLAUDE.md); uniform decode knobs are flat (+1.7 ± 12.3). An intent is a state-dependent knob, so a learned stop can arise if the data puts it there.
5. **A channel exists.** `parity3`'s memory changes a quarter of its decisions, and 16 slots play as 64 [M, BACKLOG.md, `parity3_baselines_prereg_20261008`], so slots 48 to 63 are close to spare. For `own1` this is [U]; the default arm does not depend on it.

## 1. The algorithm

### 1.1 Intents

An intent c ∈ {1..K}, K = 8 [C], is a block e_c ∈ R^{16×384}.
- **On a turn under c:** slots 48 to 63 read e_c at every decision of the side-turn. Their writes are discarded, and their natural contents return at the turn's end.
- **Default:** intent 0 is the natural slots, which is `own1` exactly.
- **Every action** is `own1`'s argmax with offset −1.5, re-decided after every roll.

**Fitting.** Hard EM on 200,000 human positions (about 22,000 side-turns), trunk frozen [C]:
- **E-step:** each turn goes to the code under which its actions have the highest log-likelihood.
- **M-step:** gradient ascent on each e_c over its own turns.
- **Initialisation:** natural slot contents taken at random turn starts.

Each code is a conditional policy that reads the realised state, so it cannot assume dice that did not fall. Cost: $1.51 for three iterations at the trainer pipeline's measured 385 forwards and 1,242 training positions a second [D, power §5].

### 1.2 Generation and the label

`own1` plays both sides in the pool at 120 jobs. At each side-turn's first decision, an arm D is drawn with P_0 = 0.5 and P_c = 1/16 [C]. It is keyed by seed and turn, and logged with the turn-start features. For each c:

    T_c = s · R · (1[D = c]/P_c − 1[D = 0]/P_0),   R = z − b(cell) − βᵀS − γ·s·V_t

R is the current design's cross-fitted residual: the full setup cell, the seven luck columns, and the value at the turn start.

- **Claim.** E[T_c | H_t] = Q̃(H_t, c) − Q̃(H_t, 0) under the coined policy, for any such coefficients: each term of R is fixed by H_t or has mean zero given a past containing D, and E[1[D=c]·z/P_c | H] = E[z | D=c, H] (the binary coin's proof with arms).
- **Variance.** About 0.688 × (16 + 2) = 12.4, against 3.67 for a near-tie coin at p = 0.25 [D, power §1].
- **Fog.** The label is the information-state advantage, read from the real game. No world model is involved and no strategy fusion arises.

### 1.3 Selector and player

- **Selector.** For each c, a ridge regression of T_c on x_t, PCA-64 of the turn-start forward's global-token output and mean own-unit token output (PCA fitted on recorded `own1` games) [C], with an unpenalised intercept per code; λ by progressive validation (batch k+1 runs selector k in log-only mode).
- **Rule.** Play argmax_c f̂_c(x_t) if it is above 0, else the default. This is the posterior-mean rule, as in the current design.
- **The player.** π_1 is `own1`, the eight blocks and the selector. It costs no extra forward: the features come from the turn's first forward, and the pinned slots are inputs only.

### 1.4 Why more strength per dollar

1. **Information.** One intent contrast matches the whole near-tie band iff its stakes exceed 3.64 times the near-tie spread [D, power §3: 38.8 turn coins against 152.2 band coins a game, at variances 12.4 and 3.67]. All eight contrasts come from the same turns. The measured turn sd, 0.139, is 39 to 87 times the band's floors.
2. **Dimension.** The band's head estimates 776 directions; the selector estimates 64 per code. The selector's fit-noise sd is 0.014 after one batch and 0.007 after four [D, power §1].
3. **Coherence.** A perturbation moving m decisions alike carries m² of information, against m for independent ones (round 5's case for matches); an intent is such a perturbation, learned from how players vary their turns.
4. **No closure risk in round 1.** The certified player is the selector itself, not a distilled network. Distillation closure on a policy target is unmeasured; agreement closure was 0.58 on another target [M].

### 1.5 A miss

Labels price advantages with both sides on intents for half their turns; the match plays an uncoined `own1`. A shortfall is excursion or inter-turn interaction (current design §3.5). Fallback: the selector at its own half-quantile, changing half as many turns ($0.80); then one-sided coins (half the labels, the opponent as in the match).

### 1.6 Invariants and plants

- **I-pin.** With e_c set to the natural contents, the prompted network reproduces `own1` decision for decision (bf16).
- **I-arms.** Realised arms match the logged P (χ²) and are independent of the pre-turn features and the dice.
- **I-record.** The record shows slots 48 to 63 pinned on exactly the intent turns.
- **Placebo arm.** An identity intent, coined as a ninth arm. Its labels have mean 0 exactly, and the selector must not prefer it beyond 2 SE.
- **P-EM** (the fitter): a planted corpus against a placebo corpus.
  - **Planted.** A seeded half of the side-turns is cut after ⌈m/2⌉ actions by inserting end_turn, which is legal and consistent. The placebo keeps the labels and cuts nothing. EM runs with K = 2 on each ($0.42 [D]).
  - **Readouts.** Agreement with the planted labels: the placebo reads 0.5 in expectation; the planted run must exceed it by more than 4 SE (SE 0.0035 [D]). The "short" code's end_turn log-probability shift at held-out cut points: positive when planted, 0 for the placebo.
  - Broken gradients, wrong slots or swapped codes read as the placebo. The share of the planted gap closed, the channel's capacity, is reported, not gated.
- **P-intent** (the label path), batch 1.
  - **Plant.** z′ = z + δ·Σ_t s_t·(1[D_t = c*] − P_c*)·(u·x_t), with s and D read from the game record by code separate from the label code.
  - **Readout.** Per side, the slope of (T′ − T)/δ on u·x is exactly 1. A wrong sign reads −1, and an arm logged but not played reads 0.
  - **Size.** δ = 0.0071 gives a 4-SE band at Var(u·x) = 1 [D, power §6]. δ is fixed after S0(c).
- **Match path.** The current design's S0(b), shared.

## 2. Regimen

| Stage | Cost [D] | Readout | Decision per outcome |
|---|---|---|---|
| **S0** | $0.29 | Records (done): §0. Box: the match path's mirror null and attack known answer ($0.24, shared); price and logging overhead ($0.05). | Match path outside its window: fix it first. |
| **S1** intents | $3.11 | P-EM; EM at K = 8 ($1.51); diversity read on 800 recorded `own1` games ($0.98): KL from the default, argmax disagreement, and decisions per side-turn in 1,000 self-play games for each code ($0.20). No prediction: no record holds a prompted `own1`. | **P-EM fails:** debug. **No code differs from the default by 2 SE:** the slot channel is too weak. Either refit through an input token with the trunk unfrozen under a KL anchor (a new path with its own P-EM, priced then), or stop at $3.4 and run the current design. **Otherwise:** S2 with the codes that differ. |
| **S2** coins | $2.51 a batch + overhead [U] | **Batch 1:** P-intent, I-arms, the placebo arm, each code's mean advantage (SE 0.0018 a turn), a cross-fit forecast. **Batches 2–4:** progressive fits, then the sealed forecast (SE 2.2 to 4.9 Elo [D, power §2]). | **A plant or the placebo fails:** the label path is broken. **After batch 1, no code above 0 by 2 SE (0.0036 a turn, about 24 Elo if played every turn [D]) and no cross-fit forecast above 0:** stop at $5.9. Intent stakes are then bounded at that line, and the shared generator goes to the current design's S1. **A code below −2 SE:** dropped (propensities are logged). **Otherwise:** run to batch 4. |
| **S3** adoption | $0.80 | π_1 against `own1`, 32,000 PURE paired games on fresh seeds, SE 1.79 Elo. Prediction: the sealed forecast, with miss line 2√(SE_f² + 1.79²). | **Lower bound > 0:** adopt; this reading is the claim. **Null with a positive forecast:** the fallback (§1.5). **Both null:** turn-start features cannot read the intents' stakes; spend the rest on the current design. |
| **S4** round 2 | $5.8–10.8 | Coins on π_1, with the default arm set to the selector's pick; refit; matches against π_1 and `own1`. | Adopt by the same rule. A null round means the repertoire is exhausted: go to S5. |
| **S5** reset | $3.7 | DAgger π_k into the trunk with the own-trajectory recipe ($2.1 [M]), then a match. Then EM on the new base's own turns sampled at temperature 1 ($1.51 plus about $0.1 of games); the bench put such turns within 0.03 of human ones on average (§0). | **Distilled ≥ π_k:** new base; the repertoire now comes from self-play. **Otherwise:** keep π_k and use the new repertoire. |

**Budget** [D, power §5]:
- Round 1 costs $7.03 to $14.56, and a null stops at $5.9.
- Three rounds and one reset cost about $40, leaving about $45.
- Near-tie coins could ride in the same games at no extra game cost, since the two randomisations are independent and both exact. They stay out of round 1 to limit excursion.
- The $1,000 case is argued only after two adoptions within their miss lines: more codes, an unfrozen trunk under a KL anchor, repertoires renewed from each base. Nothing measured links `own1` to human ratings.

## 3. Provenance table

| Number | Value | Used for | Tag | Source or derivation |
|---|---|---|---|---|
| Attack-free turn spread | 0.0193 (0.0077–0.0328); sd 0.139 | Core argument | M | `round7_challenge_turn_stakes.py` §1, 400 position resamples, luck-corrected (raw 0.0201) |
| Alternative minus own attack-free turn | mean −0.029 (−0.076 to +0.015); variance 0.032 | Plausible alternatives; S5's source | M | same script §1 |
| All candidates; with attacks | 0.0282; 0.0299 | In-turn dice add about 0.01 | M | same; round 5 read 0.0286 |
| Decisions in attack-free turns; recruit share | 7.03 (90th pct 12); 0.707 | Per-deviation bound; trait caveat | M | same script §2, `data.jsonl.gz` |
| Per-deviation stakes if additive | ≥ 0.040–0.052 | §0.2 | D | √(0.0193/m), m 12 or 7.03 |
| Split-half gap | +0.020 ± 0.027 | Counter-evidence | M | `turn_value_20260925/validation_summary.md` |
| Near-tie floors | sd 0.0016–0.0036 | Comparison | D | `current_design.md` §2 |
| Label residual | 0.688 of Var(z) | Label variances | M | `round6_provenance_cells.py` |
| Intent label variance | 12.4 a turn | SEs | D | 0.688 × (1/P_c + 1/P_0) |
| Side-turns per side-game | 19.4 | Coin counts | M | `round5_simplify_critique_decision_units.out` |
| Band decisions per side-game | 76.1 | Break-even | D | `current_design.md` §3.1 |
| Break-even | 3.64 × near-tie spread | §1.4 | D | power §3 |
| Per-code mean SE | 0.0018 / 0.0013 / 0.0009 a turn at 1/2/4 batches | S2 lines | D | power §1 |
| Selector fit noise sd | 0.014 (d 64), 0.050 (d 768), one batch | Feature choice | D | power §1 |
| Forecast SE | 2.2–4.9 Elo | S3 line | D | power §2 |
| Oracle scales | 27.6·s_t, 30.4·v z a side-game | Context only | D | power §4 |
| Game price; match | $2.51e-5; 32,000 games, SE 1.79, $0.80 | Costs | M/D | `round6_provenance_prices.out`; 347.4·√(0.847/n) |
| Trainer rates | 385 forwards/s; 1,242 positions/s | EM, diversity | M | BACKLOG.md (round 2 teacher targets; trainer speed 2) |
| EM; P-EM; diversity | $1.51; $0.42; $0.98 + $0.20 | S1 | D | power §5 |
| P-EM SE | 0.0035 | Plant line | D | √(0.25/20,000) |
| P-intent δ | 0.0071 at Var(u·x) 1 | Plant | D | power §6, set after S0(c) |
| Memory changes a quarter of decisions; 16 slots = 64 | — | Channel prior | M (`parity3`) | BACKLOG.md; `parity3_baselines_prereg_20261008` |
| Own-trajectory DAgger | $2.1; +88.6 ± 12.7 | S5 | M | `memory_own_trajectories`, `own_confirm` |
| K 8, P_0 0.5, 16 slots, PCA-64, 200k positions, 3 iterations | — | Design | C | power §1, §5 |
| Channel capacity; code diversity | — | S1 decision | U | P-EM size; diversity read |
| Intent stakes s_t at `own1`; selector predictability | — | All after S2 | U | S2 batch 1, progressive fits |
| Excursion and inter-turn loss | — | Forecast scale | U | S3 against the forecast |
| Slots 48–63 spare in `own1` | — | Intent turns only | U | I-pin, placebo arm |
| Logging overhead | — | Batch price | U | S0(c) |
| Exploitability; human level | — | Nothing | U | — |

## 4. What is new, and the nearest work

Searched 2026-10-20; finding nothing does not prove absence.

**Nearest work:**
- **BRIL** ([Justesen et al. 2019](https://arxiv.org/abs/1907.03046)): a StarCraft build-order policy conditioned on behaviour descriptors from replays, chosen by UCB1 *between games*.
- **AlphaStar's statistic z** ([summary](https://paperswithcode.com/method/alphastar), Nature text not re-read): hand-chosen build orders sampled from human games.
- **Latent language plans** ([Hu et al. 2019](https://arxiv.org/abs/1906.00744)): reported to beat direct imitation in an RTS.
- **Latent primitives under a high-level RL policy:** OPAL ([Ajay et al. 2021](https://arxiv.org/abs/2010.13611)) and SPiRL ([Pertsch et al. 2020](https://arxiv.org/abs/2010.11944)).
- **Options from demonstrations:** Directed-Info GAIL ([Sharma et al. 2019](https://arxiv.org/abs/1810.01266)).
- **A bandit over a policy family:** Agent57 ([Badia et al. 2020](https://arxiv.org/abs/2003.13350)).
- **Theory:** GPI ([Barreto et al. 2017](https://arxiv.org/abs/1606.05312)) and options ([Sutton, Precup & Singh 1999](https://doi.org/10.1016/S0004-3702(99)00052-1)).
- **Labels:** randomised arms ([Boruvka et al. 2018](https://doi.org/10.1080/01621459.2017.1305274)) with AIVAT control variates ([Burch et al.](https://arxiv.org/abs/1612.06915)).

**New, as far as I found:**
- intents as pinned memory slots of a frozen policy, fitted by hard EM (soft prompts in the sense of [Lester et al. 2021](https://arxiv.org/abs/2104.08691), used here as a repertoire);
- per-turn, within-game, contextual selection among them, identified by exact randomised arms with dice control variates and no value function;
- repertoires renewed from the policy's own sampled turns.

**Contradicting evidence:** prompt tuning matches full tuning only at billions of parameters (Lester et al.; this network has 15.6M); latent-skill low-level policies are reported unreliable under distribution shift ([Skill-Critic](https://ai.sony/publications/skill-critic-refining-learned-skills-for-hierarchical-reinforcement-learning)).

## 5. Weakest point

**The channel.** It needs two links, and neither is measured:
- that 16 memory vectors can steer a frozen 15.6M network into turns that differ in value as much as sampled turns do;
- that the 0.139 measured with `obs8` at human positions carries over to intents at `own1` positions.

The small-scale prompt-tuning results argue against the first link. The memory's quarter of changed decisions argues for it. S1 tests the first for $3.11 and batch 1 the second for $2.51, both before anything relies on them. If both fail, the route has cost about $5.9 and the shared generator goes to the current design.

Lesser points: the selector reads only the turn-start observation, while hidden units made positions rankable; half of all turns are coined, on both sides; trait draws sit inside §0's spread.

## 6. Rulings for the user

- **(e)** A per-side-turn choice of a learned intent, after which the network picks every action and re-decides after every roll. Is that within "the network picks actions, never whole turns"? If not, the intents become per-decision proposals arbitrated by the same labels, at K forwards per coined decision (priced then), or the current design runs.
- **(f)** The human corpus defines the first repertoire, with the trunk frozen; S5 replaces it with self-play turns. Imitation, as a source of gains, is parked; this uses the corpus only to define the repertoire.
- **(a), (c)** As in the current design.

## 7. Decision record

**Rejected:** a VAE encoder (collapse risk, nothing gained over hard EM at K = 8); the lineage networks as the repertoire (memory upkeep doubles forwards; near-equal cousins); memory transplants (out of distribution, no known answer); temperature-1 turns as options (not reproducible); a fork census (batch 1 measures intent stakes exactly at about the same price).

**Deferred:** the 768-dimensional selector; both coin kinds in one game; the unfrozen trunk unless S1 finds the slot channel weak.
