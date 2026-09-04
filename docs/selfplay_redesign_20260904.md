# Verdict and recommended algorithm — self-play that beats `raw:t0`

*Judge/synthesis pass, 2026-09-04 (Opus workflow: 4 literature agents, 5 designers, 10 critics, 1 judge). Inputs: five designs, ten critiques, four literature surveys, and a repo verification pass (every claim marked ✅ below was checked in the tree, not taken from a critique).*

---

## 0. Summary

**No design in the dossier survives as submitted.** Four fail on arithmetic that is an identity, not a tuning miss; the fifth (CDR) carries the best diagnosis in the dossier and an algorithm that its own information budget cannot pay for.

One lever survives, and it was identified independently by both CDR critics: **the exact combat-outcome enumerator, used as an override on the deployed argmax player, gated by a real match.** The repo pass turned it from "plausible" into "the deployed player provably has zero combat arithmetic" (§2.1). The recommended program — **XOD, exact-outcome override, distilled** — is expert iteration with that override as the expert, and it costs **$2.6 through the first verdict** instead of $9-34.

The load-bearing correction that reorders everything:

| channel | what $1 buys | usable information |
|---|---|---|
| Outcome-graded deviation rollouts (all five designs) | ~3,300 ±1 contrasts | χ² ≈ 6 per dollar at δ=0.03; **$9 ⇒ ~54 χ², i.e. 6-13 scalars** |
| A 400-game match | 3,600 games | one procedure comparison at SE 17 Elo, **unbiased for the quantity that ships** |
| Distillation of a *gated deterministic teacher* | ~1.2M decisions, ~60k corrective labels at a 5% override rate | **noiseless**; label noise is exactly zero |

The deviation channel at this budget buys about a dozen numbers. The metric estimates a dozen numbers directly, for $1.60, with no transfer assumption and no harness-fidelity risk. So: **spend the outcome channel on selecting among procedures, and spend the (noiseless) distillation channel on moving the weights.** Every design in the dossier does the opposite.

---

## 1. Verdict table

| Design | Cost as written | Corrected cost | Corrected gain vs `raw:t0` | Verdict | Reason (one line) |
|---|---|---|---|---|---|
| **FCV** — fork-contrast value | $9.00 | $4.5–22 | 0 Elo; modal outcome is a false family kill | **reject** | Step-0 gates pass on pure noise as an identity (per-parent discordance = 1−p⁴−(1−p)⁴ = **0.875** at p=0.5 against a 0.25 bar), R=1 makes the "oracle" a random top-4 tie-break, and the ≥0.60 ranking gate sits above the Bayes ceiling 0.5+δ. |
| **CPA** — counterfactual preference ascent | $7.80 | $11–13 | −30 to +10 Elo | **reject** | Its Stage-A statistic is identically δ²/2 (unreachable at any plausible δ), so it self-kills and records a *false general finding*; the contradiction rate's null is 0.5 and the accept window contains it; the pairing premise needs CRN, measured dead. |
| **RCTC** — rollout-certified turn commitment | $6.32 | $25–34 | −80 to +20 at D=4; −10 to +70 repaired | **defer** | At the budgeted depth 80–88% of grades come from the flat value head, so it *is* TCS with a longer lever; honest depth is 4.25× the price; the served plan goes stale at the first attack; the "floor is raw:t0" claim fails at a ~42% null commit rate over 7 challengers. |
| **DDPO** — deviation duels | $7.80 | $6–13 | ~0 Elo | **reject** | State-addressed CRN gives *zero* coupling once branches diverge (strictly worse than the counter-keyed path already in the tree); at R=2 the placebo split rate is exactly 0.3125 and f−f₀ ≈ −0.75δ, so the success bar is unreachable in every world; 46% of MOVE constraints point the wrong way at δ=0.03. |
| **CDR** — counterfactual deviation ranking | $9.50 | $8–12 | ~0 Elo | **repair → source of §2** | Best diagnosis in the dossier and the surviving lever (exact enumerator as proposer); the algorithm is not: χ²=54 supports ~6-13 scalars, a frozen trunk makes the update a rank-1 global tilt (`actor_head` is `Linear(384,1)` ✅), class T is a coherent pump on that one slot, and its null control cancels by symmetry so it has zero power against the bias it names. |

### 1.1 Repo facts established in this pass (all verified in the tree)

| # | Fact | Where |
|---|---|---|
| ✅1 | `COMBAT_TARGET_ALPHA = COMBAT_TYPE_ALPHA = 0.0` — **the deployed `raw:t0` player has no combat arithmetic in its priors**, by the 2026-08-06 "every hand-placed prior nudge defaults OFF" ruling. `quarantine/INVENTORY.md` §6.25: "no isolated measurement." | `wesnoth_ai/constants.py:154-155`; `docs/techniques.md` §3.5 |
| ✅2 | `enumerate_attack_outcomes(..., advancement_choice="uniform")` measured at **0.114 ms/call**, 16 outcomes on a level-1 fight — ~1% of a ~10 ms decision. Per-decision exact enumeration is affordable. | microbenchmark, this session |
| ✅3 | **`az_loop._probe` never passes `--raw-temperature-a/b`** (called with `sims=0` at `:521`), and `_build_player` falls through to the legacy factored sampler when `raw_temperature is None`. Every az-leg raw pin, **including 11-0-29**, measured *sampler vs sampler* — a player measured at −412 Elo from argmax on the same weights. | `tools/az_loop.py:178,521`; `tools/elo_eval_game.py:183-186` |
| ✅4 | **No `raw:t0` vs `raw:t0` game has ever been played.** Procedure pairs across all of `eval_games/`: (raw:t0, raw)×40, (mcts:32, raw:t0)×40, (mcts:32, raw)×20. | `eval_games/*/*.json` |
| ✅5 | CRN is dead under a pre-registered rule: median **0** matched downstream fight identities under both strict and loose keys; only 28% of pairs share even one. | `docs/archive/credit_assignment_design_20260817.md:201-213` |
| ✅6 | `rollout_outcome` and `turn_search`'s projection both pick actions with `_sample_prior_idx` — **sampling from the full prior**. Every rollout the project has computed estimated V for a policy ~412 Elo below the reference. | `tools/value_grounding.py:136,149`; `tools/turn_search.py:196,254,272` |
| ✅7 | Throughput: raw-vs-raw **40 games in 2 min** at `--jobs 10`, $0.334/h ⇒ **3,600 games/$ including process startup**, 1,200 games/h. The brief's 3,000 games/$ is within 20%; the 9,000 games/$ implied by "12 s at 10 concurrent" is the un-amortised steady-state figure. Wall clock is what bills. | `docs/box_specs.md:135-160` |
| ✅8 | `PRIOR_BIAS_END_TURN_MINI` is **mini-category only** — not applicable to ladder maps without a small edit. | `wesnoth_ai/action_sampler.py:686-704` |

---

## 2. XOD — exact-outcome override, distilled

### 2.1 The information source

Two, both outside the network, and neither is the prior.

**S1 — the exact per-attack outcome distribution.** `tools/combat_outcomes.enumerate_attack_outcomes` returns the exact probability over combat outcomes (HP, death, slow, poison, petrify, advancement chain), computed from the WML unit stats, terrain defense, time of day and weapon specials, and verified bit-exact against Wesnoth's own `[mp_checkup]` oracle on 731/731 strikes. It reads no network output: **zero the model's weights and the ranking it induces is unchanged.** The deployed `raw:t0` player has no combat arithmetic of any kind in its priors (✅1), and the seed learned human *move choices* for one epoch, not fight expectations. This is the one place the network provably lacks information that the environment can hand it for 0.114 ms (✅2).

**S2 — realized game outcomes**, spent only where its information budget is adequate: **selecting among a handful of procedures**, and gating every weight change. A 400-game match costs $0.11 and carries SE 17.4 Elo; an 800-game match costs $0.22 and carries SE 12.3 Elo. Evaluation is one to two orders of magnitude cheaper than the block it gates, which is the price inversion RCTC identified and nobody exploited.

> **rejected: fitting a state-conditional ranking function from outcome-graded deviations, because** 30,000 ±1 contrasts at δ≈0.03 carry χ² ≈ 54 — enough to determine ~6-13 scalars, not a 15M-parameter function. This is the arithmetic that kills FCV, CPA, DDPO and CDR simultaneously.
>
> **rejected: per-site certification or abstention gates on rollout contrasts, because** R ≈ 2/δ² ≈ 800–5,000 replicates per site is two to three orders of magnitude outside budget, and the affordable version (R = 1–2) selects on noise: at δ=0.03, 46% of DDPO's emitted preferences point the wrong way.
>
> **rejected: common random numbers under any keying, because** Q8 measured it dead with a pre-registered rule (✅5). **State-addressed keying is worse than doing nothing**: two branches never share a state fingerprint after they diverge, so it removes even the weak index-alignment the counter-keyed path provides, while touching the file the bit-exact replay contract rests on.
>
> **rejected: pre-measuring per-class deviation means before running a procedure gate, because** the gate costs $0.11 and is unbiased for the quantity that ships, while the measurement costs $3+ at 50% power against its own central hypothesis.
>
> **rejected: activating the analytic `combat_oracle` alphas instead of the exact enumerator, because** the analytic oracle hardcodes hit chance at 0.5, ignores terrain defense, time of day, specials and abilities, while the exact enumerator consults all of them at 0.114 ms/call. The better object is also the affordable one.

### 2.2 The teacher: `raw:t0+xc(m)`

A deterministic wrapper on the deployed player. At each decision it enumerates legal actions with priors exactly as `raw_player.RawPolicyPlayer` does, then:

1. If the policy's argmax is **not** an attack → play the argmax (variant A5 excepted, below).
2. Otherwise score the argmax attack and the top-K attacks by prior with the **expected material swing**

   `swing(a) = Σ_o P(o) · [ (v_d(before) − v_d(o)) − (v_a(before) − v_a(o)) ]`,  `v(u) = cost_u · hp_u / max_hp_u`, dead = 0, advanced = new type's cost × hp/max_hp.

3. Play the best-scoring attack if `swing(best) − swing(argmax) > m`; otherwise play the argmax.

There is exactly one free scalar, `m`, and it is **swept in games** (§4). The score's functional form has no free parameters given the linear HP-to-gold convention, which comes from the game data (`Unit.cost` ✅) and gets a `docs/design_constants.md` entry rather than a tuned value.

Four variants enter the sweep alongside `m = ∞` (bit-identical to `raw:t0`, the null control):

| arm | fires when | touches turn structure? |
|---|---|---|
| A0 `m=∞` | never | no — must land at 0.500 ± SE or the harness is broken |
| A1–A3 OVERRIDE `m ∈ {12, 6, 3}` gold | argmax is an attack, a better attack exists by > m | no (attack→attack only) |
| A4 VETO `m=6` | argmax attack's own swing < −m | no (falls back to best non-attack) |
| A5 TAKE-KILL `m=12` | argmax is *not* an attack, some attack has swing ≥ m | **yes** — K reported separately |

**Honest weakness, stated once and not softened:** the override is exact about the fight and blind about everything after it — zone of control, retaliation from the enemy's other units, the hex the attacker ends on, healing, next turn's time of day. It can be arithmetically right and strategically wrong. That is precisely what `m` is for (a large margin fires only on trades the arithmetic calls lopsided, where positional subtlety is least likely to dominate) and precisely why nothing is distilled before an 800-game match says it wins.

### 2.3 One training iteration, step by step

Round k begins with accepted weights θ_k and the margin m_k selected in the previous sweep.

1. **Gate the teacher.** 800-game paired match, `raw:t0+xc(m_k)` on θ_k versus `raw:t0` on θ_k, sides alternated, ladder maps, PURE, `--raw-temperature-b 0`. Cost $0.22. If the lower bound does not clear 0.5, stop — standing rule 2 is unsatisfied and nothing is distilled.
2. **Generate.** 4,000 games with the teacher on both sides: 70% fresh ladder starts, 30% human midgame splices via `midgame_starts.sample_midgame_start` (Go-Exploit's state-coverage graft; contact is guaranteed and the states are where the seed learned its competence). Record every decision: `GameState`, the enumerated legal list, the teacher's action, the base argmax, the swing gap. **The labels are noiseless** — the teacher is a deterministic function of the state and the frozen θ_k.
3. **Emit targets on the disagreement set only.** Where teacher action ≠ base argmax: a one-hot `MCTSExperience` (the existing `visit_counts` tuple schema carries factored actor/type/target/weapon indices ✅), `policy_weight = 1`. On an equal-sized random sample of *agreement* decisions: an anchor tag, no target.
4. **Update.** `L = CE(one-hot | disagreement) + η · KL(π_θk ‖ π_θ)` over the full legal set at anchor states. **No value loss, no auxiliary heads, no bootstrap, no consistency term.** `value_coef = 0` **and** the value head plus its input projection are `requires_grad_(False)`, asserted by a test. Trunk, embeddings and the four policy heads are trainable.
5. **Step control by behaviour, not loss.** Bisect η and epochs so that held-out **teacher agreement on disagreement decisions** rises while the held-out **argmax flip rate on agreement decisions** stays under 2%. Both are measured by replaying stored held-out decisions through the new weights — free, and split by *game*, never by state.
6. **Gate the student.** 800-game paired match θ_{k+1} at `raw:t0` versus θ_k at `raw:t0`, PURE. Promote only on a 95% lower bound above 0.5. Then an **800-game confirmation on a disjoint seed set**, which is the reported claim.
7. **Repeat.** The teacher is rebuilt on θ_{k+1}, so the override now sits on a stronger base; `m` is re-swept every second round because the base's own attack ranking has moved. This is policy iteration, and every step of it is adjudicated by games.

### 2.4 Data flow

```
θ_k ──► raw:t0 (argmax over enumerate_legal_actions_with_priors)
          │
          ├── argmax is an attack ──► enumerate_attack_outcomes (exact, 0.114 ms)
          │                            └─► expected material swing ─► override if gap > m
          └── else ──────────────────► argmax
                     │
     teacher action ─┴─► [gate: 800 games vs raw:t0]  ── fail ──► STOP
                     │
                     └─► 4,000 games (70% ladder / 30% human splice)
                          ├── disagreement decisions ─► one-hot CE targets
                          └── agreement sample ───────► KL anchor to θ_k
                                        │
                                   trainer (value_coef 0, value head frozen)
                                        │
                                       θ_{k+1} ─► [gate 800] ─► [confirm 800, disjoint seeds] ─► promote
```

### 2.5 Tempo / `end_turn`

Round 1 **does not touch it**, deliberately. Arms A1–A4 are attack→attack re-rankings, so no mechanism can shift `end_turn` mass; the KL anchor holds the act/end-turn axis at θ_k; and the targets are one-hot on named actions rather than a normalised 350-way distribution, so the truncation artifact that stripped 2–9% of attack mass per decision has no channel. Arm A5 is the single arm that can change turn structure, and it reports K separately.

If the ladder produces a winner and budget remains, tempo is tested **the same way as everything else**: as a config scalar (an `end_turn` logit offset generalised off the mini-category gate, ✅8), swept at 200 games per point and confirmed at 800.

> **rejected: a tempo term in the loss, in the value, or in search, because** that mechanism is on the failed list four times (+0.44 bonus, value centering, K-median tripwires, hand-coded prior bias) and its failure is invisible until the leg ends. A config scalar gated by an 800-game match is the same knob with an honest instrument.
>
> **rejected: making `end_turn` a standing deviation-challenger class in a pooled gradient (CDR class T), because** ~1/3 of contrasts would name the same single actor slot while every unit action appears a handful of times, and `actor_head` is `Linear(384,1)` = 385 parameters ✅ — the aligned-versus-cancelling asymmetry that decided VG2 (280× per-state) and VG3 (98.8% of the update) would make the update a global end_turn tilt derived from a noisy mean.

### 2.6 The value function

**Not learned in round 1.** `value_coef = 0`, head frozen, no gradient path into the trunk, and nothing reads it at play time. The measured pathology (value gradient 94–99% of the update direction, winner/loser components at cosine −0.9) becomes 0.000 by construction rather than by coefficient tuning, and the arm-T channel — training drifting the head's judgements on *imagined* states that no real-state telemetry measures — has no path to strength because no procedure consults the head.

**If a value function is ever needed** (only if the program reaches turn-level rollout selection, §4 rung 4), it is trained on **its own trunk**, offline, on human turn-boundary states from the 17,124 winner-labelled games, with **one boundary sampled per game per epoch** (AlphaGo 2016's fix for within-game correlation), and it never shares a gradient with the policy.

> **rejected: freezing the trunk during distillation (CDR's choice), because** `actor_head` is 385 parameters and `type_head` is 1,540 over a frozen 384-d representation, so a frozen-trunk update is a rank-1 global tilt and cannot express state-conditional combat arithmetic. It is safe to unfreeze here precisely because `value_coef = 0` removes the mechanism that poisoned the trunk.
>
> **rejected: distilling on all decisions, because** the teacher equals the base on ~95% of them; a one-hot target there is self-distillation into the mode, and argmax is invariant under sharpening, so it consumes capacity for exactly zero Elo.

### 2.7 Why the teacher is provably not the prior

Three independent checks, all cheap, all pre-registered.

1. **Algebraic.** The override's ranking is `Σ_o P(o)·Δgold`, a function of unit stats, terrain, time of day and the combat rules. Every failed teacher was a deterministic function of the network's own forward pass — visit counts `f(prior,V)`, Gumbel completed-Q `f(prior,V)`, TCS boundary grades `f(V)`, the VG consistency term `f(V)` against itself — which is why KL(target‖prior) measured 0.002. This one is `f(game rules)`.
2. **Empirical, logged.** The **override rate** — fraction of decisions where teacher action ≠ base argmax — is the direct analogue of the KL instrument that read 0.002. If it is zero the teacher *is* `raw:t0` and there is nothing to gate. **Pre-registered:** if the strength-positive margin's override rate is below 0.5% of decisions, the arm cannot move Elo and the sweep must extend to smaller margins or the family is dead.
3. **Adjudicative.** The gate is an 800-game match. A teacher equal to the prior scores exactly 0.500 and fails.

### 2.8 What was grafted from the runners-up, and why

| graft | from | why it composes |
|---|---|---|
| The deterministic-player diagnosis (J\* is piecewise constant; only argmax flips are visible) and the exact enumerator as a *proposer* | **CDR** | The diagnosis fixes the objective's shape; both CDR critics independently named the enumerator the highest-value item in the dossier. |
| Gate the operator as a *player* before distilling anything | **FCV** | The reproducible process error behind ten failed legs. Here it is rung 1, and it costs $0.22. |
| Any future value net gets its own trunk; the `value_grounding` rollout-policy defect | **FCV** | Direct answer to measured fact (e) and to ✅6. |
| `value_coef = 0` as the cleanest isolation of the policy channel; games-graded step acceptance; "argmax-invariant reweighting is worth zero Elo" | **CPA** | Removes the 94–99% channel by construction and forces mode-changing targets. |
| Abstain-to-the-incumbent as the shape that gives an operator a floor; the evaluation price inversion | **RCTC** | Realised as "the override fires only above margin m", which makes A0 an exact null control. |
| Correct match power arithmetic; matches as the tripwire rather than proxies | **DDPO** | 40 games at SE ~55 Elo cannot see any effect this program claims. |
| Human midgame splices for state coverage; one-boundary-per-game for any value fit; ExIt structure with a cheap expert; KataGo's cost multipliers deferred until a teacher qualifies | **literature digest** | Coverage and sequencing, not new signal. |

---

## 3. The ten known failure modes

| # | Failure | Why XOD does not repeat it |
|---|---|---|
| 1 | **K-collapse** (turn ends after 1–4 actions) | Arms A1–A4 are attack→attack re-rankings; the KL anchor pins the act/end-turn axis at θ_k; A5 is the only arm that can move turn structure and reports K separately. |
| 2 | **Attack mass stripped from targets** | Targets are one-hot on named actions, not a normalised 350-way distribution, so truncation has no channel; and the teacher's only edits are *within* the attack class. |
| 3 | **Invisible erosion** (proxies green, play worse) | No proxy authorises anything. Every weight change is gated by an 800-game match ($0.22) plus a disjoint-seed confirmation, at ~15% of the block cost. |
| 4 | **Prior self-distillation** (KL(target‖prior) ≈ 0.002) | The ranking contains no network output (§2.7-1); the override rate is logged and pre-registered (§2.7-2); a teacher equal to the prior scores 0.500 and fails its gate (§2.7-3). |
| 5 | **Value gradient owning the trunk (94–99%)** | `value_coef = 0` and the value head is frozen with no gradient path. The share is 0.000 by construction. |
| 6 | **Value head flat within a turn / weak early** | Nothing in the loop reads the value head — not the teacher, not the loss, not the deployment procedure. |
| 7 | **Search never leaves the side-turn** | There is no search. The teacher's information is exact arithmetic about a fight; the outcome information comes from complete games in the gates. |
| 8 | **Bootstrap / consistency self-distilling (VG1-3)** | There is no bootstrap term, no consistency term and no target network. |
| 9 | **Selection on noise / winner's curse** | The 200-game screen makes no claim; every accept requires an 800-game confirmation on a *disjoint* seed set. Max-of-6 at SE 0.0354 inflates the reported win rate by ≈ +0.045 (+31 Elo) — that is why the confirmation is mandatory and not optional. |
| 10 | **CRN assumed alive when it is measured dead** | No pairing of dice is used or needed; all match variance is priced as unpaired binomial (✅5). |
| 11 | **Teacher distilled before it was ever gated as a player** | Rung 1 is a gate and the whole program stops there if it fails. |

---

## 4. Experiment ladder (pre-registered)

Conventions throughout: PURE (decisive games only); ladder maps; sides alternated; `--raw-temperature-a 0 --raw-temperature-b 0`; buy games with 5% headroom over the required decisive count and pre-register the minimum. Rates from ✅7: **3,600 games/$, 1,200 games/h at `--jobs 10` on a $0.334/h 4090.** Power: SE(p) = √(p(1−p)/n); dElo/dp = 695 near p=0.5.

| n (decisive) | SE(p) | SE(Elo) | bar p ≥ 0.535 (one-sided α) | power @ p=0.55 (+35) | power @ p=0.58 (+56) |
|---|---|---|---|---|---|
| 200 | 0.0354 | 24.6 | 0.16 | — (screen only) | — |
| 400 | 0.0250 | 17.4 | 0.081 | 0.73 | 0.96 |
| 800 | 0.0177 | 12.3 | **0.024** | **0.80** | **0.994** |
| 1,600 | 0.0125 | 8.7 | 0.0026 | 0.89 | >0.999 |

### Rung 0 — instrument repair and the missing baseline. **$0.20, 40 games, ~35 min.**

- Ship the one-line `az_loop._probe` fix (✅3) and re-annotate `docs/archive/az_leg_20260903.md`: the 11-0-29 pin measured the legacy sampler on both sides, a player measured at −412 Elo from argmax on the same weights, so it says nothing about the argmax players.
- Play **40 games `raw:t0`(seed) vs `raw:t0`(seed)** — never done (✅4). Report decisive rate, median/mean turns, mean decisions per game, and the side-alternated null win rate.
- Confirm the 0.114 ms/call enumeration rate on the box.
- **Kill:** decisive rate < 0.90. Reading: `raw:t0` self-play stalls, so **every** play-out-based number in this document and in the dossier is mispriced and a truncation rule must precede anything else.

### Rung 1 — the teacher gate. **$0.55, 2,000 games, ~1.7 h.**

- **Screen:** 6 arms (A0–A5, §2.2) × 200 games = 1,200 games, $0.33. The screen makes no claim; its job is to pick `m`. A0 must land at 0.500 ± 0.0354 or the harness is broken and the rung stops.
- **Confirm:** the single best-scoring arm at **800 games on a disjoint seed set**, $0.22.
- **Success:** confirmation p ≥ 0.535 (≥ 428/800), one-sided α = 0.024, point estimate ≥ +24 Elo. **Power 0.80 against +35 Elo, 0.994 against +56 Elo.**
- **Also reported, not gating:** override rate, enumeration bail rate, K median (mandatory for A5), and the swing-gap distribution at the winning margin.
- **Kill:** confirmation p ≤ 0.50. Reading: an exact, parameter-free re-ranking within the attack class does not beat the seed's own attack ordering — **the seed's combat ranking is not the headroom.** That also lowers the prior for every rollout-graded operator whose candidate set is the prior's top-k, and it costs $0.75 total to learn.
- **Second kill:** best arm's override rate < 0.5% of decisions. Reading: the override never fires; extend the margin grid downward once, then stop.

### Rung 2 — distillation. **$1.66, 4,000 generated + 1,600 gate games, ~4.7 h.** Runs only if rung 1 confirms.

- Generate 4,000 teacher-vs-teacher games (70/30 ladder/human-splice), $1.11, 3.3 h. At a 5% override rate this is ~80,000 **noiseless** corrective labels.
- Train per §2.3 steps 3–5. GPU-only, ~20 min, $0.11.
- Gate 800 games θ₁ vs seed ($0.22), then confirm 800 on disjoint seeds ($0.22). **The confirmation is the claim; both numbers are recorded in the same CSV row so the pair is auditable.**
- **Success (the headline, and the only claim):** θ₁ at `raw:t0` beats the seed at `raw:t0` with p ≥ 0.535 over ≥ 800 decisive confirmation games. Power 0.80 against +35 Elo.
- **Kill:** gate p ≤ 0.50. The diagnosis is *measured, not guessed*, by the held-out teacher-agreement number: high agreement + a losing gate ⇒ the gain was procedure-only and does not live in weights; low agreement ⇒ capacity/data, and the remedy is 4× more games (linear, $4.4) exactly once.

### Rung 3 — tempo, as a procedure. **$0.44, 1,600 games, ~1.3 h.** Runs if rung 1 or 2 kills, or if budget remains.

Generalise `prior_bias_end_turn` off the mini-category gate (✅8, ~4 lines), sweep the offset over {−1.5, −0.75, 0, +0.75} at 200 games, confirm the best at 800, same bars. This is the **only** sanctioned re-proposal from the failed list, and only because the failed version was never applied on ladder and never gated by an 800-game match against `raw:t0`.

### Rung 4 — deferred, priced, not funded now: the measurement the whole rollout family needs. **$1.33, ~4,800 game-equivalents, ~4 h.**

Before any turn-level searcher is built: measure the **distribution of Q^π gaps δ** between the base's own side-turn and its best of 4 sampled alternatives — 60 boundary states × 4 candidates × 40 unpaired rollouts. SE per candidate = 0.079; SE per difference = 0.112, so it resolves δ ≥ 0.22 at 2σ. **Pre-register on the tail, not the mean** (R=40 cannot see the mean, and a certify-or-abstain operator harvests only the tail): report the fraction of side-turns with |δ| ≥ 0.25. From it, R = 2/δ² and therefore the true price of RCTC/DDPO/CDR follows.

> **rejected: building any turn-level rollout searcher before rung 4 runs, because** the family's price is currently unknown by two to three orders of magnitude, and RCTC's own corrected cost ($25–34) sits above the leg cap on an unmeasured assumption.

### Budget

| rung | $ | games | wall | cumulative |
|---|---|---|---|---|
| 0 | 0.20 | 40 | 0.6 h | 0.20 |
| 1 | 0.55 | 2,000 | 1.7 h | 0.75 |
| 2 | 1.66 | 5,600 | 4.7 h | **2.41** |
| 3 | 0.44 | 1,600 | 1.3 h | 2.85 |
| 4 (deferred) | 1.33 | 4,800 ge | 4.0 h | 4.18 |

Rungs 0–2 fit in **$2.4 and one 8-hour box**, a quarter of the leg cap, leaving room for a second policy-iteration round and rung 3 inside $10, and ~$50 of the $60 credit intact.

### Honest expectation

P(rung 1 confirms) ≈ **0.40**. For: the deployed player provably has no combat arithmetic (✅1); games are decided by leader kill; a large margin isolates the lopsided trades where the heuristic's positional blindness matters least; A0 is an exact null control. Against: human attack choice is good, so the imitation prior may already encode most of it; a greedy material chooser is a known-weak Wesnoth player; the override rate at a strength-positive margin may be below the instrument.

P(rung 2 confirms | rung 1 confirms) ≈ **0.55**. Distilling a deterministic teacher from noiseless labels usually recovers most of it, but the student must approximate exact combat arithmetic from encoder features and its errors are unbounded where the teacher's were not.

**Unconditional: P(a stronger `raw:t0` policy) ≈ 0.22, median gain conditional on success ≈ +30 Elo, P(clean kill for under $1) ≈ 0.55.** That is a modest program. Every alternative in the dossier has a lower probability at 3–13× the cost, and its kill is 5–20× more expensive.

---

## 5. Code plan

Ordered; items 1–2 ship immediately and are independent of everything else.

| # | File | Change | Tests |
|---|---|---|---|
| 1 | `tools/az_loop.py` | One line: `_probe` appends `--raw-temperature-a 0 --raw-temperature-b 0` when `sims == 0` (✅3). Plus a re-annotation of `docs/archive/az_leg_20260903.md`. | `tests/test_az_probe_procedure.py` — the built command carries the flags at sims 0 and not at sims > 0. |
| 2 | `tools/value_grounding.py` | Add `GroundingConfig.rollout_policy = "argmax"`; route `rollout_outcome`'s action choice off `turn_search._sample_prior_idx` (✅6). Old behaviour stays reachable by config for provenance. | `tests/test_value_grounding_policy.py` — with `"argmax"` the rollout's chosen index equals `priors.argmax()`; with `"sample"` it does not (fail-before/pass-after). |
| 3 | `wesnoth_ai/material.py` **(new, ~60 lines)** | `unit_value(unit)` = `cost × hp/max_hp` (dead = 0); `expected_swing(dist, att, dfd)` over an `OutcomeDistribution`, resolving the advanced type's cost. Parameter-free. | `tests/test_material_swing.py` — reuse the thief-backstab construction in `tests/test_swap_detector.py`: a strictly-dominating attack has strictly greater swing; a fight the attacker always loses is negative; the exact swing matches a large-N sampled estimate within tolerance. |
| 4 | `tools/xc_player.py` **(new, ~140 lines)** | `XcOverridePlayer`, same duck type as `RawPolicyPlayer` (`select_action`/`drop_pending`/`drop_last_pending`, `trainable = False`). Config-driven: margin, top-K attacks scored, veto on/off, take-kill on/off. Optional record sink so gating and generation share one code path. | `tests/test_xc_player.py` — at `m=∞` bit-identical to `RawPolicyPlayer(t=0)` on a fixed seed (the A0 control, and proof the wrapper adds no drift); a stub swing function that prefers a known non-argmax attack flips the played action; an enumeration bail (`None`) falls back to the argmax and is counted; the override never fires when the argmax is not an attack (except in take-kill mode). |
| 5 | `tools/elo_eval_game.py`, `tools/run_elo_batch.py`, `tools/eval_procedure.py` | `--xc-a/--xc-b` spec strings (`m6`, `m6veto`, `m12kill`), one branch in `_build_player` beside the `raw_temperature` branch, a guard that `--mcts-sims` is 0, and procedure tags `raw:t0+xc<m>[v|k]`. | Extend `tests/test_eval_procedure.py` — the new tags round-trip; an outdir mixing `raw:t0` and `raw:t0+xc6` games is refused by the existing mismatch guard. |
| 6 | `tools/xc_generate.py` **(new, ~180 lines)** | Generation driver through `tools/actor_pool.py`: teacher on both sides, 70/30 ladder/`midgame_starts` mix, emits one-hot `MCTSExperience` records on disagreement decisions and anchor-tagged records on a matched agreement sample. Shards written incrementally. | `tests/test_xc_generate.py` — a synthetic game yields exactly one target per disagreement decision and zero per agreement decision; the anchor sample size matches the disagreement count; factored index identity is preserved end to end. |
| 7 | `wesnoth_ai/trainer.py` | `TrainerConfig.freeze_value` (value_coef 0 **and** `requires_grad_(False)` on the value head + input projection, asserted) and a `KL(π_θk ‖ π_θ)` anchor term over the full legal set on anchor-tagged experiences. Reuse `_mcts_factored_policy_loss` for the one-hot CE — no new loss code, so the search-time/train-time index-basis contract that `_OOB_INDEX_EVENTS` guards is preserved. | `tests/test_xc_distill.py` — one step on a synthetic one-hot target moves the argmax at that state to the target; the value head's params have `grad is None`; the anchor is exactly zero at θ = θ_k; the anchor-state KL stays bounded after a step that flips a disagreement state's argmax. |
| 8 | `tools/xc_loop.py` **(new, ~220 lines)** | Driver shaped like `az_loop.py`: gate → generate → train → gate → confirm → promote. One CSV row per round: override_rate, bail_rate, disagreement_labels, heldout_teacher_agreement, heldout_agreement_flip_rate, k_median, class_mass_delta_{attack,move,recruit,end_turn}, gate_wdl/elo/se, confirm_wdl/elo/se, dollars, wall_seconds. HF escrow after each stage. **A wall-clock and dollar ceiling that stops at a stage boundary and tears the box down** — the VG leg paid ~8 h idle for the lack of one. | `tests/test_xc_loop_budget.py` — a stub stage that overruns trips the ceiling at the boundary and does not start the next stage. |
| 9 | `configs/xc.json` **(new)** | Every knob: margin grid, top-K, veto, take-kill, generation mix, anchor ratio, epochs, screen/gate/confirm game counts, seed sets. Nothing hard-coded (config-first principle). | — |
| 10 | `docs/xod_spec.md` **(new)** | The pre-registration: predictions, bars, kills, cost table — committed **before** the box is rented. | — |
| 11 | `docs/design_constants.md` | The linear HP-to-gold convention in `unit_value`; the 800-game bar's power arithmetic; why the screen is 200 and the claim is 800; the max-of-6 selection bias (+0.045 win rate) that makes the confirmation mandatory. | — |
| 12 | `CLAUDE.md`, `BACKLOG.md` | Status block and NEXT ACTIONS. | — |

Roughly 600 new lines and three one-to-four-line edits, against 1,000–1,700 for each dossier design. Nothing touches `tools/wesnoth_sim.py` — **the RNG derivation is not modified**, so the bit-exact replay-export contract is untouched by construction rather than by a guard.

---

## 6. What would prove the whole proposal wrong

Ordered by how early it fires.

1. **Rung 0: `raw:t0` self-play does not terminate decisively (< 90%).** Then the play-out assumption under every cost model here and in the dossier is wrong, and the first thing to build is a truncation rule, not a teacher. Cost to find out: $0.20.
2. **Rung 1: no margin beats `raw:t0` at p ≥ 0.535 over 800 confirmation games.** Then the exact enumerator's information is real (§2.7) and worthless at the deployed operating point — the seed is not making arithmetically detectable combat errors that matter. XOD is retired, and the prior drops for any operator whose candidate set is the prior's top-k. Cost: $0.75.
3. **Rung 1: the winning arm's override rate is below 0.5% of decisions.** Then the teacher is `raw:t0` in disguise and any positive gate reading is selection noise — the same instrument that read KL 0.002 for Gumbel and TCS, pointed at this design.
4. **Rung 2: the teacher wins, the student loses, and held-out teacher-agreement is high.** Then the improvement is procedure-only and does not live in weights. The user's objective ("make the policy play better") is unreachable by distilling this teacher; the honest options are to ship the procedure under its own tag or to stop. This is the outcome I would find most informative and least welcome.
5. **Rung 2: the teacher wins, the student loses, and agreement is low, and a 4× data increase does not move agreement.** Then the encoder cannot represent exact combat arithmetic from its features, and the constraint is the representation, not the algorithm.
6. **The information-budget argument itself is falsified if** rung 4 returns ≥ 5% of side-turns carrying |δ| ≥ 0.25. Then a turn-level rollout teacher at R ≈ 30 is affordable (a 40-game gate for $2–4), the "deviations are unaffordable" claim holds for *atoms* but not for *turns*, and RCTC — repaired at honest depth, with the divergence-repair from `tools/turn_policy.py:118-131` restored and the commit rule set to unanimity — becomes the right next design. That is the one branch in which a dossier design comes back, and it comes back only behind a measurement nobody has taken.
7. **The whole ordering is wrong if** a 400-game match turns out to carry materially more variance than binomial — i.e. if outcomes cluster by map × matchup beyond what side alternation balances. Check it in rung 0 by reporting a cluster-robust SE over map × matchup cells alongside the binomial one; if the ratio exceeds ~1.3, every game count in §4 must rise by that factor squared and the budget table is wrong.
