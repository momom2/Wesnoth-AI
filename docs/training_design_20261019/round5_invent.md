# Round 5 (invent): supervising what the dice depend on

## Summary

Every gain in the current design passes through κ: whether a critic trained on chance siblings ranks decision siblings (`round5_provenance.md` §5). This memo keeps the provenance stance's frame and regimen and adds three components, each measured before anything relies on it.

1. **Law supervision.** The critic, the student and later the distilled player each carry an auxiliary head. It predicts, for every legal attack at every training state, the exact outcome law the core enumerates. The labels cost CPU only, carry no noise and cover the attacks not played. Nothing rule-computed enters play.
2. **Exact-truth plants and a sibling-transfer probe.** These are planted checks whose answer the core computes without playouts. One asks κ's question for an exactly known quantity, for cents, before the critic is paid for.
3. **The turn-over rule.** End the side-turn iff P(end_turn) ≥ ½. It is derived from why the joint argmax under-acts, has no tuned number, and is tested in S0 for $0.48.

The derivation in 2.5 removes one family of designs: those that couple two runs through shared random draws.

## 1. What the records settle

1. **An argmax decision flips at a rate set by the density of its margin.** In 100 recorded `parity3` side-games, the top-two gap has a density near 0 of 1.19 per nat per decision, 320 per side-game (`analysis/round5_invent_margin_flips.py`).
   - A symmetric logit change d at every decision flips a decision with probability f·d/2. That gives 1.6 flips per side-game at d = 0.01 nats and 4.8 at 0.03.
   - The replay disagrees with the record at 1.8% of decisions (`prior_gaps.json`: 26,684 of 27,171 agree), which corresponds to d ≈ 0.03. Batch arithmetic alone therefore makes two runs of one game diverge.
   - Keyed Gumbel tie-breaking at T = 0.05, 0.1, 0.25 and 0.5 nats leaves 0.94, 0.86, 0.69 and 0.52 of argmax's flip density. At those values it deviates from argmax at 10.6, 19.7, 40.7 and 63.1 decisions per side-game (same script).
2. **Coupling on sampled play is weak too.** Shared seeds at temperature 0.5 keep 0.71–0.90 of the pair variance between states that differ only in the mover's village and gold estimates (`analysis/round5_invent_coupling_record.py`, re-read).
3. **Attack choices are frequent.** An attack is the top action at 52.9 decisions per side-game. Two or more attacks sit in the top eight at 25.2% of multi-legal decisions, 67.8 per side-game (`analysis/round5_invent_turnover.py`).
4. **The observation holds a fight's ingredients, not its law.** The tokens carry weapons, resistances, traits, statuses, time of day and the hex's terrain (`wesnoth_ai/encoder.py`, parity unit row), but no hit chance or kill probability. Those require crossing the movement type with the terrain and running an alternating strike sequence with a kill threshold.
5. **The turn-over rule mostly agrees with the −1.5 offset where the offset ends the turn.** End_turn is the top action under the offset at 40.5 decisions per side-game. The turn-over rule would continue at no less than 1.0% of them. The record cannot read the other direction (`analysis/round5_invent_turnover.py`).

From the provenance stance, with its scripts re-read: a corrected match keeps 0.747 of the result's variance; Match C reads z = −0.035 ± 0.035.

## 2. The algorithm

### 2.1 Frame (kept from `round5_provenance.md` §2)

- argmax self-play with deviation and stop coins, propensities logged;
- a privileged critic C trained by the multistep identity loss, with a cross-fold corrector;
- κ per action type, with a blind SE;
- a Longstaff–Schwartz stop head on forced paths;
- a fogged student that regresses Â_C over the top four candidates, is calibrated, and drives a switch rule;
- adoption only by a certified corrected match.

### 2.2 Law supervision

**Target.** At a state x, for each legal attack b of the side to move, the core enumerates the law p_b(·|x) over outcome keys, with advancement merged as in the luck walk. From the law it reads

    φ_b(x) = (P(defender dies), P(attacker dies), E[HP_def]/max, E[HP_att]/max,
              P(defender slowed or poisoned), P(attacker slowed or poisoned))

The attack dialog derives its numbers from the same law; which ones it displays is unchecked.

**Head.** A linear readout maps four tokens of a network's last trainable layer (attacker, defender, the attacker's hex, global) to φ̂_b. The loss is ℓ_law = Σ_b ‖φ̂_b − φ_b(x)‖², over all legal attacks at every training state.
- **On the critic (input x).** φ_b is a deterministic function of the input, so the held-out error is pure approximation error.
- **On the student (frozen `own1` tokens plus two trainable layers).** The minimiser is E[φ_b | h]. The fog's part, φ_b − E[φ_b | h], is zero wherever every unit whose ability enters the fight is visible. S0(g) counts the exceptions.

The auxiliary gradient enters shared layers only while its cosine with the main gradient is positive ([Du et al. 2018](https://arxiv.org/abs/1812.02224)), the design's rule for consequence heads, which keeps the main objective's stationary points. Each loss is divided by its value at initialisation; the readouts in S1b and S2 test the result.

**Why it should help.** For an attack, Â_C(x, b) = Σ_o p_b(o|x) C(x∘b∘o) − C(x∘a0).
- **The student.** The factor p_b is rule arithmetic. Without its own labels, a head must learn it from value targets that carry the result's noise (0.747 of Var(z) per game after correction). Law supervision labels that factor exactly at every sibling, so the expensive labels only need to fix the values of outcomes.
- **The critic.** Where fights are imminent, V's next changes are law-weighted children values. Short-run material is much of what the record's evaluators rank: the HP margin after the turn reads 0.40 (seed §3), and the best step-1 critic 0.518 ± 0.056 against the margin's 0.380 (`critic_step1_20261009/readout.md`).

For: TD-Gammon went from strong intermediate to "truly staggering" once hand-crafted inputs were added, among them the probability of a blot being hit ([Tesauro 1994](https://mlanthology.org/neco/1994/tesauro1994neco-tdgammon), abstract; feature list from [Wikipedia](https://en.wikipedia.org/wiki/TD-Gammon)). Removing KataGo's ownership and score targets cost 190 Elo ([Wu 2019](https://arxiv.org/abs/1902.10565), Table 2).

Against: auxiliary tasks can transfer negatively (hence the gating); targets give less than inputs; and law supervision adds no decision information, since Q(x, b) − Q(x, a0) across fights is still pinned only by coins and the critic's generalisation (round 4's critique of the factored arm). The gain is approximation; S2 measures it on identical targets.

**Cost.** An enumeration takes 0.06–0.11 ms of CPU (seed §2) and a game has at least 452, so at least 27–50 ms of CPU per game. No forward pass is added.

**Invariants (tests from the spec).**
- I-L1: the training pipeline's φ_b equals the core's enumeration on the luck walk's fights.
- I-L2: swapping the two roles, or shifting the weapon index, changes φ exactly as the law does.
- I-L3: the student's inputs hold no hidden unit (the design's fog-leak test).
- I-L4: the exact-truth plant (2.3) passes, and a key-only readout fails it.

### 2.3 Exact-truth plants and the sibling-transfer probe

**Exact-truth plant.** A booked stream, leaving the game unchanged, adds 1 when a ranged attack kills a defender on a village. Its value for candidate b is exactly 1(village)·1(ranged)·P_b(defender dies), so its band comes from the head's held-out residuals, with no playouts. It passes through the same builder as Â_C (children summed with law weights), testing the builder's weights, roles and signs and the head's reading of h. Readouts: slope of truth on prediction 1 within 2 computed SE; R² above, by 2 SE, the ceiling of a key-only predictor (outcome-key features and range bit, no hex), computed by grouping the data as Var(E[truth | key inputs]) / Var(truth). If the data shows the event too rare, the plant books every ranged kill.

**Sibling-transfer probe (S0).** On `own1`'s replayed states, a head on frozen `own1` tokens is fitted to φ at **played** attacks only. It has two arms:
- **Exact arm:** trained on the exact law.
- **Realised arm:** trained on the realised outcome, which is unbiased given x.

Both are scored against exact φ on **unplayed** legal attacks and on held-out played ones; an arm transfers if its sibling R² is within 2 SE (bootstrap over positions) of its in-distribution R². The exact arm isolates distribution shift; the realised arm adds the noise of learning from outcomes, as the critic does.
- **Both transfer:** attack siblings are not where κ fails.
- **Only the exact arm transfers:** label noise dominates, so the critic's variance reduction is what matters.
- **Neither transfers:** shift dominates. Law supervision, which labels every sibling, is the targeted remedy, and its S1b arm is read first.

### 2.4 The turn-over rule

**Model.** Let S be the set of actions the human will still play this side-turn, with k = |S|. The head learns:
- P(end | x) = P(k = 0 | x);
- for an acting action a, P(a | x) = E[1(a ∈ S)·w(a | S)], where w is the human's choice of which pending action comes first.

Where pending actions can be played in any order, such as moves of units that do not interact, w ≈ 1/k. Then max_a P(a | x) ≈ (1 − P(end))/k_eff, and the joint argmax ends whenever P(end) ≥ (1 − P(end))/k_eff. It ends too early by log k_eff, and more so the more actions are pending. A constant offset corrects the mean: c = −1.5 corresponds to k_eff = e^1.5 = 4.5.

The event "the turn is over" has its own probability, and its MAP rule is P(end) ≥ ½. That rule has no number to tune and follows k_eff state by state. It fails where one acting action dominates while P(end) > ½: it ends although a valuable act remains.

**Record.** The actor rule corrects splitting within one unit and gained +193 ± 14; the offset gained +229 ± 15 (`docs/endturn_rule_prereg_20260919.md`). The turn-over rule also corrects splitting across units, and was never tested. §1 item 5 bounds only one direction of disagreement, so no sign is predicted; S0(j) measures it.

### 2.5 Coupled runs, rejected

Keyed dice across forks (deferred since round 4), keyed tie-breaking, paired forks and paired matches need two runs to share their draws for long enough. By §1 item 1, a 0.03-nat change at every decision, the size batch arithmetic already produces, flips 4.8 decisions per side-game; after the first flip the runs fight different fights. Keyed noise cuts flips by a third only at T = 0.25, which deviates at 40.7 decisions per side-game, as many deviations as the band coins spend (25 to 43, `current_design.md` §2.1). And at ρ = 0 a coin game carries about 40 times a fork pair's information at equal price (`round5_provenance.md` §1); coupling multiplies a fork's by 1/(1 − ρ), so forks win only above ρ = 0.975.

## 3. Regimen

Prices are the provenance stance's; any stage not listed is the provenance stance's, unchanged.

| Stage | Cost | Readout | Decision per outcome | Plant |
|---|---|---|---|---|
| **S0** | $3.0 | Provenance (a)–(f). (g) Law census: legal attacks per decision, law(h) ≠ law(x) share, CPU. (h) Linear probe of φ on frozen `own1` tokens. (i) Sibling-transfer probe. (j) Turn-over rule against `own1` at −1.5, 3,200 corrected games, SE 5.3 Elo, no sign predicted. | (j) within 2 SE: π₀'s stop on non-inferiority (provenance ruling c), before any coin game. Below by 2 SE: −1.5 stays. Above by 2 SE: fresh-seed confirmation first (kv15: +35.7, then +1.7). (i): 2.3's readings. (h): S2's baseline. | P-coin |
| **S1a** | $2 | P1, as in the provenance stance. | Outside the band: the critic path is broken. | P1 |
| **S1b** | $7 | Provenance pilot plus a law-supervised arm on one fold of the same games, paired against that fold's critic: ρ̂, κ per type. No prediction. | Law arm better by 2 paired SE in ρ̂ or κ_attack: S1c uses it; else not. | — |
| **S1c** | ≤ $20 | Unchanged. | Unchanged. | P1 if the code changed |
| **S2** | $6.4 | Provenance S2, plus the exact-truth plant through the Â_C builder and a student A/B with and without law supervision on identical Â_C targets: attack R², argmax regret, paired by position. | Plant fails: law path unused. Law student better by 2 paired SE: adopted. Predicted null if S0(h)'s R² on φ matches the trained head's within 2 SE. | Exact-truth, P2, P3 |
| **S3** | $6 | Unchanged. | Adoption rule. | — |
| **S4** | $10 each | Unchanged. The law readout is refitted at no generation cost. | Unchanged. | P1, P2, P3, exact-truth |
| **S5** | $5 | P5. The distilled trunk keeps the law target if S2 adopted it. | P5 fails: distillation broken. | P5 |

**Budget.** S0–S3 about $45 at S1c's cap ($3 above the provenance stance); with rounds 2–3 and S5 about $70, leaving about $15 for one remedy branch or the batch-invariant kernels ($7). **Beyond `own1`:** the $1,000 case is argued only from two certified rounds within 2 SE of their forecasts; law labels scale with critic data at no generation cost; nothing measured links `own1` to human ratings.

## 4. What is new, and the nearest work

I searched on 2026-10-20. Finding nothing does not prove absence.

1. **Law supervision:** exact, noise-free targets from the rules engine for every legal stochastic action, so decision siblings are labelled, shaping a fogged student and a privileged critic with nothing rule-computed in play. Nearest, none supervising with exact chance laws: learning from hints (Abu-Mostafa, J. Complexity 1990; not re-read); [UNREAL](https://arxiv.org/abs/1611.05397); [DFP](https://arxiv.org/abs/1611.01779) (per-action future measurements, from experience); [Stochastic MuZero](https://openreview.net/forum?id=X6D9bAHhBQ1) (chance distributions, unsupervised); KataGo's auxiliary targets; TD-Gammon's hit-probability inputs; [SkyNet](https://arxiv.org/abs/2603.27751)'s winner and rank heads.
2. **Exact-truth plants and the sibling-transfer probe:** generalisation from played to unplayed actions tested against simulator-computed truth, no playouts. Nearest: [bsuite](https://arxiv.org/abs/1908.03568); linear probes ([Alain & Bengio](https://arxiv.org/abs/1610.01644)).
3. **The turn-over rule:** the end_turn bias of a factored imitation head as mass split across the orders of pending actions, and its MAP rule. Nearest: [Eikema & Aziz 2020](https://arxiv.org/abs/2005.10283) (MAP decoding's inadequacy); [Grinsztajn et al.](https://arxiv.org/abs/2110.10632) (action-sequence equivalence).

**Not new:** 2.5 rests on common random numbers ([PEGASUS](https://arxiv.org/abs/1301.3878); [Gumbel-max SCMs](https://proceedings.mlr.press/v97/oberst19a.html)). Only the measured flip density is this project's.

## 5. Weakest point

Law supervision adds no decision information. If κ fails for values, representing laws better will not open it, because across fights Q(b) − Q(a0) is pinned only by coins and by how the critic generalises. Its gain is approximation, bounded by what the frozen tokens already carry (S0(h)).

The transfer probe asks κ's question only of a quantity that is a deterministic function of h. Values are harder, so a pass is evidence, not proof.

The turn-over rule can lose where one valuable act remains and P(end) > ½.

Beyond that, this proposal inherits the provenance stance's weakest point.

## 6. Provenance

| Number | Value | Used for | Tag | Source or derivation |
|---|---|---|---|---|
| Margin density near 0 | 1.19 per nat per decision; 320 per side-game | 2.5 | measured | `analysis/round5_invent_margin_flips.out` (G < 0.05; 1.24 at 0.02, 1.15 at 0.1) |
| Multi-legal decisions | 269.2 per side-game (`parity3`) | scaling | measured | same |
| Replay disagreement | 1.79% (487 of 27,171) | numerics as a perturbation | measured | `critic_step1_20261009/prior_gaps.json` |
| Equivalent perturbation | d ≈ 0.030 nats | 2.5 | derived | 2 × 0.0179 / 1.19, symmetric two-candidate model |
| Flips per side-game | 0.5, 1.6, 4.8, 16 at d = 0.003, 0.01, 0.03, 0.1 | 2.5 | derived | f/2 × d × 269 |
| Keyed-noise flip density | 0.94, 0.86, 0.69, 0.52 at T = 0.05, 0.1, 0.25, 0.5 | 2.5 | derived from measured G | E[f_L(G/T)/T], same script |
| Keyed-noise deviations | 10.6, 19.7, 40.7, 63.1 per side-game | 2.5 | derived from measured G | Σ 1/(1 + e^{G/T}), same script |
| Band coins' deviations | 25 to 43 per side-game | 2.5 | measured | `current_design.md` §2.1, reproduced by the round-4 critique |
| Coins against forks | about 40× per dollar at ρ = 0 | 2.5 | derived | `round5_provenance.md` §1 (180 × 0.19 × A² against A²/1.22) |
| Fork break-even coupling | ρ > 0.975 | 2.5 | derived | 1 − 1/40 |
| Coupling, sampled play | 0.71–0.90 kept; 138 of 7,728 identical | context | measured | `analysis/round5_invent_coupling_record.out` (temperature 0.5) |
| Attack as top action | 52.9 per side-game | label volume | measured | `analysis/round5_invent_turnover.out` |
| Two or more attacks in top 8 | 25.2% of decisions; 67.8 per side-game | same | measured | same |
| Attacks in top 8 | 0.84 per multi-legal decision | label count, lower side | measured | same |
| Law labels per game | ≥ 452 | CPU | derived | 0.84 × 269.2 × 2 sides |
| Enumeration time | 0.06–0.11 ms per attack | CPU | measured | seed §2 |
| Law CPU per game | ≥ 27–50 ms | cost | derived | 452 × 0.06–0.11 ms |
| Legal attacks per decision | — | label count | unknown | S0(g); only the CPU share depends on it |
| law(h) ≠ law(x) share | — | the student's fog part | unknown | S0(g); reported |
| Frozen probe R² on φ | — | S2's baseline | unknown | S0(h), forwards of S0(a)'s replay |
| Sibling transfer | — | which remedy S1b reads first | unknown | S0(i); each outcome mapped (2.3) |
| Law arm's effect on ρ̂, κ | — | S1c | unknown | S1b, paired, SE computed in the run |
| Law arm's effect on attack R² | — | the student | unknown | S2 A/B, paired by position |
| End-top decisions | 40.5 per side-game | 2.4 | measured | `analysis/round5_invent_turnover.out` |
| Turn-over continues where the offset ends | ≥ 1.0% (0.41 per side-game) | 2.4 | measured, lower bound | same; the tail beyond the top eight is unread |
| k_eff at c = −1.5 | 4.5 | 2.4 | derived | e^1.5 under the stated plan-set model |
| Actor rule; offset | +193 ± 14; +229 ± 15 | 2.4 | measured | `docs/endturn_rule_prereg_20260919.md` |
| Turn-over rule's strength | — | S0(j) | unknown | S0(j); no prediction |
| Corrected variance kept | 0.747 (0.733–0.758) | match sizes | measured | `analysis/round5_provenance_match_variance.out`, re-read |
| Corrected-match SE | 5.3 Elo at 3,200 games | S0(j) | derived | 347.4 × √(0.747/3200) |
| Elo per unit of z | 347.4 | all SEs | derived | 800/ln 10 at p = ½ |
| Decisive match game | $1.49e-4 | S0(j) | derived | $0.42/h × 17 min / 800 (seed) |
| S0(j) cost | $0.48 | budget | derived | 3,200 × $1.49e-4 |
| S1b law arm | $2 | budget | derived, upper side | one fold of 5,000 games, read paired against that fold's critic without the law head: 5,000 × about 1,000 critic evaluations per game (110 events × 9 children + 24 states) × $7.8e-8 × 5 passes, every evaluation priced as a training position; the pass count is the pilot's choice |
| S2 student A/B | ≤ $0.4 | budget | derived, upper side | two fits, each priced as one trunk pass over 2.4 M student states (50,000 games × 24 states, the record's per-game cap in `critic_step1`, × 2 folds) at $7.8e-8; a head on cached tokens costs less than a trunk pass per epoch |
| Training position | $7.8e-8 | costs | derived | $0.42/h / (1,500 × 3,600 s) |
| Provenance stage costs | $2.5, $2, $5, ≤ $20, $6, $6, $10, $5 | budget | derived | `round5_provenance.md` §3 |
| Match C | z = −0.035 ± 0.035 | stop known answer | measured | `analysis/round5_provenance_stop_acts.out`, re-read |
| Turn ranking | 0.22, 0.27, 0.40, 0.53 | 2.2 argument | measured | seed §3 |
| Step-1 critic | 0.518 ± 0.056 against 0.380 ± 0.097 | 2.2 argument | measured | `critic_step1_20261009/readout.md`, adjusted truth, read 0 |
| Events per game | 110 fights, 46 trait rolls | volumes | measured | `match_luck_20261019/readout.txt` |
| KataGo auxiliaries | −190 Elo when removed | evidence | measured elsewhere | [Wu 2019](https://arxiv.org/abs/1902.10565); carries no decision |
| TD-Gammon features | qualitative | evidence | measured elsewhere | [Tesauro 1994](https://mlanthology.org/neco/1994/tesauro1994neco-tdgammon); carries no decision |
| Winner's curse | +35.7 ± 12.3, then +1.7 ± 12.3 | S0(j) confirmation | measured | `docs/decode_knobs_prereg_20261012.md` |
| Loss scale, 2-SE gates, coin-free quarter | — | conventions | choices | the readouts test them; nothing derived rests on them |
| Budget | $85 at $0.42/h | everything | given | seed §1 |

## 7. Decision record

**Rejected:**
- **Keyed coupling of runs:** the margin density makes argmax runs diverge within about a side-turn under any perturbation as large as batch arithmetic (2.5).
- **Laws as inputs to the player:** not without ruling (b).
- **A fixed auxiliary weight:** gradient-cosine gating and the A/B replace it.
- **The factored arm's composition in play:** it needs the outcome support at decode, which the rules compute, so it falls under the same ruling.

**Flagged for the user:** laws as inputs to the training-only critic. The critic never plays and the law is the rules' arithmetic, but the choice of features is hand-made in the TD-Gammon sense. If allowed, it becomes a third S1b arm at $2.
