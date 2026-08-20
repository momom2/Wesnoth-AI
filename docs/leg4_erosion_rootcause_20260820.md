<!-- Provenance: 17-agent Opus 5 workflow wf_aa4c578c-c63, 2026-08-20,
launched on user directive after the leg-4 pin measured 1-24 vs 2516k and
0-13 (partial) vs its own seed. 82 dossier findings, 12 hypotheses,
8 adversarially verified, 2 survived (weakened). -->

# Leg-4 root-cause synthesis (2026-08-20)

**Explanandum.** `tier_b_l4` trained ~495k decision-steps of TCS self-play and came out **0–16 against its own seed** and **1–24 against the 2516k anchor** (joint Elo −277 ± 114), while ~28 iterations of internal telemetry stayed green. 17 candidate mechanisms were put through independent adversarial review. **Two survived, both weakened, and the adversary's own simulation of the shipped code showed they are not rivals — they are two necessary arms of one conjunctive failure.** Six were refuted outright.

---

## 1. Surviving root-cause candidates, ranked

### The headline, in one paragraph

The training loop contains a term that pushes the policy a little bit toward "all legal moves equally likely" on every single distillation step, and a term that is supposed to push back by saying "these particular moves are better." The pushing-toward-uniform term was **on by accident** (config drift). The pushing-back term was **carrying almost no information** (the value head cannot tell two candidate turns apart at the resolution the target channel uses). So for 495k steps the policy relaxed toward random-legal-move. This does not show up in training play, because at training time the *planner* — not the policy — picks the moves, and the planner brute-forces hundreds of full turns per side-turn, so it plays fine off a mush prior. At eval there is no planner: the checkpoint plays 32-sim Gumbel MCTS, which samples 16 candidates *from the prior* out of a ~200-action legal set. Sixteen near-random candidates is near-random play, and near-random play in Wesnoth loses the opening. Median leader death: turn 8.

---

### **R1 — Prior-flattening ratchet (the decay arm)** — `strong`, with a `measured` core and one `suggestive` gap

**Plainly:** every time TCS writes a training target, it first raises the model's own move probabilities to the power 0.9 and renormalizes. That operation, iterated, has exactly one fixed point: uniform. In the *old* (exp) link this was deliberately balanced by an additive value term applied to **all** actions. The linear link — new as leg-4's default — replaced that with a multiplicative bonus applied only to the ≤5 actions actually evaluated at that coordinate, out of ~200 legal. So ~97% of the taught mass at every state is pure flattening with nothing pushing back.

- Code: `tools/turn_search.py:512-537` — `base = p ** lam; base /= base.sum()`, then `factor[i] = 1 + beta*(q_i - LOO_mean)` set only on `np.flatnonzero(ev)`. λ read at `:487-488` from `mcts_config.distill_prior_discount`; `build_coordinate_target` (`:571-574`) emits a tuple for *every* action with w > 1e-9; `trainer._mcts_factored_policy_loss` (`:1015`) is a plain weighted NLL — no entropy floor, no re-sharpening. `[measured]`
- **The target was flatter than the model's own prior at 26 of 26 real iterations**, and `distill_sharpen_top` (the search's contribution to sharpening the top action) was **negative at 26 of 26**, decaying −0.032 → −0.004. This is a same-state, same-iteration comparison, so it carries **no state-drift confound**. `[measured]`
- What the model's prior then did: `distill_prior_entropy` 3.215 → 5.147; `distill_prior_top80` (share of decisions where the top action holds >0.8 mass) 0.090 → 0.011 — an **8× loss of confident decisions**; `distill_kl_prior` 0.041 → 0.010. `[measured]`
- Behavioural signature is what uniform-over-legal predicts and nothing else does: the legal set is dominated by move-targets, so flattening reallocates mass to moves. `action_attack_pct` 19.9 → 4.5 (ρ = −0.986, monotone from iteration 0), `action_move_pct` 64.9 → 85.4, **`actions_per_turn_median` unchanged at 11–16**. Composition changed; magnitude did not. `[measured]`
- **This was config drift.** `--distill-prior-discount 0.9` is not in BACKLOG's LEG-4 LAUNCH CONFIG. `scripts/vast_onstart.sh:786` emits it **unconditionally**, un-gated, while the anchors at `:776-781` are env-gated. Both `tools/sim_self_play.py:3166-3175` and `tools/policy_anchor.py:15-18` state the F1 ruling that **exactly one** prior protection runs per leg. Leg 4 ran two. `[measured]`
- **This is also the only variable that distinguishes leg 4 from leg 3.** `docs/leg3_passivity_rootcause_20260817.md:105` records leg 3 ran `lam = 1.0`. Leg 3 collapsed differently (K truncation, draw flood) and its human-probe actor@1 stayed **flat** at 0.47–0.49; leg 4's eroded monotonically 0.558 → 0.489. Same GBC, same aux-score, same anchor family, different λ, different failure. `[measured]`

**What is weak about it.** The "λ̂ = 0.9185 lands within 0.019 of the shipped 0.9" fingerprint from the first pass **does not survive**: the entropy deficit contracts as ~λ², the ceiling `log n_legal` is a free parameter that slides from n≈119 to n≈1240 across the plausible λ range, and the confound-free per-iteration regression implies an effective contraction nearer λ ≈ 0.94. The force is real and directional; **the effect size on the weights is not established**. `[suggestive]`

And the headline series `distill_prior_entropy` is a mean over the leg's *own drifting states*. Passive play removes ZOC and contact, which plausibly grows reachable-hex sets 2–3× (+0.7–1.1 nats of ceiling) with **frozen weights**. So part of the entropy rise may be a *consequence* of passivity rather than its cause. **No frozen-state measurement of this policy has ever been taken.** `[suggestive — and this is the single cheapest thing to fix, see E1]`

---

### **R2 — Grader-null (the "nothing pushes back" arm)** — `strong`

**Plainly:** the linear link was designed so that a *random* judge cannot inflate the actions it happens to look at. The flip side is exact: when the judge carries no information, the target degenerates to `prior^λ` plus mean-zero noise. R2 says the judge carried no information **in the target channel**, from iteration 0.

- The target pass (`turn_search.py:754-816`) grades each alternative with **one materialization at one salt and one boundary forward**, with no revalidation and no paired averaging. The accept gate (`two_stage_accept`, `:440-449`) averages `reval_salts = 3` paired materializations behind a 2σ threshold. **These are different estimands with different signal-to-noise.** The A4 bake-off's 6.9:1 real-vs-placebo separation certifies the *gate*, not the *target*. This distinction was found during adversarial review and is the thing that makes R2 coherent. `[measured — code]`
- The Q8 CRN probe already established that a single coordinate edit changes *which fights happen* (median 0 matched downstream fight identities under both strict and loose keys; only 28% of pairs share even one). So single-salt `q_incumbent` vs `q_alt` is dominated by combat luck. `[measured, prior]`
- **The decisive evidence is a sign test on the shipped function.** Running `tcs_target_distribution` + `gumbel_top_k_alternatives` directly (N=120 legal, 5 evaluated/round, β=5, 600 rounds):

| arm | end entropy | top mass |
|---|---|---|
| λ=0.9, null grader | 4.753 (uniform = 4.787) | 0.027 |
| λ=0.9, decay only | 4.787 (exactly uniform) | 0.008 |
| λ=0.9, informative (SNR 1) | 0.794 | 0.761 — **concentrates** |
| λ=0.9, informative (SNR 4) | 0.100 | 0.983 — **concentrates** |
| λ=1.0, null grader | 3.033 from 3.200 | 0.173 — **no collapse** |
| λ=1.0, decay only | 3.200 (stationary) | — |

  **λ<1 is necessary AND grader-nullity is necessary. Neither alone reproduces leg 4.** With a working grader, λ=0.9 drives the policy the *opposite* way, to a near-delta. Leg 4's measured trajectory sits squarely on the null-grader branch. `[measured — simulation of shipped code]`
- Why nobody saw it: **every statistic that measures the grader on TCS's own estimand was structurally dark for the entire leg.** `boundary_pairs_n = 0` and `boundary_sum = nan` in all 28 rows, because `harvest_boundary_pairs` is called only from `finalize_game` and the spool ingest — the actor-pool branch (`tools/sim_self_play.py:1422-1433`) just does `policy._queue.extend(pool_exps)`, and leg 4 ran `--actor-pool 19`. `search_q_spread` and `search_overturn_frac`: empty in all 28 rows. `tcs_plans` / `tcs_accepts_per_plan` / `tcs_replans_per_plan`: computed in `drain_distill_stats` but never added to the CSV field list (`:2295-2325`) — INFO log only. `gbc_loss` and `aux_loss`: same. `[measured]`

**What is weak about it.** R2's own causal story — "the head drifted in-leg from supervision starvation" — **does not survive its own telemetry.** `distill_sharpen_top` is already at the full null-grader value (−0.032 measured vs −0.028 simulated) at **iteration 0**, before any starvation accumulated. So the target channel was near-uninformative essentially from the seed, and the starvation columns (`value_signal_states` 2048 → 1038, A2 value anchor off, `--value-coef` doubled to 1.0 on the shrinking remainder) are at most an aggravator. **Implication that matters for planning: repairing value supervision would not have saved leg 4.** `[measured]`

Also: R2's slogan "link=linear IS a decay operator by construction" is **false**. The decay is entirely λ. At λ=1.0 the same null grader is exactly stationary. R2 must be stated conjunctively with R1 or it misnames the agent.

---

### **R3 — Contributing config findings (not causes, but they shaped what was visible)** — `measured`

Not root causes; each independently verified and each worth a line in the log.

1. **The A2 human *value* anchor was OFF all leg and no ruling exists for that.** `human_anchor_loss` empty in all 28 rows; the launcher default is ON (`vast_onstart.sh:417`); BACKLOG left it open ("redundant with the A3 seed? — decide at launch"). It can also fail silently (`:453-457` warns and unsets). The value corpus **was** on the box — the midgame arm read the same directory and ran normally (4–6 games/iter).
2. **Value supervision fell by half.** `value_signal_states` 2048 → 1038 as winnerless censoring (`mcts_policy.py:617`) met a rising cap rate, with `--draw-value-weight` left at its 1.0 default and no human value rehearsal to compensate. `train_n_transitions` stayed pinned at 2048, so the loss of signal was **invisible in the batch-size column**.
3. **Two shipped rulings disagree about capped games.** Search still scores cap terminals by material margin (`--draw-tiebreak-cap` 0.3) while the trainer censors those same games out of the value loss entirely. Neither shipping ruling reviewed the other.
4. **Scenario mix was 60/20/20 by inherited launcher defaults**, against the technique review's Phase-1 prescription of 100% ladder. 40% of training games are off the eval distribution.
5. **`configs/leg_l4.json` pins 4 of ~13 leg-4 rulings.** Everything signal-critical lives in argparse defaults or shell env defaults, none of which the config file asserts. The leg-3 "ruling that lived only in a launch env" failure repeated one level up.
6. **Verified correct, no drift:** GBC genuinely on at coef 0.1 end-to-end; turn-cap jitter 60–100 as a code default; linear link β=5.0; projection OFF; stalemate rule off.

---

## 2. Definitively ruled out this round

| Candidate | Killed by |
|---|---|
| **Boundary antisymmetry offset** (value head's side-switch bias distorts eval MCTS but cancels in TCS) | The required asymmetry is **inverted in the only lineage data**. `docs/autonomous_run.md:2344` measures the SL/imitation-fit family as the *worst* offender (+0.43/+0.65, called "structural, inherited"), and the 2516k-era lineage at ~0 (rolling \|mean\| 0.079). H1 needs the seed clean and the pin biased. Self-refuting at its own magnitude: the seed — the closest instance of an imitation-fit head — goes **16–0**. Also, its explanation for green K is superfluous: **K × `distill_et_prior` = 0.846 ± 0.085** across all 26 real iterations, and `et_prior` moved only 0.075 → 0.068. K was structurally pinned; no cancellation story was needed. |
| **C51 quantization / value-evidence collapse** ("siblings in the same atom have exactly zero ΔV") | **The premise has no source line.** `model.py:419-422`: V = `(softmax(value_logits) * atoms).sum()` — an expectation, continuous. The atom grid bounds the *support*, not the resolution. Its two headline series are **arithmetically forced by R1 with the value factor set to exactly 1** (simulated: KL 0.040→0.011, sharpen_top −0.03→−0.004 from prior entropy alone). And `link_clip_frac` — which fires only when a boundary value is ≥5 atoms below its LOO mean — is **flat at ~0.055 from iteration 0 to 25**, directly contradicting a value channel that stopped supplying evidence. |
| **Value head became a material proxy → search declines trades** | **The material scorer is HP-blind.** `draw_tiebreak.py:110-118`: unit value = sum of *recruit costs* of living units; `weight_gold = 0.0`. Damage moves it by exactly zero; a favourable attack that kills is strictly *positive*. The stated mechanism ("attacker takes damage, so ΔV is negative at commitment") describes a scorer that does not exist here. And leg 4 moved **against** all three things that scorer prices: recruits 7.6→2.1, villages 0.431→0.367, kills 29→13, while hoarding 180–250 gold the scorer values at zero. `--mcts-aux-score` is also emitted unconditionally by the launcher — constant across legs 1–4. |
| **Scale mismatch** (turn-scale grader consumed as a ply-scale one) | Under TCS, `turn_policy` appends one experience **per coordinate**, so at K≈13 roughly **92% of the value head's training mass is mid-turn states** — exactly the states the hypothesis declares untrained. Boundary states are the ~1/K starved class. It cites that starvation figure as support; it points the other way. No loss term references ΔV at either scale — both statistics are read off one E[z\|s], so there is no gradient that can rotate one without the other. Fails the project's own admissibility bar (name a symmetry-breaking mechanism). |
| **Anchor-vs-distill objective conflict** (F1 anchor's unmasked BC overfit a 500-game pool) | The dominance premise is false at the optimizer: **AdamW, both paths clipped to grad_norm 1.0**, 4 anchor steps vs **16** distill steps per iteration, and `train_grad_norm` reads 1.0–5.8 pre-clip (distill usually clipped to full size). Its named force points the **wrong way**: anchor class weights are recruit 1.748 / attack 0.628 / end_turn 1.435 vs move 0.189 — a restoring force for exactly the two metrics that collapsed. Hard-target BC is entropy-*reducing*; the policy went uniform. The "train down / holdout up" gap shrinks to −0.12 nats of post-iteration-0 drift vs +0.47, over **under one pass** through distinct pairs. Its head-localization prediction fails on `type_top1` (0.9276 → 0.9235, flat) — the one head that actually carries the class weights. |
| **GBC / aux-score trunk contamination of the actor head** | **Architecturally impossible as stated.** All four policy heads read `unit_ctx` (`model.py:271, 588, 593, 598`); GBC's gradient additionally enters `hex_ctx` (the target head's key side) and `global_ctx` — it touches *more* of the target head's inputs than the actor head's. The novelty premise is factually wrong: `BACKLOG.md:90` puts GBC in the **leg-2** config, `--gbc default=True` since 2026-08-14. Leg 3 ran identical GBC with actor@1 **flat**; leg 1 ran **no** GBC and eroded the human prior *faster*. And the actor-vs-target asymmetry has a mundane cause: `actor_top1` is an unconditional ranking over ~15–25 slots while type/target/weapon top-1 are **teacher-forced on the ground-truth actor** (`supervised_train.py:911-923`). The measured head table orders by conditioning, not token provenance: type −0.4%, target −2.4%, weapon −3.0%, actor −12.4%. |

**Also re-confirmed dead / do-not-re-propose:** end_turn force-inclusion ratchet (exchangeable null reproduces the whole surplus curve, zero free parameters; and `distill_et_target < distill_et_prior` at **26/26** leg-4 iterations — the link *demotes* end_turn); one-ply horizon truncation; fog-invisible Q ties; unpriced tempo; value-resolution collapse silencing the accept gate; luck ledger / AIVAT; CRN; all fallback graders and hand-tuned loop quantities (user ruling); HCA log-ratio; raising `--mcts-sims`; gumbel_m 16→8; λ-returns.

**Two "green" claims from the leg-4 brief are contradicted by the final CSV and should not be repeated:** "draw share w ~0.1" (reaches 0.302 / 0.286 / **0.656** / 0.382 / 0.365 at iters 21–25) and "~96% decisive" (358/359 for iters 0–14 but **221/258** for iters 15–25, and 13/22 at iter 23 alone). The human-probe CE also was not green in the sense reported: it rose monotonically for 12 of 14 points and finished **0.006 nats under** its own abort bar.

---

## 3. Pre-registered discriminating experiments

Ordered by cost. **E1 is free and should run before anything else** — it settles the confound in the leading hypothesis's core evidence, and it is the one measurement the entire leg never took.

> **Prerequisite (5 min):** the 495k pin (step 3,304,339) is **not** in `training/checkpoints/` locally — only `seed_imit_tierb_start.pt` and `tier_b_tcs2_leg3_end.pt` are. Pull `tier_b_l4.pt` from the HF escrow first.

---

### **E1 — Frozen-state policy shape: weights-drift vs state-drift**
**Cost:** free, local CPU, ~1 hour. **Tooling:** ~50 lines reusing `wesnoth_ai.action_sampler` + `trainer._masked_actor_logits` (`trainer.py:769`) over the existing 1,200-pair human-probe holdout (`replays_dataset_imitation`) plus ~2,000 frozen iter-0 self-play coordinates. No new games, no forks, one forward pass per state per checkpoint.

Score **seed** and **pin** on **identical** states. Report per state: `n_legal`; masked policy entropy H; H / log(n_legal); top-1 mass; share of states with top mass > 0.8 (same gauge as `mcts_policy.py:678`); p(end_turn); policy mass by action type vs the legal set's own type composition; and Spearman ρ(p_actor, n_actor_legal) against a within-state permutation null.

| | **If R1 is real (weights drift)** | **If it was state drift** |
|---|---|---|
| pin H / log(n_legal) | ≥ 0.95, materially above seed | within noise of seed |
| pin top80 | near zero, ≪ seed | comparable to seed |
| type-mix vs legal-set composition | pin matches within a few points **except** end_turn | pin retains seed-like attack/recruit skew |
| ρ(p_actor, n_actor_legal) | pin **rises sharply** above seed (p^λ makes "which unit acts" a mobility ranking — the mechanism behind attack 19.9→4.5 with K flat) | no systematic coupling |
| p(end_turn), frozen states | ≈ seed's (anchor-pinned) | ≈ seed's |

This also closes **OPEN #2** (weights-drift vs state-drift, prescribed as leg-3 M1 and never run) at zero extra cost, and reports `n_legal` — which nobody has measured, and without which "prior entropy 5.15 = near-uniform" is an unbacked claim.

---

### **E2 — Single-salt advantage ICC on the seed head: is the target channel noise?**
**Cost:** free, local CPU, ~1–2 hours. **Tooling:** `tools/turn_counterfactual_probe.py` (the A4/rung-0 harness; `--checkpoint`, `--states`, `--reval-salts`, `--n-alt`, `--rounds`, `--no-placebo`) driving `turn_search.plan_turn` with `full=True`, plus a small patch to re-grade the *same* alternative set at 3 extra independent salts.

Compute the intraclass correlation of the realized `(q_i − LOO_mean)` that the **target** consumes at `turn_search.py:520-527`: ICC = between-alternative variance / total variance.

| | **If R2 is real** | **If the target channel carries signal** |
|---|---|---|
| ICC on seed head | ≤ 0.15 — the single-salt advantage is mostly salt noise even though the 3-salt gate separates 6.9:1 | ≥ 0.40 |
| implication | R1+R2 conjunction confirmed; the fix is **both** λ=1.0 **and** paired/replicated grading in the target pass | leg 4 becomes a pure λ story; R2 dies |

Run this on the **seed**, not the pin — the question is whether the channel was ever informative, and R2's own signature says it was near-null from iteration 0.

**E2b (same harness, +30 min):** run `turn_counterfactual_probe.py` for **pin and seed on the same state set** to remove A4's state-distribution confound and get the only direct read on whether the pin's value head degraded in-leg. Note this measures the **3-salt gate**, not the target channel — report it as such.

---

### **E3 — Search-off matches: is it the prior or the value head?**
**Cost:** ~$2–3 on the cheap CPU eval box, or overnight local. **Tooling:** `tools/run_elo_batch.py --mcts-sims 0` (verified: `elo_eval_game._build_player` returns the raw policy when `sims == 0`, `tools/elo_eval_game.py:50-71`). 20 games per arm, same seed list, pin vs seed at sims ∈ {0, 32}.

| | **If R1+R2 (flat prior)** | **If the value head is the problem** |
|---|---|---|
| pin at sims=0 vs sims=32 | sims=32 **better** than sims=0 (a working head picking best-of-16-random beats 1 random) | sims=32 **worse** than sims=0 (search amplifies a bad ranker) |
| seed degradation from removing search | small (its prior is peaked) | small |
| pin degradation from removing search | **large** | small |

This is the cheapest arm that separates the surviving conjunction from any residual value-side story, and it is a genuine prediction rather than a re-confirmation. Do **not** run it before E1: if the pin's prior really is near-uniform, both arms are catastrophic and the contrast may not resolve.

---

### **E4 — λ=1.0 three-iteration sign test (confirmatory only)**
**Cost:** ~$3, 1–2 box-hours. **Run only if E1 or E2 comes back ambiguous.**

Restart 3 iterations from the seed with the **exact** leg-4 cmdline minus `--distill-prior-discount` (i.e. λ=1.0), changing nothing else. Primary readouts: `distill_prior_entropy` and `distill_prior_top80`.

| | **If R1 is the decay agent** | **If not** |
|---|---|---|
| `distill_prior_entropy` over 3 iters | ~stationary (simulation: λ=1.0 is exactly stationary under **either** grader quality) | continues rising ~+0.08/iter |
| `distill_prior_top80` | holds near seed level | continues falling |

Three iterations suffices: leg 4 moved +0.46 nats over its first three. Caveat carried from review: the unmasked BC anchor is itself an entropy-affecting force, so this arm is not perfectly clean — which is why it is confirmatory, not primary.

---

### **Ship regardless of any result — free, and required before any leg 5**

1. **Wire `harvest_boundary_pairs(exps)` into the actor-pool drain** at `tools/sim_self_play.py:1422-1433`, mirroring the spool ingest at `:1276`. This instrument has now silently read n=0 through **two** full campaigns (2026-07-29 spool path, 2026-08-19 pool path).
2. **Add to the CSV field list** (`:2295-2325`): `tcs_plans`, `tcs_accepts_per_plan`, `tcs_replans_per_plan`, `search_q_spread`, `search_overturn_frac`, `gbc_loss`, `aux_loss`.
3. **Arm a trend tripwire on `distill_prior_entropy` and `distill_prior_top80`, and a slope (not level) tripwire on probe CE and `actor_top1`.** Every existing tripwire — CE level, AUC floor, decisive rate, K — is blind to this failure **by construction**. The leg recorded its own collapse in columns nobody watched, and the level bar it did watch was missed by 0.006 nats.
4. **Remove the unconditional `--distill-prior-discount` emit** at `scripts/vast_onstart.sh:786`; env-gate it like the anchors at `:776-781`.
5. **Make the leg config file assert the rulings** rather than pin 4 of 13. A launch that violates "one prior protection per leg" should fail at startup, not 495k steps later.

---

## 4. The single most damning unexplained fact

**In the final five iterations the flattening partially reversed — and the checkpoint that lost 0–16 is from the reversed regime.**

| iter | 21 | 22 | 23 | 24 | 25 |
|---|---|---|---|---|---|
| `distill_prior_entropy` | **5.147** (peak) | 5.006 | 5.101 | 4.904 | **4.830** |
| `distill_prior_top80` | 0.0200 | 0.0217 | 0.0212 | 0.0292 | **0.0310** (+55%) |
| `distill_et_prior` | 0.0575 | 0.0547 | 0.0547 | 0.0671 | **0.0678** |
| probe CE (last 3 points) | — | — | 3.701 | 3.672 | **3.582** |

A monotone ratchet cannot do this. Four independent channels — the model's own prior entropy, its share of confident decisions, its end_turn mass, and the human-holdout CE — all turned around over roughly the last 15% of the leg, recovering a meaningful fraction of what they had lost. Meanwhile the *value-side* channels kept degrading through exactly that window (`value_signal_states` 1601 → 1038, `mean_turns` peaking at 54.96, caps 5/5/6 per iteration, `z_draw_frac_w` 0.302 → 0.656 → 0.382 → 0.365), and those last ~5 iterations contributed roughly **180k of the leg's 495k steps**.

Neither survivor explains this. R1 is monotone by construction. R2 is stationary by construction. And it matters operationally: **the pin we measured at −277 Elo is the one whose policy flattening had partly recovered**, which is weak evidence that by leg end the terminal driver had handed off from the policy channel to the value channel — the one arm we ruled *out* as the originating cause but which was demonstrably accelerating at the end. `[measured; interpretation suggestive]`

**Runner-up, same region of the record:** the leg absorbed ~43% of the taught entropy change but only ~3.2% of the taught end_turn change — a **13× inconsistency** in how much of the *same* target, taught through the *same* CE, reached the weights. (`distill_et_target < distill_et_prior` at 26/26 iterations, yet realized `et_prior` is flat 0.0746 → 0.0678.) The best code-supported repair is that the F1 anchor's hard-label BC steps pin the end_turn actor slot — ~8% of human actions, one slot — while barely constraining a ~200-way move-target distribution. That would make K's greenness an artifact of the **anchor**, not of the search, and it is untested. E1 measures it directly.

**Honest framing for the log:** we have established a **force** and its **direction** with high confidence, and its **config-drift provenance** with certainty. We have not established its **effect size on the weights**, because not one measurement in this leg was taken on a fixed state set. Every number in the dossier is either a mean over the run's own drifting self-play states or an unmasked BC statistic on human states. E1 costs nothing and closes that hole.

---

## E1 RESULT (2026-08-20, run same day — tools/policy_shape_probe.py)

1,200 identical frozen human-holdout states, paired design, real
enumeration path (`training/metrics/policy_shape_20260820.json`):

| metric | seed | pin (495k) | R1 prediction | verdict |
|---|---|---|---|---|
| H (nats) | 2.924 | 3.949 | pin above seed | **+1.03 nats on frozen states** |
| H / log(n_legal) | 0.528 | 0.706 | >= 0.95 | direction yes, magnitude partial |
| top80 share | 0.138 | 0.040 | pin << seed | **3.5x collapse** |
| top-1 mass | 0.349 | 0.232 | — | down |
| attack mass (legal share 0.033) | 0.267 | 0.187 | toward legal comp. | down 30%, still 5.7x over legal |
| recruit mass (legal share 0.081) | 0.083 | 0.163 | toward legal comp. | **ANOMALY: moved AWAY from uniform, 2x legal** |
| rho(actor mass, mobility) | 0.353 | 0.417 | rises sharply | rises moderately |
| p(end_turn) | 0.070 | 0.074 | ~= seed (anchor-pinned) | flat, as predicted |

**Verdict: the state-drift confound is DEAD — the flattening lives in
the weights.** On identical states the pin is +1.03 nats more
entropic with confident decisions down 3.5x. R1's direction is
confirmed on every axis except one; the magnitude is partial
(H_norm 0.71, not ~1.0), consistent with the synthesis's "force
established, effect size not," and with the pin coming from the
partially-recovered tail regime.

**Open anomaly:** recruit mass DOUBLED (0.083 -> 0.163, 2x its legal
share) — pure flattening cannot do that; something in the leg
actively taught recruit preference. Unexplained; note the telemetry
showed recruits/game *falling* (7.6 -> 2.1) in self-play — mass up
on frozen states, realized recruits down in vivo.

(Also corrects a workflow assumption: mean n_legal on human-holdout
states is ~386, not ~200.)
