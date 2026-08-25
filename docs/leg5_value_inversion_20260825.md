<!-- Provenance: 17-agent Opus workflow wf_b3c310c4-43a, 2026-08-25.
The verify/synthesis agents RAN the decisive measurements locally
(head grafts, probe replication to 7 digits, channel decomposition);
artifacts in the session scratchpad, key dumps re-runnable via the
dump_ev*/graft.py scripts noted at the end. -->

Analysis complete. Final synthesis:

# Leg-5 value-AUC inversion — synthesis

**All six pre-registered hypotheses were refuted.** I then ran the decisive measurements offline. The question is answered: **the leg-5 trunk (not the value head) represents position quality in a basis where raw unit count is weighted positively and cost-weighted material negatively; on human ladder games that direction is anti-correlated with the winner, so the ranking inverts.** The probe instrument is also badly broken, but separately.

Everything below tagged `measured` was computed this session on local data; scripts and dumps are in
`C:\Users\amaur\AppData\Local\Temp\claude\C--Users-amaur-Desktop-Perso-projects-Wesnoth-AI\a6979d51-1bba-4acf-bd8d-2928a72e0ebf\scratchpad\`.

---

## 0. Instrument identity confirmed first

`measured` — Replaying the probe's exact 1,200-pair stream offline on the leg-5 seed (`training/checkpoints/seed_imit_tierb_start.pt`, decision_step 2,809,659) gives pooled AUC **0.7003136**, against the in-vivo launch read of **0.7003137** (`tier-b/holdout_probe.csv` on HF, row 1). Seven-digit agreement. Every decomposition below is of the real statistic, not a lookalike.

`measured` — The pool is 3 games: Micro_Isar (winner side 1) 58 pairs, Aethermaw (w1) 217, Caves_of_the_Basilisk_Turn_38 (w2) 925. 674 winner-mover / 526 loser-mover = 354,524 comparisons, of which **63.0% are within-game and 59.6% are inside the Basilisk game alone**.

---

## 1. Ranked surviving mechanisms

### M1 — Trunk-level proxy rotation onto unit count (`measured`, primary)

The head-graft settles trunk-vs-head. Caves_of_the_Basilisk, uniform stride-4, n=232, winner = side 2:

| arm | AUC | t1-10 | t11-20 | t21-30 | t31-40 | pc(V,count \| material) | pc(V,material \| count) |
|---|---|---|---|---|---|---|---|
| seed trunk + seed head | 0.584 | 0.000 | 0.329 | 0.895 | 1.000 | +0.415 | +0.393 |
| leg-4 trunk + leg-4 head | **0.962** | 0.837 | 0.926 | 1.000 | 1.000 | **−0.724** | **+0.874** |
| leg-5 trunk + leg-5 head | **0.338** | 0.000 | 0.148 | 0.388 | 1.000 | **+0.662** | **−0.289** |
| leg-5 trunk + **seed** head | 0.347 | 0.000 | 0.154 | 0.410 | 1.000 | +0.673 | −0.259 |
| **seed** trunk + leg-5 head | 0.577 | 0.000 | 0.318 | 0.890 | 1.000 | +0.417 | +0.372 |
| *[material margin alone]* | 0.887 | 0.381 | 0.849 | 1.000 | 1.000 | — | — |
| *[unit-count differential alone]* | 0.339 | 0.025 | 0.060 | 0.309 | 1.000 | — | — |

Swapping the **head** moves AUC by ≤0.01 (corr of the grafted output with the plain output: 0.998 and 0.997). Swapping the **trunk** moves it completely. Q1's "the trunk is the cap" is confirmed for this failure specifically.

The mechanism: on this human pool, cost-weighted material reads AUC 0.888 and raw unit count reads 0.339 (human winners trade down into fewer, higher-value units). In leg-5 self-play both channels predict `z` — the leg-4 postmortem substrate has the winner ahead on material 24/24 and on units 24/24 at turns ≥10 — so keying on count instead of value is **free in-distribution** and inverting on human play. `train_value_loss` falls 0.727→0.467 across exactly the window where the human ranking inverts, because the head is genuinely fitting self-play better.

Leg 4 rotated the *opposite* way from the same seed (pc(count|material) −0.72, the human-correct sign) and reads **0.9122** pooled on the full 1,200. So this is a rotation with a sign, not decay.

Fit against the five binding constraints: inversion not drift ✓ (an anti-correlated direction, quantified); speed ✓ (a re-weighting between two channels correlated at +0.77, not new feature learning); healthy train_value_loss ✓ (by construction); precedents ✓ (leg 3's sub-chance entry, leg 4's swings = which way the rotation lands); A3 fit on a frozen trunk ✓ (the graft shows the head is now irrelevant).

`suggestive` — The leg-5 readout is also *shallower*: standardized OLS of E[V] on 7 board channels (507-state uniform set) gives R² seed 0.803, **leg-4 0.536, leg-5 0.677**. Leg 4's trunk moved toward reading something the simple channels don't capture; leg 5's did not.

### M2 — The instrument's absolute level is uninterpretable (`measured`, co-primary for the decision)

Not a cause of the delta, but it invalidates the tripwire that killed the leg.

- `measured` The seed's 0.700 is a mixture: **turn ≤15 → 0.4931 (chance)**, turn >15 → 0.9446. On the game holding 77% of the pairs it is **fully inverted in turns 1-10 (AUC 0.001)** and only rescued by turns 21-40 (0.904, 1.000). The "qualify PASS at 0.700" certified a judge that is at chance on the early game.
- `measured` Re-sampling the *same three games* more evenly (507 states instead of 1,200) moves the seed from 0.7003 to **0.8345** — +0.13 from sampling alone, on the same model.
- `measured` The ±0.017 Hanley-McNeil SE (`tools/supervised_train.py:948-956`) assumes 674×526 independent samples; n_eff is 3 games. "Below chance, CI-solid" is not established by this instrument.
- `measured` `tools/supervised_train.py:886` never passes the `max_pairs_per_replay` kwarg that `_pair_stream_serial` already supports (line 194-204), so the stream is first-1200, file-sequential, uncapped.
- `measured` 364 of the 369 manifest-holdout games are inside the A3 value head's own training split (`tools/value_head_fit.py:224-232` splits its own index, ignoring `manifest.jsonl`). Now largely moot given the graft result, but the baseline was never clean.

### M3 — Level/offset channels (`measured`, bounded, not the driver)

- Side-bit ablation (forcing `encoder.py:1249`): seed V(side2)−V(side1) = **−0.270 ± 0.148**; leg-4 pin **−0.006 ± 0.084**. The channel is real and lives in the *seed*; self-play removes it.
- Sensitivity sweep on the seed's full 1,200: a constant offset on every side-1-to-move state moves pooled AUC 0.700→0.573 at +0.40, and would need ≈+0.9 to reach 0.43. So the channel caps out around 0.13 AUC — real, but an order too small.

---

## 2. Ruled out, and by what

| Ruled out | Killed by |
|---|---|
| **"Benign comeback game"** (a correct positional head must read <0.5 there) | `measured`: the winner is materially **ahead** — material reads AUC 0.885 on that game, 0.888 pooled, villages 0.909. The pool's natural reading is ~0.9, not <0.5. |
| **Random forgetting / drift toward chance** | `measured`: iid Gaussian noise of sd 0.8 (1.4× the E[V] sd) on the seed only reaches 0.62. Noise cannot reach 0.41. |
| **Absolute-side offset as the driver** | `measured`: needs c ≈ +0.9; measured causal side-bit effect is −0.006 in the post-self-play checkpoint (and −0.27 in the seed, wrong leg and wrong sign). |
| **Per-game-constant offset (map/faction prior)** | Arithmetic: 63% of pair mass is within-game and cancels exactly; max attainable excursion from arbitrarily large per-game constants is 0.051 AUC. |
| **Sign/perspective inversion anywhere in the pipeline** | Code, 4 sites: `turn_policy.py:95,142`, `mcts_policy.py:586-589`, `value_corpus.py:136-148`, `supervised_train.py:898-902`. All mover-frame, all agree. Encoder is mover-relative (`encoder.py:12,997`). |
| **Label smoothing / weighting asymmetry** | `trainer.py:678-682` uniform over K atoms, sign-symmetric; weights at :1224-1228 depend on \|z\|, not sign; game_weight is per-side equalized. |
| **Boundary-pair telemetry as a training signal** | `mcts_policy.py:1006-1073` appends to a deque and computes under `no_grad`; nothing in `trainer.py` reads it. |
| **Aux-material axis capture** | `measured`: material reads 0.888 on the pool, so absorbing it *raises* AUC. Also `--mcts-aux-score` is emitted unconditionally (`vast_onstart.sh:771`) — identical in legs 3/4/5. |
| **Fresh-head burn-in overwrite** | Telemetry: `gbc_loss` and `aux_loss` floor at iter1; the 11-SE AUC move is iters 2-3, after. Identical burn-in in leg 4, which ended at 0.912. |
| **GBC fog-censoring asymmetry** | `measured` on 11,328 roster entries: own-unit `dies` positive rate 1.66%/5.67% vs **enemy 2.16%/6.18%** — asymmetry runs the other way. Exposure's residual-after-material AUC is 0.497. |
| **Anti-attrition / anti-material sign** | Self-play winners end ahead on material 24/24, villages 22/24 — no gradient for it. (The surviving variant is M1: *pro-count*, not anti-material.) |
| **A2 rehearsal being "inert" as the enabling condition** | Leg 4 ran with A2 **off** and read 0.912; leg 3 entered below chance. Also the CE-vs-floor argument is invalid — a perfect-AUC C51 head scores CE 2.5-6.0 against delta targets, far above the 0.85 "floor". |
| **Grader-null / head stopped ranking** | E2: sighted ICC 0.958-0.995, blind fraction 0-8%. |
| **C51 quantization; draw-flood starvation; phase-support collapse** | V is an expectation over atoms (continuous); `z_draw_frac_w` ~0, `value_signal_states` 1978/2048; and leg-5 games run 24.6-28.6 turns with `ended_max_turns` 0, while the pool is 55% turn ≤15. |
| **"Self-play erodes the human value ranking"** as a general law | `measured`: same seed + 495k leg-4 self-play steps → **0.9122** (up from 0.7003), turn ≤15 0.4931→0.7705. |

---

## 3. Cheapest pre-registered discriminating experiments

**Already done this session** (no further cost): head graft, per-game/per-turn decomposition, channel AUCs, side-bit ablation, offset and noise sensitivity. Results above.

### X1 — Game-stratified, leak-aware probe (free; ~1-2 h laptop CPU, minutes on a box). **Blocking.**
Pass `max_pairs_per_replay=8` at `tools/supervised_train.py:886`, `--eval-pairs ~2952`, and replace the Hanley-McNeil block (:948-956) with per-game AUC + between-game SE. Run on seed (2,809,659), leg-4 pin (3,304,339), leg-5 kill (2,931,890). Record as a **new** series; the CE column's sampling changes too.

| outcome | reading |
|---|---|
| leg-5 mean ≈0.4-0.5 with tight between-game SE, seed ≈0.7-0.8 | M1 confirmed and general; the leg-5 head really is inverted on human play |
| leg-5 mean ≈0.55-0.65, wide between-game spread, Basilisk an outlier | the 0.41 was a 3-game artifact; the tripwire fired on noise, M2 dominates |
| all three checkpoints ≈0.5 with huge spread | the whole human-holdout AUC family is unusable; switch gate |

### X2 — Channel-decorrelation readout (free; rides X1's forward pass)
Per checkpoint, over the stratified games: **pc(E[V], unit-count differential | material margin)**. This is the statistic that tracked AUC perfectly across all three checkpoints (+0.415 / −0.724 / +0.662 → 0.584 / 0.962 / 0.338), is offset-free, and needs no cross-game comparison.

| prediction | verdict |
|---|---|
| seed ≈0, leg-4 clearly negative, leg-5 clearly positive | M1 confirmed; adopt pc as the gate |
| all three ≈0 across many games | M1 is a 3-game artifact; fall back to X1 |

### X3 — Self-play holdout vs human holdout on the same checkpoint (free)
The leg-5 self-play holdout sidecar is escrowed (`tier-b/tier_b_l5.pt.holdout`). Compute the same winner-mover AUC on it and on the stratified human set.

| prediction | verdict |
|---|---|
| self-play AUC high (≥0.8), human AUC low | distribution-specific proxy — M1's exact signature; the head is not broken, it is fitted to the wrong distribution |
| both low | genuinely broken value path; re-open the ruled-out list |

### X4 — Which leg-4/leg-5 delta caused the opposite rotation (~$3-4, 8 iterations)
The only value-side flag difference is **A2 human value rehearsal (OFF in leg 4, ON in leg 5)**; the other candidate is **λ 0.9→1.0** (leg 4's distillation was ~97% flattening, so its trunk was shaped mostly by value+aux+GBC). Two 8-iteration arms from the seed, probed with X1/X2 each iteration: (a) A2 OFF, λ=1.0; (b) A2 ON, λ=0.9.

| arm result | reading |
|---|---|
| (a) holds pc ≤0 and AUC ≥ seed; (b) inverts | A2's 4-of-20 full-unfreeze pure-value steps are shaping the trunk shallow — drop or restructure A2 |
| (b) holds, (a) inverts | the informative distillation gradient is doing it — λ or the distill target is the lever |
| both invert | it is a transient of early self-play from this seed; leg 4 passed through it and recovered — extend the tripwire window instead of changing training |

### X5 — Config-only counter, if M1 confirms (no new mechanism needed)
Add unit-count differential as a **second** aux regression target next to `material_margin` (`mcts_policy.py:598`), so the trunk must represent count and cost-weighted value separately and the value readout cannot conflate them. Prediction: pc(count|material) moves toward or below 0 within ~4 iterations; if it does not, the aux head is too weak a lever and the value target itself must change.

**Do not spend on:** re-fitting the A3 head, freezing the value head, or a `--gbc-coef 0` ablation. The graft shows head surgery moves the statistic by <0.01, and the GBC label rates are measured in the wrong direction.

---

## 4. Resume criteria

**Blocking, before any resume:**

1. **Retire the pooled 1,200-pair AUC floor.** `PROBE_AUC_FLOOR=0.52` (`scripts/holdout_probe_loop.py:74`) fires on a statistic that is 60% one game's internal comparisons, whose SE is wrong by roughly an order of magnitude, and whose level moves ±0.13 under benign re-sampling of the *same* games. Ship X1 and re-derive the floor from the seed's own stratified read (seed-level minus 3 between-game SE), not from 0.5.
2. **Re-qualify the seed on the stratified statistic.** The 0.700 gate certified a judge that is at chance below turn 15 and at AUC 0.001 on the dominant game's first ten turns. `QUALIFY_AUC_MIN` should be set against the stratified number.
3. **Wire the qualify gate into the launcher** — BACKLOG already records it is run by hand for legs 4 and 5 (`grep qualify scripts/vast_onstart.sh` returns nothing).

**Then resume behind a second, offset-free gate:**

4. Add **pc(E[V], unit count | material margin)** on the stratified human holdout to the probe row, with an abort on 3 consecutive readings > 0 (leg 4's trunk achieved −0.72; the seed sits near +0.4, so the seed itself will not pass a strict version — set the initial bar at "not increasing from the seed's entry value").

**What would prove it fixed** (all four, on a fixed state set, pre-registered):

- Stratified per-game human AUC at iteration N ≥ the seed's stratified entry level, within between-game SE — i.e. the leg is not trading human-play value discrimination for self-play fit.
- pc(count | material) ≤ 0, or at minimum not rising from entry.
- `train_value_loss` still falling and `value_signal_states` ≥ ~1900/2048 — the fix must not be bought by weakening the self-play value signal.
- Existing guards unchanged and green: CE within +0.5 of t0 (3.207), K median ≥10, decisive rate, `distill_prior_entropy` trend.

**Honest gaps.** (a) Only the rolling leg-5 checkpoint (2,931,890) is escrowed, so I could not measure whether the rotation is a transient — X4's third branch is live. (b) `tier-b/holdout_probe.csv` on HF now holds leg-5's 6 rows only; the leg-4 probe series was overwritten, so the claimed 0.41-0.92 leg-4 oscillation is still undocumented — what *is* documented now is that the leg-4 pin at +495k measures 0.9122 offline. (c) I did not run the side-bit ablation on the leg-5 checkpoint; the graft and offset sweep bound that channel below the effect size, but it is not directly measured for leg 5.

**Artifacts** (absolute paths, all in the scratchpad above): `seed_recs.json`, `l4_recs.json` (full 1,200-state E[V] dumps, format `[game, winner, mover, ev, turn]`), `l5_recs.json` + `l5_bas.json` + `l5_plain_s4.json` (leg-5), `graft_L5trunk_seedhead.json`, `graft_seedtrunk_L5head.json`, `h2_rows.json` (per-state board channels for the same 1,200), `dump_ev2.py`, `dump_ev3.py`, `graft.py`, `tier_b_l5.pt` (downloaded, step 2,931,890).
