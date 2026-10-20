# Round 7, provenance stance: the coin-label design re-priced, its null readings freed of the Gaussian shape

The core of `current_design.md` stands: near-tie coins with logged propensities, labels exactly unbiased through the simulator's laws, a closed-form ridge head on frozen `own1`, adoption only by a PURE paired match. I audited every number it rests on (table at the end), reran `round6_synthesis_power.py`, `round6_provenance_cells.py`, `round6_simplify_pairs_raw.py` and `round6_provenance_critique_attack_stage1.py` (identical outputs) and ran three new analyses. Five things change:

1. **Price.** A long batch pays the steady-state rate, $1.91e-5 a game, not the $2.51e-5 whole wall of a 600-game arm. Every stage together costs $61.59 of $85.
2. **DPI is deferred.** The plan fits without it; it would save at most $1.91–3.82 a round, and its code sits in the label path. S0(a) shrinks to a $0.04 census.
3. **A shape-free ceiling.** For near-tie advantages of any shape with nonnegative mean, the sign rule's first-order gain grows at most as √N in the labels. Without the Gaussian model, this bounds the interim stop's error (P ≤ 0.041 at an adoptable round) and a same-band second round (≤ 0.41 of round 1's gain).
4. **One round per span.** A same-band round 2 can be certified only if round 1 gained ≥ 8.6 Elo. In the nonnegative regime it runs only then; otherwise the reset (S4) follows round 1.
5. **Smaller fixes.** The mirror null becomes exact; S5's claim is a two-network match (+$3.13); the d_eff = 200 scenario, which had no source, is dropped.

## 1. What the records and derivations settle

### 1.1 The price a long batch pays

**Steady rate.** `analysis/round7_provenance_price_tail.py` reads every completion line of the 120-job arms in `generation_speed_r5_20261015`.
- Between the 120th and 480th completions all 120 slots are busy, and the reference path completes 21,966 games an hour: **$1.912e-5 a game**.
- The bit-identical priors-only arm reads the same, and the 160-job arm 21,873, to the logs' one-second resolution (±1.7%).
- The 129-s arm wall adds 16 s of start-up, a 20-s last wave and the server's shutdown, which a 4.55-h batch does not pay. Round 6's price was 1.31× high.

**Overhead room.** The same arm's stage table: a worker's decision cycle is 39.3 ms, of which 33.2 ms waits on the server; workers use 7.1 of 31 cores; the server is busy 0.71 of its wall. Added worker work costs at most its share of the cycle:
- the luck walk (0.321 s a game, measured): ≤ 1.8%;
- the per-game sufficient statistics (about 9 × 10⁷ multiply-adds): under 0.1% [D].
The server's feature gather is unmeasured; S0(c) reads the total as the ratio of two steady windows on the rented host, coin smoke against S0(b)'s plain games.

**Budget** (`round7_provenance_budget.py`): batch $1.91; 32,000-game match $0.61; round 1 $15.91; a null round 1 $6.63 (stopped at batch 3) or $16.80. All stages $61.59; without S3 $44.46, leaving $40.54 (1.53 more reset-and-round cycles). On a 1.41× slower host: $82.78 and $58.62.

**DPI deferred.** Admission saved one batch a round at r ≤ 0.602 ($1.91) or two at r ≤ 0.516 ($3.82) [D, `round6_synthesis_power.out` §5]. No stage needs that money, hand-written forms stopped near 0.69, and DPI adds per-fight pricing, token extraction at attacks, a 338-weight fit and two centring invariants to the label path. Its $0.38 replay goes with it. It returns in the $1,000 case, where a 10–20% cut of coin spend pays for its code; its reading's paired SE is known (0.012–0.015 on 800 `own1` games, `round6_invent_critique_checks.out`).

### 1.2 What more labels on one span can buy

Round 6 states the interim stop's meaning, round 2's value and what a null exhausts under a ~ N(μ, v). They can be stated without the shape.

**Claim.** A student fitted on N labels predicts f = a + e, e Gaussian with variance η ∝ 1/N (OLS on a fixed span). The sign rule gains g(η) = E[a Φ(a/√η)] per band decision.
- With s = √η, t = a/s: d(s·g)/ds = E[a·w(t)], w(t) = Φ(t) − tφ(t), increasing (w′ = t²φ), in (0, 1).
- If E[a] ≥ 0: E[a·w] = E[a]E[w] + Cov(a, w(a/s)) ≥ 0, since a and w(a/s) both increase in a (Chebyshev's association inequality).
- So s·g is nondecreasing, and for any shape with mean ≥ 0: **g(N′) ≤ √(N′/N) · g(N) for N′ ≥ N.**

It holds when the noise scale varies independently of a (apply it per scale). It is exact for OLS sign decisions and approximate for the held-out ridge. With E[a] < 0 it can fail: the value of a noisy signal is then convex at low precision (Radner & Stiglitz 1984; [Chade & Schlee 2002](https://ideas.repec.org/a/eee/jetheo/v107y2002i2p421-452.html)).

**Check** (`round7_provenance_scaling_bound.py`): eight mean-zero shapes (Gaussian, Laplace, t₃, uniform, two skewed, two sparse) at sd 0.002 and 0.003, with equal and varying noise. g8/g3 spans 1.06–1.58 (bound 1.633), g16/g8 1.00–1.35 (bound 1.414). With a negative mean the Gaussian posterior-mean rule exceeds the bound in 4 of 6 cases: g16/g8 = 1.44–6.33 at μ = −0.0018, 1.71 at μ = −0.0005 and sd 0.002.

**Consequences** (η₈ = 2.34e-5):
- **Interim.** If student 8 would be adoptable at 80% power (5.1 Elo), student 3 gains ≥ √(3/8) × 5.1 = 3.1 Elo, so the stop fires with P ≤ 0.041, whatever the shape (first order; curvature can lower J3).
- **Same-band round 2** (16 batches pooled). Its first-order increment is ≤ 0.414 × round 1's gain; the Gaussian model gives 0.19–0.36. π_2 against π_1 reaches the adoption line (3.6 Elo) at that ceiling only if round 1 gained ≥ 8.6 Elo (≥ 12.3 for 80% power). Re-pricing under π_1 is not bounded: it is a curvature term like the forecast miss, whose scale S2 reads.
- **Exhaustion.** Doubling the labels buys at most 41% of a gain the round could not see; in this regime the exhaustion rule rests on that.

**The regime is read in batch 1**, from the mean label per margin bin: SE 0.00085–0.00145 per 0.1-nat bin, 0.00049 for the band (same script, §4). Continuity forces same-kind means to 0 at a tie (moves lead 66 of `parity3`'s 91 band decisions); away from ties the argmax's runner-up is likely worse. A bin reading negative by 2 SE is convex: only the Gaussian model, fed the measured mean, forecasts more labels there, tagged model-bound.

### 1.3 The mirror null becomes exact

Game records hold the command stream, every fight's law and a state digest per turn (read in `games_gen_own1_selfplay`). In a seed-shared mirror pair, identical command prefixes must give identical digests, whatever the numerics: a mismatch is a seed or dice bug. The first divergence must be a command, and its logged gap is binned against the cross-box record (flip rates 0.53, 0.12 and 0.012, 95% up to 0.042, per reading in [0, 0.0136), [0.0136, 0.1), [0.1, 0.35); none resolved at ≥ 0.35; `round5_simplify_critique_extra.out` §1). A first divergence at G ≥ 0.35 is investigated before S1. At 1.76% flips a decision, pairs stay identical for about 57 decisions; the attack known answer covers the rest.

### 1.4 Smaller corrections

- **Attack known answer.** The nearest knob vector, kv23 (attack +0.84, end_turn −1.51), reads −0.240 ± 0.097 z over 100 games (−83 ± 34 Elo; `round5_provenance_knob_switches.out`), against the stage-1 fit's −0.189. The −78 ± 16 prediction at +1 stands.
- **d_eff.** The held-out λ search includes λ = 0, so floors at d = 776 are the conservative ones. The 200 case is dropped; S0(a)'s spectrum becomes a report.
- **S5's claim** (reset network against `own1`) is two-network: $3.74 at the eval median. Round 6's three-round line becomes $2.81e-5, not $2.92e-5.

## 2. The algorithm

**What is learned.** A switch head on frozen `own1`, f = c_uv + b_uv·G + ŵᵀx, x = φ(a1) − φ(a0), predicting the advantage of the runner-up a1 over the top choice a0 where the post-offset gap G < w = 0.35 [C]. φ concatenates the actor token, q(actor) ⊙ k(target hex), kind indicators and the four factored log-probabilities (d = 776). The network picks every action; the head reorders two of its candidates at one decision, and play re-decides after every roll.

**Labels.** π_0 is `own1` at `raw:t0+eo-1.5`, 64 slots, on both sides in the pool. At each band decision a coin keyed by seed and decision index plays a1 with P_i = 0.25 [C]:

    T_i = s_i ω_i R_i,   ω_i = (D_i − P_i)/(P_i(1 − P_i)),   R_i = z − b(cell) − βᵀS − γ s_i V_i

- z is side A's result and s_i the mover's sign. b is the full setup cell's mean, from earlier batches (batch 1 cross-fits). S holds the walk's seven luck sums, V_i the value head before the coin; β and γ are fitted outside the batch.
- **Unbiased:** E[T_i | H_i] = Q̃(H_i, a1) − Q̃(H_i, a0) under the coin policy, for any such coefficients, since each baseline term is fixed by H_i or has mean zero given a past containing D_i. Under fog it is the information-state advantage: no strategy fusion.
- **Variance:** the ω's are independent with mean zero, so labels sharing a game are uncorrelated; Var T ≈ E[R²]/(P(1 − P)) = 3.67. 76.1 band decisions per side-game give 15.2 M labels a batch.

**Update.** Closed-form ridge from per-game sufficient statistics. λ by progressive validation: batch k+1 is generated with student k logging f, so its labels score student k out of sample. Typed terms obey c_vu = −c_uv, c_uu = 0, b_vu = b_uv. Switch where f > 0.

**Why more signal per dollar.** Every band decision is a label whichever way its coin falls, and one result prices about 152 of them: 39.0 information units per game-equivalent against 0.82 for fork pairs [D, round 5], at the generator's own price, with no playout and no critic. The laws remove luck without bias; the cell removes the setup. Correctness depends on no unknown; only the gain depends on the spread v.

**Certification.** A PURE paired match, seed-shared and side-swapped: 32,000 games, SE 1.79 Elo, $0.61. The forecast Ĝ_pf (batches 5–8's labels summed at students 4–7's switches) is sealed first. Adopt iff the lower 2-SE bound > 0 (false adoption 0.023). A miss beyond about 9.9 Elo is second order (curvature, or the opponent's coins; `current_design.md` §3.5) and leads to fallback E, the same student at half its switches.

**Planted signals and invariants per code path.**
- **Label:** P-coin per mover side in batch 1 (δ set after S0(c) measures Var(u·x)); I2, the exact duel with the coin on each seat; I1, I5, I9.
- **Student:** I6, I7, I-φ.
- **Match:** the exact mirror null (§1.3); attack +1 through the per-seat hook, −78 ± 33 Elo.
- **Distillation:** P-distill, gated on the direction of the end_turn shift; its size [0.82, 1.30] is indicative.

## 3. Regimen

| Stage | Cost [D] | Readout and prediction | Decision |
|---|---|---|---|
| **S0** | $0.28 | **(a)** Replay of `own1`'s 2,000 self-play games ($0.04): band count by pair type, runner-up mix, margins, E[R²] with V; no prediction beyond 76.1 and 0.688. **(b)** Exact mirror null, 3,200 games; attack +1 on one seat, 6,400 games, −78 ± 33 Elo. **(c)** 2,000 coin games: I1, I5–I7, I9, I-φ, Var(u·x); overhead as the steady window against (b)'s. | **(a)** Replaces 76.1 and 0.688; the power script reruns. **(b)** A digest mismatch, a first divergence at G ≥ 0.35, or the attack outside its window: fix the match path. **(c)** Price × (1 + overhead) above the host's line: another host. |
| **S1** | $15.91 | **Batch 1:** P-coin, I2, mean label per bin (fixes the regime). **After batch 3:** J3. **After 8:** v̂, c, b; Ĝ_pf sealed. | **Plant or invariant fails:** debug. **J3 < 0, null forecast:** stop (P 0.50 at a null; ≤ 0.041 at an adoptable round in the nonnegative regime; up to 0.15 in the convex one [D, Gaussian]). **J3 < 0, forecast > 0 by 2 SE:** match E3. |
| **S2** | $0.61 | π_1 against `own1`; Ĝ_pf, miss line ≈ 9.9 | **Lower bound > 0:** adopt; go to D2. **Null:** S3-null. |
| **D2** | none | Ceiling 0.414 × max(J1, Ĝ_pf) | **Nonnegative regime, both < 8.6 Elo:** S3 cannot certify; S4. **Otherwise:** S3 (model-bound forecast in the convex regime), then S4. |
| **S3** | $17.13 | As S1–S2 on π_1; π_2 against π_1, one network | Adopt by the rule |
| **S4** | $6.17 | P-distill; π_k distilled; ρ̂ on held-out positions; distilled network against π_k at offsets 0 and −1.5 (12,800 games each, eval price) | As `current_design.md`: the incumbent stays until beaten directly. P-distill's direction fails: distillation broken. |
| **S5** | $20.26 | Round on the distilled base; claim against `own1` two-network | Adopt by the rule; then D2 |
| **S3-null** | none until chosen | The bound left | User's choice: w = 0.7; critic column; one-seat coins; in the convex regime only, twice the batches on the same band (there a null does not exhaust it) |

**Main line.** S0 → S1 → S2 → D2 → (S3) → S4 → S5 → D2 …
- Without S3, $44.46 with the miss reserve; the $40.54 left buys one more reset-and-round cycle ($26.43), so a third adopted round.
- With S3, $61.59. On a 1.41× slower host the line without S3 still fits ($58.62); S0(c) decides the host.
- Ruling (b) "no" costs 1.225× the coin games, about $3.4 a round.

**Toward human level.** The $1,000 case is argued after two adopted rounds, from their measured gains, miss ratios and ρ̂, with DPI back in it. Nothing measured links `own1` to human ratings; nothing rests on it.

## 4. What is new, and the nearest work

Searched 2026-10-20; finding nothing does not prove absence.
1. **The √N ceiling on sign-rule gains** for any advantage distribution with nonnegative mean, used to schedule rounds: the interim's error bound, round 2's certifiability, what a null exhausts.
   - Nearest: the nonconcavity of the value of information, which is the negative-mean case where it fails (Radner & Stiglitz 1984; [Chade & Schlee 2002](https://ideas.repec.org/a/eee/jetheo/v107y2002i2p421-452.html)).
   - Also: plug-in treatment rules ([Manski 2004](https://doi.org/10.1111/j.1468-0262.2004.00530.x); [Kitagawa & Tetenov 2018](https://doi.org/10.3982/ECTA13288); recalled, not re-read), whose regret bounds the distance to the oracle, not the growth from a noisy start.
   - Two searches found no statement of it; being elementary, it may exist under another name.
2. **One round per frozen span:** policy iteration with a representation refresh, scheduled by the ceiling. Nearest: least-squares policy iteration on fixed features ([Lagoudakis & Parr 2003](https://www.jmlr.org/papers/v4/lagoudakis03a.html)), whose iterations stay in one span.
3. **An exact mirror null** from state digests under shared seeds. Nearest: Fishtest's paired openings; exactness needs a deterministic simulator.
4. **Carried:** coin labels in deterministic self-play, a micro-randomized trial at the decode's near-ties ([Boruvka et al. 2018](https://doi.org/10.1080/01621459.2017.1305274); [Qian 2025](https://arxiv.org/html/2502.13500v2)), exact through the simulator's laws ([AIVAT](https://arxiv.org/abs/1612.06915)).

## 5. Weakest point

- **The ceiling's premise may hold on part of the band only.** It needs a nonnegative mean in-span advantage; away from ties the runner-up is likely worse. Where batch 1 reads a negative mean, labels can have increasing returns, and D2, the interim's meaning and the exhaustion rule fall back on the Gaussian model as in round 6. The spread v stays unmeasured until S1.
- **The reset comes earlier.** Distillation closure on a policy target is unmeasured (the one measured closure, 0.58, was on another target), and the end_turn bias may return. If the reset loses most of round 1's gain, the main line stalls at π_1 and the money goes to S3 or widening.
- **The steady price rests on 600-game arms on one host.** Three arms agree, but hosts have differed by 1.41×.

## 6. Rulings for the user

- **(a)** Results regressed on randomized near-tie coins, pooled by one linear head: allowed? If not, branch L (`current_design.md` §4).
- **(b)** Control variates in labels only: the seven luck columns (DPI deferred).
- **(c)** Adoption and claim by one PURE paired match in `run_elo_batch` at 120 jobs (SE 1.79 at 32,000 games); after a reset, the claim is two-network, at the eval price.
- **(d)** The −1.5 offset stays inside π until a distilled network at offset 0 is non-inferior.
- **(e) New: one round per span.** After an adopted round 1 whose ceiling is below the adoption line, distillation (S4) comes before any same-band round. S4 is the plan's only network training, so it falls under the 2026-10-15 ruling: the optimizer discussion must precede round 2.

## Provenance table

| Number | Value | Used for | Tag | Source or derivation |
|---|---|---|---|---|
| Hourly price; budget | $0.42; $85 | everything | given | seed §1 |
| Pool price, steady | $1.912e-5 a game (21,966/h) | every coin and match cost | M | `round7_provenance_price_tail.py` on `arm_ref_p120x6.log`, completions 120–480 |
| Same, two other arms | 21,966/h (priors-only), 21,873/h (160 jobs) | reproducibility | M | same |
| Pool price, whole wall | $2.51e-5 | check | M | `round6_provenance_prices.out` |
| Start-up; tail; shutdown | 16 s; 20 s; about 14 s of a 129-s arm | why the wall overstates | M | same logs |
| Host spread | 1.41× | slow-host budget | M | BACKLOG.md via `round6_provenance_prices.py` |
| Eval price | $1.17e-4 a decisive game | S4, S5 two-network matches | M | `round6_provenance_prices.out` |
| Box setup | $0.018 | S0 | M | same |
| Worker cycle; CPU; cores; server busy | 39.26 ms; 3.12 ms; 7.1 of 31; 0.71 | overhead bound | M | `speed.json`, in `round7_provenance_price_tail.out` |
| Walk | 0.321 s a game; ≤ 1.8% | columns online | M; D | `round6_simplify_walk_cost.out`; 321/465/39.26 |
| Statistics | under 0.1% | logging | D | 152 × 776² multiply-adds a game against 18 s of worker cycles |
| Server feature gather | — | overhead | U | S0(c), steady window against S0(b) |
| Replay rate | 2,440 decisions/s | S0(a) | M | BACKLOG.md via `round5_simplify_critique_extra.py` |
| `own1` self-play games | 2,000 decisive (800 + 800 + 400) | S0(a) | M | `match_luck_20261019/rows.jsonl` |
| Census cost | $0.043 | S0 | D | 2,000 × 449.1/2,440 s × $0.42/h |
| Stage costs | S0 0.28; S1 15.91; S2 0.61; S3 17.13; S4 6.17; S5 20.26; reserve 1.22 | budget | D | `round7_provenance_budget.out` |
| Totals | $61.59 / $44.46; slow host $82.78 / $58.62 | plan | D | same |
| Null round 1 | $6.63 / $16.80 | S1 risk | D | same |
| S5 claim correction | +$3.13 | budget | D | 32,000 × ($1.17e-4 − $1.91e-5) |
| DPI saving | $1.91 or $3.82 a round | deferral | D | `round6_synthesis_power.out` §5 at the steady price |
| DPI reading resolution | paired SE 0.012–0.015 (800 games) | when revived | M | `round6_invent_critique_checks.out` §2 |
| Var(z) | 1.0004 | σ² | M | `round5_simplify_variance.json` |
| Label residual | 0.688 (self-play cells 0.651, 0.58–0.72) | σ² | M | `round6_provenance_cells.out` (rerun identical) |
| Cell alone | 0.843 | ruling (b) "no" | M | `round6_simplify_pairs_raw.out` (rerun identical) |
| σ² | 3.67 at p 0.25 | every size | D | 1.0004 × 0.688/0.1875 |
| Pairs kept | 0.847 ± 0.009 | match SE | M | `round6_simplify_pairs_raw.out` |
| Match SE | 1.79 Elo at 32,000 | adoption | D | 347.4 √(0.847/32,000) |
| Elo per z | 347.4 | all | D | 800/ln 10 |
| Band share; band | 0.339; 76.1 per side-game | volume | M (`parity3`); D | `round5_simplify_band.json`; × 224.6 |
| `own1` decisions per side-game | 224.6; 19.4 side-turns | volume | M | `round5_simplify_critique_decision_units.out` |
| Band by bin | 25.6 / 22.0 / 19.0 / 8.8 per side-game | regime SEs | D | density × 224.6/271.7, `round7_provenance_scaling_bound.out` §4 |
| Mean-label SE | 0.00085–0.00145 per bin; 0.00049 band | regime | D | √(σ²/labels), same §4 |
| η_k | 6.24e-5, 2.34e-5, 1.17e-5 (k 3, 8, 16) | ceilings | D | σ² × 776/(k × 15.2 M) |
| √N ceiling | g(N′) ≤ √(N′/N) g(N), mean ≥ 0 | interim, D2, exhaustion | D | §1.2; 16 shape cases, `round7_provenance_scaling_bound.out` §1 |
| Observed ratios | g8/g3 1.06–1.58 (≤ 1.633); g16/g8 1.00–1.35 (≤ 1.414) | check | D | same §1 |
| Failure at μ < 0 | g16/g8 above 1.414 in 4 of 6 cases, up to 6.33 | convex regime | D | same §2 |
| Interim false stop | ≤ 0.041 at an adoptable round | S1 | D | Φ(−√(3/8) × 5.1/1.79) |
| Same-band increment | ≤ 0.414 g8; Gaussian 0.19–0.36 | D2 | D | same §3 |
| D2 threshold | 8.6 Elo (12.3 at 80% power) | D2 | D | 3.6/0.414; 5.1/0.414 |
| Interim P(stop) | 0.50 at a null; up to 0.15 convex | S1 | D (Gaussian for the latter) | `round6_synthesis_power.out` §3 |
| Floors | sd 0.0016 (μ 0); 0.0034–0.0036 (μ −0.0018) at d 776 | null meaning, convex regime | D (Gaussian) | same §2–3 |
| μ scenario | −0.0018 (−0.0061, +0.0017) | illustration | M | `round5_provenance_stop_acts.out` |
| Forecast SE; miss line | 4.6; 9.9 Elo (from logs at run time) | S2 | D | `round6_synthesis_power.out` §4 |
| False adoption; adoptable gain | 0.023; 5.1 Elo | adoption; ceilings | D | one-sided 2 SE; 2.84 × 1.79 |
| Attack known answer | −78 ± 16; window ±33 Elo | S0(b) | M via D | `round6_provenance_critique_attack_stage1.out` (rerun identical); 2√(16² + 4.0²) |
| kv23 direct | −0.240 ± 0.097 z | known-answer check | M | `round5_provenance_knob_switches.out`; √((1 − 0.24²)/100) |
| Flips by gap bin | 0.53 / 0.12 / 0.012 (≤ 0.042) / none at ≥ 0.35; 1.76% of decisions | mirror null | M (`parity3`, cross-box) | `round5_simplify_critique_extra.out` §1; `round5_provenance_numerics_ties.out` |
| Identical prefix | about 57 decisions | null coverage | D | 1/0.0176 |
| Information per game-equivalent | 39.0 coins; 0.82 fork pairs | why coins | D | `round5_simplify_power.py` |
| Caps, `own1` self-play | 0 of 1,600 | caps dropped | M | `round6_simplify_walk_cost.out` |
| Tempo; agreement closure | 1.00–1.06 (0.82–1.30); 0.58 | P-distill, indicative | M | `round6_provenance_closure.out`; prereg 2026-10-11 |
| Distillation; P-distill | $2.1; $1.07 | S4 | M; D | prereg; `round7_provenance_budget.py` |
| w, p, batch, batches, match, interim, d | 0.35, 0.25, 100,000, 8, 32,000, after 3, 776 | design | C | consequences derived in round 6 and here |
| Spread v; c, b by type; switched share q | — | the gain; regime | U | S1: batch 1 (means), progressive fits (v); nothing earlier relies on them |
| Re-pricing under π_1 | — | S3's value beyond the ceiling | U | scale read from S2's miss |
| Second-order loss | — | a miss | U | S2; E's match |
| ρ̂, spillover, end_turn bias after distillation | — | S4 | U | S4's held-out reading before its match |
| Two-network pool price | — | S5 claim | U | bounded by the eval median until timed |
| κ_T; compounding; exploitability; human level | — | S3-null option; $1,000 case | U | nothing before them relies on them |
