# GBC — Goal-Basis Completion (reachability head): approved spec (2026-08-13)

**Status: concept APPROVED by user 2026-08-13. Authorization scope:
rung 0 only (offline measurements, laptop, no model code).** Each
later rung re-proposes on its predecessor's numbers.

Provenance: Opus-workflow research report + two adversarial reviews
(session 8b044cb2, workflow `wf_884c8c53-16d`, journal results #0,
#2, #7), amended by the 2026-08-13 user discussion. Line numbers
verified by the reviews against code as of 2026-08-12.

---

## 1. Claim

Two verified facts compose into the design:

1. **The improvement operator has a named hole.** At `gumbel_m=16`
   vs 100–200 legal actions, every simulation goes through
   candidate root edges only; `_completed_q` assigns ALL unvisited
   edges the single scalar `v_mix`; `extract_gumbel_policy_target`
   consumes that vector over all legal actions. **~85–90% of the
   target's support is ranked by prior alone, forever** (verified:
   `mcts.py:1248-1271`, `:1631-1712`; no top-k cap on edges). This
   is the precise mechanism behind "coverage pinned at m=16 at
   every budget" and the prior ratchet.
2. **Label density** (the CORRECTED rationale — the report's
   C51-quantization argument is WRONG and must not be recorded as
   project belief: the C51 mean is continuous; atoms bound support,
   not output precision; one atom = 0.04 is a noise *proxy*, per
   `docs/design_constants.md`). The honest argument: z is one bit
   per game; `dies(u)` within k turns is a hindsight-observable
   event occurring dozens of times per game. A head predicting a
   frequent event reaches a given standard error with orders of
   magnitude fewer trajectories than one predicting the terminal
   outcome.

GBC: train small heads predicting per-entity event probabilities
from hindsight labels, and use their **per-action goal advantage**
to replace the constant in the slots of fact 1. It is primarily a
**policy-target enrichment**, not a search enhancement: the target
becomes "16 searched actions + ~134 priced by a 2-turn reachability
model" at zero additional simulation cost.

**Honest scope (review #7):** after the 2026-08-12 rescale-floor
fix, low-spread roots already fade to the prior by design. With
visited-only rescaling (§4), GBC's addressable surface is "the
unvisited actions at roots where search already found real
spread." Pre-register that smaller claim, not "raises the horizon
at every root."

**Known forfeit:** Danihelka et al.'s policy-improvement proof
assumes the completion is `v_mix`. Adding a learned term converts
a proven operator into an unproven one resting on head B's
off-policy accuracy — which is why rung 2 is load-bearing, not
optional.

## 2. Vocabulary and labels

Goals = (predicate, entity) over tokens the trunk already emits:
`dies(u)` (BOTH sides' visible units — opportunity and threat in
one class, sign learned never asserted), `flips(v)` (villages),
`levels(u)`. |G| ≈ 30–75 per state. Horizon k ∈ {1,2,3} **game**
turns (not side-turns, not a discount).

`unit_reaches(u,h)` is dropped on two legitimate grounds ONLY:
one-turn reach is exact via `pathfind_sim.unit_reach`, and
multi-turn reach is a U×H head. ("dies/flips is where the win
probability lives" is an unverified strategic assertion — recorded
as hypothesis Q8, measured by rung 0b, never a premise.)

**Fog: achievement = CONFIRMED achievement.**
C(s,g,k) = P(the observer sees g become true within k turns | s).
God-view labels would train the head to predict events the
observer cannot condition on — irreducible noise concentrated
exactly where fog matters (CLAUDE.md §6 contract).
- **A1 (review): the observer is the side-to-move at s, for every
  goal, regardless of entity ownership** (as written, enemy-death
  labels would be censored by the ENEMY's visibility — a silent
  inversion). Unit test with a synthetic fogged kill.
- Implement against **`wesnoth_ai/visibility.py`**
  (`units_visible_to`, `visible_hexes_for`) — NOT `tools/fog.py`,
  which is orphaned legacy code with zero importers (fog-semantics
  fork hazard).

Labels: hindsight forward scan over stored trajectories (the same
backward walk `MCTSPolicy.finalize_game` already does for
`aux_target`). **Keyed by `unit.id` / hex (x,y), NEVER slot
indices** — unit slots sort by (y,x,id) and shift as units move
(the `fa95da5` stored-index failure class). Stratified sampling:
drop already-true goals; hold each (predicate,k) bucket ≥5%
positives.

## 3. Heads (MVP = A + B only)

All read contextualized tokens from the one existing trunk
forward. Goal embedding z_g = E_pred[predicate] + entity's
contextualized token. Built behind a flag, default OFF, sticky in
the checkpoint (the `aux_score`/`moves_left` peek-and-OR pattern).

- **(A) State achievement:** C(s,g,k) = sigmoid(MLP([z_g;
  z_global])[k]); masked BCE vs hindsight labels.
- **(B) Action-conditioned:** a zero-init bilinear logit
  correction in the same pointer-attention form as
  `target_logits`: Δ(s,a,g,k) = ⟨U_a z_actor + U_t z_target,
  U_g z_g⟩/√d + b_k; C(s,a,g,k) = sigmoid(logit C(s,g,k) + Δ).
  One matmul prices EVERY legal action × every goal ≈ 0.1–1% of a
  forward — what makes the head consumable inside search at all.
  Labels exist only for taken actions; untaken-action estimates
  are generalization — the entire bet, measured head-on at rung 2.
- **(C) Worth — REPLACED by amendment A3:** do NOT learn w online.
  Rung 0b already fits a linear map from goal-event indicators to
  turn-scale ΔV; **freeze that fitted vector as a config table**
  (predicate × unit-type; config-first per CLAUDE.md). Removes the
  report's own stated reservation (the one place a learned scalar
  acquires authority to say "this event is good"), removes the
  w↔V training circle, makes grounding auditable. A learned head C
  returns only if the frozen table demonstrably underperforms.

Goal advantage: GA(s,a) = Σ_g w(g)·[C(s,a,g,k*) − C(s,g,k*)],
k* = 2.

## 4. Consumption (MVP = Seam 2 only)

- **Seam 2 — the distillation completion:** unvisited actions
  complete at `v_mix + β_target·GA(s,a)` instead of raw `v_mix`.
  Lands in both `_score` and the target automatically.
  **Amendment (review #7): add an explicit `include_gbc` argument
  to `_completed_q`** — because it feeds `_score`, a naive
  "β_search=0" arm still changes which candidate sequential
  halving plays; the target-only A/B arm must be actually
  target-only.
- **A2 — rescale containment (replaces the report's ±κ·W_vis
  clip, which still dilutes):** `_rescale_q` computes lo/hi from
  **visited edges only**; completed values clamp into [lo,hi].
  Visited-edge σ gaps stay byte-identical to control; GBC purely
  reorders unvisited actions within the existing window. Removes
  the dilution failure structurally (the 2026-08-12 floor fix's
  evidence channel is untouched). Add `sigma_span_visited`
  telemetry next to `kl_prior`. β=0 must verify byte-identical on
  200 logged roots.
- **Seam 1 (FPU at interior nodes) — DEFERRED.** Root uses
  sequential halving, so FPU is interior-only; it requires the
  goal head in `_forward_impl_batch` (the second forward path) and
  a fused D2H transfer (per-leaf readback lands in the documented
  `.item()` stall). Not in the MVP.
- Secondary consumers (post-MVP, each with placebo): search-budget
  allocation via `n_sims_override` at high-tension roots;
  minibatch oversampling of high-tension states; readability
  diagnostics ("knight at 70% death risk, chiefly from the mage" —
  masked-ΔC proposes, sim-counterfactual disposes).
- **REFUSED consumer (standing prohibition, user-ratified):**
  adding any opportunity/threat gap D to Q — "prefer states where
  I could do better" is a disguised aggression prior.
- Expectile minimax heads C↑/C↓ (τ flips with side-to-move;
  turn-boundary bootstrap keeps k≤3 to ≤3 backups): **rung 5,
  deferred**, gated on rung 4 shipping + directed-episode
  verification of the gap.

Implementation notes: `MCTSEdge` uses `__slots__` — caching
per-edge `goal_adv` means editing the slots tuple (loud, trivial).
`material_end` does not exist in games.jsonl — the berserker
tripwire must use `units_end` + `draw_tiebreak.material_margin` or
add the field.

## 5. Failure modes → observables

| Failure | Observable | Rule |
|---|---|---|
| Decomposition false (ΔV not ~linear in events) | rung 0b held-out R² | kill w/completion below 0.15; proceed ≥ 0.25 |
| Rescale dilution | `sigma_span_visited`; `kl_prior` moving while visited σ gap shrinks | A2 removes structurally; telemetry confirms |
| Berserker collapse (w overweights kills) | `attacks_attempted`/turn +50% with decisive rate flat; losing-side material differential vs control | standing decisive/stall tripwires unchanged |
| Ratchet (head fits its own footprint) | mean \|GA\| rising 30 iters while held-out ECE degrades | tripwire; λ damping + bounded σ already structural |
| Label degeneracy (calibrated but action-flat) | median Var_a[GA] ≥ 3× shuffled-goal placebo (rung 1d) | must not advance if flat |
| Fog leak | ECE gap >0.05 between continuously-visible vs intermittently-fogged strata | censoring is wrong, not fog hard |
| Off-policy extrapolation of head B | rung 2 predicted-vs-realized on UNTAKEN actions | the load-bearing number of the ladder |

## 6. Validation ladder

**Distribution rule (review #7): rungs 1d and 2 must share ONE
policy.** Use `imit_tierb_start.pt` as trunk AND rollout policy;
generate its own trajectories (raw-policy rollouts, ~1.5 h for 300
games on the laptop — MCTS rollouts are ~30× and infeasible; the
raw-policy/search mismatch is a named limitation). Human games =
head-A pre-training only, never the C_π substrate. Budget ~1,000
regenerated self-play games for 1d/2 (300 is under-sized).

**Rung 0 (laptop, half a day, no model code — AUTHORIZED):**
- 0a label yield: ≥5% positives per (predicate,k,fog-stratum)
  bucket after stratification, else prune vocabulary.
- 0b decomposition R² of turn-ΔV ~ event indicators (~2,000
  imitation games, held out): proceed ≥0.25; abandon w below 0.15.
  Also the test of dropping `unit_reaches` (Q8).
- 0c premise test, corrected form: on 200 logged decision points,
  sim-step every legal action, compare Var_a[ΔV] against the value
  head's own noise scale (C51 spread / `cliffness`), NOT against
  atom width; and against exact Var_a[ΔP(kill≤1 turn)] from the
  combat DP (≥5× in normalized terms to proceed). If ΔV is
  comfortably above V's noise, the premise is false — abandon.

**Rung 1 (few GPU-hours, ~$2, trunk FROZEN, heads A+B, dies/flips
only, k∈{1,2}):** 1a ≥ +0.05 AUC vs logistic baseline on {HP,
adjacent enemies, terrain defense, nearest-enemy distance}, per
predicate; 1b ECE ≤ 0.05 per bucket AND fog-stratum gap ≤ 0.05;
1c exact-oracle gate for dies@k=1 vs the combat DP under the
executed plan — Brier ≤ 0.12, slope [0.8,1.2] (DP admissible as
validation instrument — sim-as-judge, never reachable from a play
path; ruling ratified); 1d Var_a[GA] ≥ 3× shuffled-goal placebo
**on regenerated self-play states**. All four to advance.

**Rung 2 (cheap CPU box, ~1 day):** 150 held-out states × sampled
goals; head predicts C for 5 UNTAKEN actions; fork sim, N=100
raw-policy rollouts, one rollout scores all goals simultaneously
(else 10× cost). Proceed iff Spearman ρ ≥ 0.4 (p<0.01) AND
shuffled-goal placebo ρ within ±0.1 of 0. By-product:
counterfactual-labelled extra training data for head B.

**Rung 3 (~$5):** 150 logged roots × {baseline, GBC, placebo} at
32 sims vs a gold standard that is **NOT m=|A|** (at m≈150 the
halving schedule gives ~1 sim/action — a one-ply V sweep, the
exact quantity 0c shows is sub-resolution; self-undermining).
Gold = sim-exact rollout outcome frequency (rung-2 machinery at
larger N) or moderate-m deep search (m=48, 4096 sims). **A4:
oracle self-agreement pre-check first** — gold-vs-gold top-1
agreement on two seeds ≥ ~70% or the ±5-point gate is
undecidable. Proceed: (GBC−baseline) ≥ +5 top-1 agreement points,
|baseline−placebo| ≤ 2, `sigma_span_visited` not shrinking.

**Rung 4 (honest price: ~$30–50, three boxes × ~30 h, 60-iteration
window so the floor-relative stall tripwire can fire):** control
(β=0) / GBC-target-only (via `include_gbc`) / placebo (shuffled
goals). Primary metric: floor-relative fresh_value_ce; secondary:
human-holdout probe CSV, `distill_kl_prior`. SHIP if GBC beats
control ≥0.02 over 40 iters AND placebo ≈ control (±0.01).
**KILL if placebo ≈ GBC — the advice-channel outcome, called
immediately, not explained away.**

## 7. Relation to TCS

Complementary: TCS is the improvement operator (makes turns
better, no notion of what matters); GBC is the perception
primitive (knows what's at stake, proposes nothing). Natural
composition if both gate through: GA/tension directs TCS's
perturbation budget toward contested coordinates. Neither depends
on the other. Both lean on the value head only in its measured
good regime (turn-scale), relieve it at micro-action scale, and
feed it (boundary value targets; trunk shaping + tension
oversampling) with terminal outcome as the arbiter.
