# Design constants — derivations and rationale

This document collects numerical constants used in our model,
training, and search code that AREN'T arbitrary tuning knobs but
are DERIVED from a specific assumption or measurement. It exists
because re-deriving a constant from a docstring fragment costs
real time, and because constants get copy-pasted across files
faster than their justifications do.

## How to use this document

**When a constant is derived (from math, from a measurement, or
from a fixed external standard), add an entry here.** Required:

- The constant's name + value (e.g. `cliffness_max = 0.577`)
- Where it's defined in code (file:line or symbol name)
- The derivation: math, measurement protocol, or external source
- A "why this number specifically" note — what it would mean if it
  were different, what bounds it on each side

**When you find yourself writing "where does this value come from?"
in a code comment, the rationale belongs HERE, not in the
comment.** A two-line code comment + cross-reference here is fine
and preferred.

**Tuning knobs that are arbitrary defaults DON'T belong here.**
This doc is for derived / measured / canonical constants only.
Things like learning rate, c_puct, dirichlet_alpha live in
`constants.py` with their own comment block; if they're
defended by experiment, the experiment goes in BACKLOG.md.

---

## Table of contents

- [Value head / cliffness](#value-head--cliffness)
- [Encoder normalizations](#encoder-normalizations)

---

## Gumbel target

### `gumbel_rescale_floor = 0.04` (one C51 atom)

**Defined:** `tools/mcts.py` (`MCTSConfig.gumbel_rescale_floor`),
CLI `--mcts-gumbel-rescale-floor`, worker `--gumbel-rescale-floor`.

**Derivation:** the C51 value head quantizes [-1, +1] into 51 atoms,
so its resolution is `2 / (51 - 1) = 0.04` — one atom. A root whose
completed-Q spread is below one atom is indistinguishable from value
noise. The sigma rescale divides the Q vector by
`max(spread, floor)`, so with the floor at one atom, sub-resolution
spreads scale the injected target perturbation down proportionally
(smooth fade to the prior), while any spread ≥ one atom is passed
through unchanged.

**Why it exists (2026-08-12 diagnosis, "the self-play loop is
distilling its own noise"):** the legacy floor of `1e-8` made the
min-max rescale fully scale-invariant, so a pure-noise root received
the same `(c_visit + max_N) · c_scale ≈ 5.2`-logit target
perturbation as a decisive one — measured KL(target‖prior)
independent of value-noise level. Iterated, that is a sharpening
random walk on the policy; it also explains the measured
raise-`--mcts-sims`-doesn't-help result (more sims raise the sigma
gain, not the signal). Regression tests:
`test_gumbel_rescale_floor_fades_noise_targets_to_prior`,
`test_rescale_floor_caps_rank_noise_amplification`. Live instrument:
the `distill_kl_prior` telemetry column.

---

## Value head / cliffness

### `cliffness_max = 0.577` (≈ 1/√3) — HISTORICAL since 2026-08-10

**Status:** the two in-search consumers (`cliffness_bootstrap_alpha`
Bayesian backup shrinkage and `adaptive_sim_budget`, with
`cliffness_max` as its normalizer and `_BOOTSTRAP_PRIOR_VAR = 1/3`)
were DELETED 2026-08-10 (technique review, user ruling X2 — both
uncalibrated and never measured; `git log -S cliffness_bootstrap_alpha`
recovers the code). `output.cliffness` itself and the root-cliffness
debug log remain. The derivations below are preserved for any future
epistemic-uncertainty revival (BACKLOG item 8).

**Derivation:** `cliffness = std(Z(s))` is the standard deviation
of the network's predicted categorical value distribution over
atoms in [V_MIN, V_MAX] = [-1, +1]. The MAXIMUM possible std for
ANY distribution supported on [-1, +1] is achieved by the
two-point distribution placing mass 0.5 on each endpoint, which
has std = 1.0. The maximum std for a UNIFORM-ish distribution
(maximum entropy under a fixed support) is the std of the
continuous uniform on [-1, +1]:

```
σ_uniform = sqrt((V_MAX - V_MIN)² / 12) = sqrt(4/12) = 1/√3 ≈ 0.5774
```

The discrete uniform on K=51 atoms over [-1, +1] gets to within
3 decimal places of this; verified in
`test_distributional_value.test_cliffness_high_when_distribution_spread`.

**Why this number specifically:** 0.577 is the practical "I have
no idea what's going to happen" upper bound — corresponding to
the network outputting uniform-over-atoms logits, which is the
max-entropy state of a freshly-initialized C51 head. Cliffness
above 0.577 means the network's distribution is BIMODAL or
otherwise more spread than uniform — possible in principle but
unusual to see during training. The deleted consumers used 0.577
as the normalizer for the adaptive sim budget
(cliffness/cliffness_max in [0, 1]) and 1/3 as the prior variance
in the Bayesian bootstrap shrinkage
(`scale = σ²_prior / (σ²_prior + α·cliffness²)`, uniform prior on
[-1, +1] matching a fresh C51 head's max-entropy output; at
cliffness² ≈ 1/3 that gives a 50/50 blend with the prior).

**Cross-references:**

- `model.py` `VALUE_V_MIN, VALUE_V_MAX, VALUE_N_ATOMS`: define
  the support that 1/√3 is computed against. If the support
  changes, this number changes (proportional to support range).

---

## Encoder normalizations

### `HP_NORM, MOVES_NORM, EXP_NORM, ...`

**Defined:** `constants.py`, re-exported from `encoder.py`.

**Derivation:** these are ROUGH normalizers ("typical max"
values), not derived constants — chosen to keep encoded inputs
in [0, ~1] for stable training. They live in `constants.py`
with their own comment block explaining "scale rationale".
Listed here only for completeness; if you wonder where these
come from, check the `constants.py` block — they're era-mod
overridable in one place.


## Material-margin village normalization (2026-07-12)

`DrawTiebreakConfig.weight_village = 2.0` (recalibrated 2026-07-21;
the original derivation below used 10.0 — configs/draw_tiebreak.json
is authoritative), applied as
`w * (Δvillages / MAP_TOTAL_VILLAGES)` inside the shared material
score (`tools/draw_tiebreak.py`), with `score_scale = 5.0` unchanged.

Derivation: village counts vary ~10-30 per ladder map, so the old
1.0/village term meant "one village" was worth 3x more signal on a
small map than a big one, and a full sweep saturated differently
everywhere. Normalizing to the map fraction makes the semantics
map-invariant; the 10x multiplier calibrates magnitudes:

  - half the map's villages ->  10 * 0.5 / 5 = 1.0 -> tanh = 0.76
    (a dominant position reads as near-saturated margin);
  - one village on a 20-village map -> 10 * 0.05 / 5 = 0.1
    (same order as the old per-village 0.20, now map-invariant);
  - with `aux_value_bonus = 0.3`, one village moves a search leaf by
    ~0.03 -- ~10 villages differential reaches outcome order (user
    calibration target, 2026-07-12).

Gold / unit-value weights (0.05) are NOT normalized: gold scales are
already map-independent (start ~100, village income fixed).

## Spool-worker VRAM budget (2026-07-18, revised 2026-07-20)

`tools/sim_self_play.py`:
`SPOOL_WORKER_VRAM_BYTES = 640 MiB`, `TRAINER_VRAM_RESERVE_BYTES = 15 GiB`,
consumed by `_assign_spool_devices` ("auto" mode:
`K_cuda = (total_vram − reserve) // per_worker`).

Derivation — measured on the 45230879 campaign box (RTX 4090,
23.52 GiB usable, 5.0M-param model):

  - per-worker VRAM: torch's OOM reports listed worker processes at
    388–586 MiB on 2026-07-18 and 564–618 MiB on 2026-07-20 (CUDA
    context ~300 MiB + model weights + forward buffers; the spread
    is batch-in-flight variance). 640 MiB is the latest observed
    ceiling rounded up to a clean budget unit.
  - trainer reserve: the learner's backward peak GROWS with play
    quality — 7.14 GiB on 2026-07-18, 12.6 GiB on 2026-07-20
    (12.05 GiB in use + a failed 556 MiB allocation) at UNCHANGED
    `--train-batch-size 64` / 2048-transition minibatches. Longer,
    denser games mean bigger per-batch activation graphs. 15 GiB ≈
    1.2× the latest peak; the multiplier is deliberately modest
    because the peak history (not a one-off measurement) is the
    real guide — REVISIT if a future OOM shows the peak passing
    ~14 GiB.

Consequence on a 24 GiB card: K ≈ 13 cuda workers; requesting more
workers spills the remainder to cpu instead of starving the trainer.
Incident history: 2026-07-18 crash-loop (3 OOM deaths, 56 all-cuda
workers, no budget); 2026-07-20 OOM (auto-assign granted 19 cuda
workers under the stale 12 GiB reserve < 12.6 GiB actual peak).
`--spool-cuda-workers` / `SPOOL_CUDA_WORKERS` overrides the
constant-based K with a measured cap (the campaign box pins 8 via
env.sh). Re-measure both numbers if the model grows past ~10M
params or the replay minibatch changes materially.

`DEMOTION_HEADROOM_BYTES = 2 GiB` — the reactive-demotion margin
(2026-07-20): each iteration the learner recomputes
`headroom = total − trainer_peak − n_cuda × per_worker` from the
iteration's measured backward peak and gracefully demotes one cuda
worker (between-games ctl-file exit, zero data loss) when headroom
drops under the margin. Derivation: the trainer peak history shows
≤ ~500 MB growth per iteration (7.1 → 12.6 GiB over ~19
iterations, front-loaded); 2 GiB ≈ 4× the largest observed
single-iteration step, so the guard fires at least one iteration
before exhaustion even on the fastest observed trend. The margin
makes the spawn-time constants above non-load-bearing: they seed
the initial split, and the ratchet converges the fleet on any
card/model combination. A residual OOM (a single-step jump past
2 GiB) is caught in `run_iteration`'s train_step retry: empty
cache, HARD-demote one worker (process kill frees its ~300 MB CUDA
context; costs that worker's one in-flight game), retry once.

## Gumbel q-transform: `c_visit = 50`, `c_scale = 0.1`, rescale to [0,1]

`tools/mcts.py` — `_gumbel_sigma` / `_rescale_q`, consumed by BOTH
sequential-halving selection and `extract_gumbel_policy_target`.

    sigma(q) = (c_visit + max_b N(b)) * c_scale * rescale(q)
    rescale(q) = (q - min q) / max(max q - min q, 1e-8)     -> [0, 1]

**Where the numbers come from.** They are the reference implementation's
defaults, not free knobs: DeepMind's `mctx`
(`_src/qtransforms.py::qtransform_completed_by_mix_value`) ships
`value_scale=0.1`, `maxvisit_init=50.0`, `rescale_values=True`, and
`_rescale_qvalues` min-maxes the completed-Q vector into [0,1]. Verified
against the source 2026-07-28.

**Why the rescale is load-bearing (not cosmetic).**

1. *Bounded sharpening.* The logit spread sigma can contribute is exactly
   `c_scale * (c_visit + max_N)` — about **8.2** at 32 sims — regardless of
   whether this node's raw Q values span 0.02 or 2.0. Without it, sharpening
   is at the mercy of the value head's current scale.
2. *Offset invariance.* Adding a constant to every Q leaves the target
   unchanged. This matters concretely: a side-to-move bias in the value head
   was measured drifting 0.06 -> 0.37 over ~1M decision steps, which without
   the rescale becomes a ~30-logit shove on every cross-turn comparison.

**The bug this replaced (2026-07-28).** We ran the PAPER's `c_scale = 1.0`
on RAW q in [-1, 1] with no rescale — the paper's constant without the
paper's normalization. That multiplies Q differences by `(50 + max_N)`
≈ 50-82, saturating the softmax: the distillation target became a near
one-hot label, so every iteration distilled `argmax` rather than a policy
improvement. Measured on the live campaign: an action whose PRIOR was ~0.16
received target mass **0.000-0.002 across four independent searches** of the
same state, while `end_turn` was inflated from a ~0.10 prior to 0.43-1.00
target. That is a systematic teaching signal, and it tracked the observed
collapse in recruiting (prior p90 mass 0.48 -> 0.30 over the regressing leg).

Guarded by `tests/test_gumbel_qtransform.py` (reference defaults, unit
interval, softer-than-old, offset invariance, monotonicity). `MCTSConfig.
gumbel_rescale_q` exists only as an A/B escape hatch.

**Two implementation notes.**

*Completed-Q, not raw Q.* Both call sites feed `_completed_q(root, edges)`
into sigma: visited edges keep their own q, unvisited ones take `v_mix`.
Using raw `edge.q_value` would inject **0.0** for every unvisited edge, and
on a node whose visited Q are all negative those zeros anchor the rescale
window's upper bound — distorting the transform state-dependently. Caught
in review of 4fecbca; guarded by
`test_completed_q_uses_v_mix_not_zero_for_unvisited`.

*The escape hatch is not "the old behaviour".* `gumbel_rescale_q=False`
alone, at the new `c_scale=0.1`, is a THIRD regime (soft but unrescaled).
To A/B against the pre-2026-07-28 setting you must set BOTH
`gumbel_rescale_q=False` AND `gumbel_c_scale=1.0`.

## TCS linear-link advantage gain: `target_beta = 5.0` (2026-08-17)

`tools/turn_search.py::tcs_target_distribution(link="linear")` builds
the TCS distill target as
`prior^lam * max(0, 1 + beta*(q - LOO_mean))`, where `LOO_mean` is
the mean boundary value of the OTHER evaluated actions at the same
coordinate. The link is linear in q by USER RULING (2026-08-17):
"random draw among the evaluated actions should not push their
probability up" — under an uninformative value head, evaluation
EXPOSURE must carry no expected mass gain, which the exp/sigma link
violates through Jensen's inequality (the leg-3 `end_turn` exposure
ratchet: +0.068 expected target mass per coordinate for an
always-evaluated action under pure noise vs +0.002 for an
equal-prior decoy; see docs/leg3_passivity_rootcause_20260817.md R2
and tests/test_turn_target_link.py's decoy invariant).

**Why beta = 5.** The natural unit for value differences is the C51
atom, width `2/(51-1) = 0.04` (docs/design_constants.md, C51
section). Anchors:

- The rung-1 probe's **median accepted improvement was 0.070 ≈ 2
  atoms** (docs/tcs_spec.md validation). At beta=5 a 2-atom
  advantage earns a multiplicative factor 1 + 5*0.08 = **1.4** —
  a firm but not saturating push, comparable in spirit to the
  Gumbel path's post-fix "soft improvement, not argmax" regime.
- The zero-clip lands at advantage `-1/beta = -0.2` = **5 atoms
  below the evaluated peers' mean**: an action has to be
  decisively outclassed before its target mass zeroes. Clip
  frequency is telemetered (`link_clip_frac` rides the distill
  stats); a high clip rate in production means beta is too hot.
- Sensitivity is linear (it is the point of the link): halving
  beta halves every push. No cliff, no saturation, so the knob is
  safe to tune from telemetry.

`beta` scales down with `distill_target_temp` (temperature applies
to the advantage, not as an exponent, to preserve the linearity
that makes the link exposure-invariant).

## CERT_RESERVE_PAD = 2.0 (tools/plan_tournament.py)

Safety factor on the plan-tournament certification reserve's
half-turn cost estimate. The reserve guards the invariant "an
optimistic cost estimate shrinks selection, never certification"
(review round 6). 2.0 is PICKED, not derived: no half-turn length
distribution has been measured yet (review round 8, C4) -- the
telemetry columns pt_half_est and pt_cert_starved_rate from the
step-1 match are the measurement that will replace it. Scope
(round-21 C3): the pad only discounts while PAD x est <
project_max_actions, i.e. est < 10 at the defaults -- at the
documented operating point (est ~12-17) the reserve sits at the
hard bound and the pad is inert, so its calibration loop applies
to the short-half-turn regime only. Priced at the hard bound
(project_max_actions) instead, the reserve over-charged ~250
forwards and pushed fundable long side-turns into value-only
abstention (review round 7, C0).

## _T_CRIT = {2: 3.37, 3: 2.0, 4: 1.72, 5: 1.61} (tools/plan_tournament.py)

n-aware critical factors for certification acceptance
(mean > crit * sd / sqrt(n)). The legacy rule (factor 2 at n=3,
from turn_search.two_stage_accept) has band-free false-accept rate
P(t_2 > 2) = 0.0918; these are the t_{n-1} quantiles at 1 - 0.0918,
holding that alpha for every replicate count a mid-stage budget
death can leave (n=2 under the raw factor-2 rule inflates alpha to
0.148 -- review round 8, C0, verified by simulation). n=2:
tan(pi*(0.5-0.0918)) = 3.37; n=4/5 from the t-quantile tables.
Coverage rule (round-9): config_from_args clamps cert_redraws to
the tabled range, so every reachable replicate count has an
alpha-holding factor; extend the table before raising the knob
ceiling.
