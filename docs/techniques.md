# Techniques inventory — everything beyond bare REINFORCE

This document is a **tree-structured catalog of every learning
technique the training system actually implements**, for contributors
who need to know what is in the box before changing it.

## How to read this document

- **Every entry is verified against the code**, with a `file:line`
  citation. Nothing aspirational is listed as if it worked; features
  that exist but are inert are marked as such.
- **`[ON]` / `[OFF]` is the DEFAULT**, i.e. what you get with no flags.
- **There are TWO default layers and they disagree on purpose.** The
  library dataclasses (`MCTSConfig`, `TrainerConfig`, `ReplayConfig`)
  carry conservative "legacy" defaults so tests and eval paths are
  byte-stable; the **training entry point**
  (`tools/sim_self_play.py`) overrides several of them. Where they
  differ, both are stated, e.g. *"`[OFF]` in `MCTSConfig`, `[ON]` at
  the training CLI"*. Reading only one layer will mislead you.
- Line numbers drift. Function/dataclass-field names are the stable
  handle; treat line numbers as a starting offset.
- Related reading: `docs/design_constants.md` (derivations of the
  numeric constants), `docs/wesnoth_rules.md` (engine-fidelity
  rules), `docs/autonomous_run.md` (what was measured and what was
  refuted), `BACKLOG.md` (open questions).

**Two framing facts that explain most of the design:**

1. **The production path is MCTS + distillation, not REINFORCE.**
   REINFORCE (`Trainer.step`) still exists and is the CLI default,
   but the campaigns run `--mcts`. Under `--mcts` the per-step
   shaping reward path is **structurally inert**:
   `MCTSPolicy.uses_step_rewards = False`
   (`tools/mcts_policy.py:119`) makes the rollout loop skip
   `compute_delta` entirely (`tools/sim_self_play.py:345`, `:482`),
   and `MCTSPolicy.observe` is a documented no-op
   (`tools/mcts_policy.py:493-500`). Anything in §6 (reward shaping)
   only affects the REINFORCE path.
2. **Self-play is strictly current-weights vs current-weights.**
   There is no opponent pool, no league, no frozen-snapshot
   opponent in the training loop (see §7.6).

---

## 1. Search — MCTS

Entry points: `tools/mcts.py` (algorithm), `tools/mcts_policy.py`
(policy adapter + experience bookkeeping). Config dataclass:
`MCTSConfig`, `tools/mcts.py:186-510`.

### 1.1 Core algorithm

- **AlphaZero-style MCTS, no rollouts.** Leaf value comes from the
  network's value head, never from a Monte-Carlo playout.
  *Why:* a 30-action playout would cost 30× the forwards or 30× the
  sim steps per simulation; trades accuracy for tree depth.
  `tools/mcts.py:15-20`, `_expand` at `tools/mcts.py:706`.
- **Factored action space as tree edges.** Each edge is a concrete
  `(actor, type, target, weapon)` tuple produced by
  `enumerate_legal_actions_with_priors`, wrapped in `MCTSEdge`.
  *Why:* avoids materializing the `A×T×H×4` joint.
  `MCTSEdge.__init__`, `tools/mcts.py:535`.
- **PUCT selection at interior nodes.**
  `U = Q + c_puct · P · sqrt(ΣN) / (1+N)`. `c_puct = 1.5` **[ON]**.
  *Why 1.5:* legal-action counts swing from ~3 (early) to 200+
  (mid-game), so a slightly aggressive constant stops the search
  latching onto the model's first guess.
  `_puct_select`, `tools/mcts.py:768`; constant at `:230`.
- **Sign convention: value is always from the node's own side.**
  Backup flips sign whenever the parent's side differs from the
  leaf's. `_backup`, `tools/mcts.py:1146` (`:1208`).
- **Selection-depth hard cap `_MAX_SELECT_DEPTH = 4096`** **[ON]**.
  Defense-in-depth: exceeding it means an undiscovered cycle, so the
  search treats the current node as the leaf rather than hanging.
  `tools/mcts.py:163`.

### 1.2 Root policy — Gumbel AlphaZero (the default root)

Danihelka et al. 2022, "Policy improvement by planning with Gumbel".

- **Gumbel-Top-k candidate sampling + sequential halving.**
  `[ON]` — `gumbel_root = True` (`tools/mcts.py:336`), and at the CLI
  `gumbel_root = not --mcts-classic-root`
  (`tools/sim_self_play.py:3402`, flag at `:2842`).
  Samples `gumbel_m` distinct candidates ∝ prior via `g + logits`,
  then splits the sim budget tournament-style, halving the field each
  phase. *Why:* provably a one-step policy improvement even at tiny
  budgets — which is the regime we run in.
  `_gumbel_root_search`, `tools/mcts.py:1600`.
- **`gumbel_m = 16` candidates** **[ON]**. `tools/mcts.py:337`;
  CLI `--mcts-gumbel-m` (`tools/sim_self_play.py:2849`).
  ⚠ Measured constraint: coverage stays pinned at `m` visited edges
  at *every* sim budget — raising `--mcts-sims` concentrates the
  target rather than converging it (CLAUDE.md §Current status).
- **Completed-Q policy target
  `π = softmax(logits + σ(completed_q))` over ALL legal actions**
  **[ON]**. Unvisited actions fall back to the mixed value estimate
  `v_mix` instead of an implicit zero, so no simulation is wasted.
  `extract_gumbel_policy_target`, `tools/mcts.py:1958`;
  `_completed_q`, `tools/mcts.py:1559`.
- **σ q-transform with min-max rescale to [0,1]** **[ON]**.
  `σ(q) = (c_visit + max_b N(b)) · c_scale · rescale(q)`,
  `c_visit = 50`, `c_scale = 0.1`, `gumbel_rescale_q = True`
  (`tools/mcts.py:338-343`). *Why these exact values:* they match the
  reference implementation (mctx `qtransform_completed_by_mix_value`).
  **Provenance — this was a real bug:** until 2026-07-28 we ran
  `c_scale = 1.0` on raw Q with no rescale — the paper's constant
  without the paper's normalization — which multiplied Q differences
  ~50-80× and collapsed the distillation target to near-one-hot
  (measured: recruit target mass 0.000-0.002 against a ~0.16 prior).
  The rescale also buys **offset invariance**, so a drifting
  side-to-move value baseline can't shift the target.
  `_rescale_q`/`_gumbel_sigma`, `tools/mcts.py:1531`/`:1585`;
  derivation in `docs/design_constants.md` "Gumbel q-transform".
- **σ rescale spread floor `gumbel_rescale_floor = 1e-8`**
  **[OFF as a repair]** — 1e-8 is the legacy no-op value.
  *Why it exists:* a Q spread below the value head's own resolution
  (~1e-3, seen on 7.8% of late mini decisions) still receives the
  full ~8-logit σ gain, amplifying pure rank noise into a near-step
  target. Candidate repair value 0.01, held pending a pre-registered
  A/B. `tools/mcts.py:370`, `_rescale_q` comment `:1548-1555`.
- **Two-level (hierarchical) Gumbel candidate selection** **[OFF]**.
  `gumbel_hierarchical = False` (`tools/mcts.py:375`;
  `--mcts-hierarchical-gumbel`, `tools/sim_self_play.py:2783`).
  Actors would compete with their *total* prior mass instead of
  per-edge slivers, removing the structural halving advantage of
  single-edge actions like `end_turn`. Implemented at
  `tools/mcts.py:1629-1660`; held for a pre-registered A/B.
- **Distillation-target damping (the "prior ratchet" repair)**
  **[OFF]** — `distill_prior_discount = 1.0`,
  `distill_target_temp = 1.0` (`tools/mcts.py:365-366`; CLI
  `--distill-prior-discount` / `--distill-target-temp`,
  `tools/sim_self_play.py:2775`/`:2807`).
  *Why it exists:* the Gumbel target is **self-referential in the
  prior** — `softmax(log(prior) + σ(q))`. Measured on fixed-ToD mini
  maps: prior gaps ~3.9 logits vs ~0.5 logits of σ restoring force,
  so distillation re-teaches the prior each round and the iteration's
  fixed point is one-hot (`end_turn` p>0.9 collapse). λ<1 makes the
  prior a decaying memory of accumulated evidence: equilibrium logit
  gap = σ_gap/(1−λ), bounded by value evidence rather than prior
  history. `tools/mcts.py:344-366`;
  applied in `extract_gumbel_policy_target`, `tools/mcts.py:1995-1999`.
  **Note the symmetry contract:** these must be forwarded to spool
  workers, since the WORKERS build the training targets
  (`tools/sim_self_play.py:1147-1154`).
- **Unspent-budget spill.** Sequential halving can under-consume
  (single-candidate roots run 0 sims; per-phase floor splits drop a
  few). Leftover sims fall through to the classic PUCT loop so the
  "n_simulations total" contract — which tree-reuse visit accounting
  depends on — holds regardless of edge count.
  `tools/mcts.py:1870-1879`.

### 1.3 Root policy — classic AlphaZero root (fallback)

Active only under `--mcts-classic-root`.

- **Dirichlet root noise** `P' = (1−ε)P + ε·η`, `alpha = 0.3`,
  `eps = 0.25`, `add_root_noise = True`. Skipped entirely in Gumbel
  mode (exploration comes from the Gumbel draws instead).
  `_add_dirichlet_noise`, `tools/mcts.py:810`; constants `:237-239`;
  gating `:1831-1837`.
- **Visit-count temperature schedule.** Sample ∝ `visits^(1/τ)` for
  the first `temperature_decisions = 30` decisions of a game, then
  argmax. `temperature = 1.0`. *Why:* without it self-play games are
  near-deterministic and the data distribution collapses.
  `sample_action`, `tools/mcts.py:2031`; constants `:306-307`;
  schedule applied in `MCTSPolicy.select_action`,
  `tools/mcts_policy.py:413-425`.
- **Visit-count policy target.** `extract_visit_counts`,
  `tools/mcts.py:1926` (5-tuples; legacy 4-tuples still train).

### 1.4 First-play urgency (FPU)

- **`fpu_reduction = 0.25`** **[ON]** — an unvisited edge scores at
  `clamp(node.value − 0.25)` rather than AlphaZero's `Q=0`.
  *Why:* with `Q=0` init, in a clearly losing position every
  unvisited edge outranks the best known move, so the search
  degenerates into a one-visit sweep of 100-200 legal actions and
  never deepens — fatal at 25-100 sim budgets. 0.25 follows Leela's
  default; KataGo (Wu 2020 §5.1) reports the same shape.
  `tools/mcts.py:289`; `_puct_select`, `tools/mcts.py:786-789`;
  CLI `--mcts-fpu-reduction` (`tools/sim_self_play.py:2827`).
- **`root_fpu_reduction = 0.0`** **[ON]** — no FPU penalty at the
  root when noise is on, so Dirichlet-boosted priors can actually
  win a visit (KataGo does the same). `tools/mcts.py:296`.

### 1.5 Chance nodes — stochastic actions handled honestly

- **Chance nodes for attacks and recruits** **[ON]**
  (`chance_nodes = True`, `tools/mcts.py:392`). Every traversal of a
  stochastic edge re-forks the parent sim with a fresh seed salt and
  re-steps, sampling from the TRUE distribution (the sim is
  bit-exact); the edge keeps one child per distinct outcome state.
  *Why:* the edge's Q then converges to
  `E_outcome[V(adaptive response)]` — expectation AFTER the max, not
  before. EV-collapse (the alternative) systematically undervalues
  multi-pronged aggression. `_STOCHASTIC_ACTION_TYPES`,
  `tools/mcts.py:134`; `_select_one`, `tools/mcts.py:830`.
- **Exact outcome enumeration** **[ON]**
  (`exact_outcome_enumeration = True`, `tools/mcts.py:405`; disable
  with `--mcts-no-exact-outcomes`). Attack edges lazily compute the
  exact outcome distribution via a `prob_matrix`-style DP using the
  SAME parameters as the bit-exact resolver; while unseen mass
  remains, traversals keep sampling, and once observed children cover
  ~all the mass, selection switches to exact-probability choice with
  no sim fork. *Why:* zero Monte-Carlo noise and cheaper traversals in
  the common ~10-outcome case. Fights the DP refuses (petrify,
  possible advancement, berserk/complexity caps) fall back to sampling
  automatically. `tools/combat_outcomes.py:1-30`;
  coverage threshold `_EXACT_COVERAGE_EPSILON = 1e-3`,
  `tools/mcts.py:178` (parallels the engine's own truncations).
- **Adaptive outcome bucketing (Tier 2)** **[OFF]**
  (`outcome_buckets = False`, `tools/mcts.py:492`;
  `--mcts-outcome-buckets`). Groups outcomes into buckets sharing one
  representative network forward, refined only where member values
  diverge. Event hard-split (dead/slowed/poisoned flags) is never
  merged; the split trigger is a significance test.
  *Lit basis:* PARSS (Hostetler et al.) for the coarse→fine
  split-in-half backbone, OGA-UCT (Anand et al.) for the
  value-heterogeneity trigger. `tools/outcome_buckets.py:1-28`;
  MCTS integration `_record_and_maybe_split`, `tools/mcts.py:1119`.
  Only implemented on the serial Gumbel path — the CLI force-disables
  it under `--mcts-classic-root` rather than letting it silently
  no-op (`tools/sim_self_play.py:3416-3424`).
- **No-op resample sentinel `_NOOP_KEY`** **[ON]**. A stochastic step
  that changes nothing observable (canonically: a recruit attempt on
  a fog-occupied castle hex) is routed to a distinct pseudo-terminal
  instead of a child. *Why:* the transposition table maps the
  unchanged key back to the PARENT, so the descent self-loops
  forever, forking a sim per iteration until OOM — observed
  2026-06-13, one game ran 3.5h then MemoryError'd.
  `tools/mcts.py:138-156`.

### 1.6 Tree structure reuse

- **Transposition table** **[ON]** (`use_transposition_table = True`,
  `tools/mcts.py:435`). Paths converging on the same `state_key(gs)`
  share one `MCTSNode`, which is the correct PUCT semantic for
  `N(s)`. Built fresh per `mcts_search` call.
  *Measured:* hit rate is low (~0.4% at 20 sims/move) because
  per-action HP/MP/village mutations keep keys unique; the win comes
  from intra-turn move-reordering convergence. Hit/miss counts are
  surfaced on the returned root (`root.tt_hits`, `tools/mcts.py:1906`).
- **Subtree reuse across consecutive decisions** **[ON]**
  (`tree_reuse = True`, `tools/mcts.py:421`; disable with
  `--mcts-no-tree-reuse`). The caller reuses the played edge's
  subtree **iff `state_key(live) == state_key(searched child)`** —
  deterministic actions match and inherit the whole subtree's visits;
  combat RNG diverges and rebuilds from scratch.
  *Why safe:* zero unsoundness by construction, modulo the 64-bit
  `state_key` collision assumption the TT already makes. Dirichlet
  noise is re-applied to reused roots (matching Leela/KataGo).
  Stash + check in `MCTSPolicy.select_action`,
  `tools/mcts_policy.py:368-381`, `:475-484`.

### 1.7 Batching and parallel-in-tree

- **Virtual loss + batched leaf evaluation** — `batch_size = 1`,
  `virtual_loss = 1.0` in `MCTSConfig` **[OFF by default in the
  library]**, but the CLI auto-selects **B=16 on CUDA**
  (`tools/sim_self_play.py:3374-3386`, `--mcts-batch-size` default
  `None`). *Why the split:* on CPU at ~1750-token sequences PyTorch's
  batched MHA is a NET SLOWDOWN (measured B=1 33ms/sim, B=4 54ms,
  B=8 56ms); on GPU the inverse holds and B=1 starves the device.
  `tools/mcts.py:247-266`; `_run_sim_batch`, `tools/mcts.py:1421`.
  Virtual loss temporarily marks selected edges "bad" so parallel
  sims in a batch diverge; `_backup` undoes it (`tools/mcts.py:1202`).
- **Forward-on-GPU / sampler-on-CPU split** **[ON, CUDA-only]**.
  `_leaf_to_cpu` moves the model output and the two encoded tensors
  the sampler reads as values to CPU in ONE pass, then enumerates on
  host tensors. *Why:* `enumerate_legal_actions_with_priors` does
  dozens of per-actor `.item()`/`.tolist()` reads, each a serializing
  D2H sync, × n_sims leaves per move (measured on a T4: enumerate was
  26% of the rollout before counting syncs inside `forward`'s 41%).
  CPU inputs pass through untouched, so the CPU path is
  byte-identical. `tools/mcts.py:1246`; spec in
  `docs/gpu_perf_patches.md`.

### 1.8 Search budget control

- **`n_simulations`** — `100` in `MCTSConfig` (`tools/mcts.py:192`),
  **`50` at the training CLI** (`--mcts-sims`,
  `tools/sim_self_play.py:2882`).
- **Playout-cap randomization (KataGo, Wu 2019)** — decouples the
  moves that ADVANCE a game (cheap) from the moves that GENERATE
  training data (full search). A move is "full" with probability
  `playout_cap_prob` (full budget AND its target recorded);
  otherwise "fast" (`n_sims//4` by default, nothing recorded).
  **`[OFF]` in `MCTSConfig` (`tools/mcts.py:508`), `[ON]` at the
  training CLI** — `--mcts-playout-cap` is
  `BooleanOptionalAction, default=True`
  (`tools/sim_self_play.py:2882`, user ruling 2026-08-05); library
  and eval paths stay uncapped. `prob = 0.25`.
  *Why:* ~3-10× more games per GPU-hour (KataGo) while targets still
  come from full-strength searches. Implementation:
  `MCTSPolicy.select_action`, `tools/mcts_policy.py:388-395`,
  `:437-441`; `n_sims_override` in `mcts_search`, `tools/mcts.py:1845`.
- **Wall-clock `time_budget`** **[OFF]** (`None`,
  `tools/mcts.py:245`) — aborts a search early; checked in both the
  Gumbel and classic loops.
- **Cliffness-driven adaptive sim budget** **[OFF]**
  (`adaptive_sim_budget = False`, `tools/mcts.py:473`). Would
  interpolate `n_simulations` between `n_simulations_min = 100` and
  `n_simulations_max = 400` by root cliffness. *Why off:* the linear
  interpolation shape is uncalibrated; root cliffness is logged
  unconditionally so a schedule can be chosen from real positions
  first. `_adaptive_n_sims`, `tools/mcts.py:1226`.

### 1.9 Search-side value shaping (all default OFF)

- **Cliffness-scaled bootstrap weighting** **[OFF]**
  (`cliffness_bootstrap_alpha = 0.0`, `tools/mcts.py:471`).
  On backup, treat a non-terminal leaf's `v` as a noisy estimate with
  variance `alpha·cliffness²` and shrink toward a uniform prior:
  `scale = σ²_prior / (σ²_prior + α·cliffness²)` with
  `σ²_prior = 1/3` (variance of uniform on [−1,+1], which is also
  what a freshly-initialized C51 head emits). At α=1 this is the
  Bayes-optimal posterior mean — zero free hyperparameters once you
  accept the prior. Terminal leaves bypass it (their value is exact);
  visit counts are never scaled, only `w_value`.
  `_backup`, `tools/mcts.py:1146-1198`;
  `_BOOTSTRAP_PRIOR_VAR`, `tools/mcts.py:1116`.
- **Aux-head value bonus** **[OFF]** (`aux_value_bonus = 0.0`,
  `tools/mcts.py:215`; `--mcts-aux-value-bonus`). Leaf value becomes
  `clamp(v + bonus·aux_pred, −1, 1)`. *Why it exists:* the anatomy
  diagnostic showed ~zero village captures in 100-turn games —
  nothing in-horizon rewarded expansion, because in-search terminals
  are never reached mid-game. `_aux_adjusted`, `tools/mcts.py:696`.
  ⚠ Under the default Gumbel root this also shapes the distilled
  policy target (intended, documented in the flag help). The CLI
  warns if set without the aux head (`tools/sim_self_play.py:3458`).
- **Moves-left utility in PUCT** **[OFF]**
  (`moves_left_utility = 0.0`, `tools/mcts.py:205`). Subtracts
  `w·Q·M` from a visited edge's score, so winning lines prefer
  ending sooner and losing lines prefer dragging.
  *Why it exists:* the Tier-a verdict — 42/66 eval games died to the
  action cap; nothing priced time. `_puct_select`,
  `tools/mcts.py:801-803`. `m_sum` backup is perspective-free (game
  length, not value → no sign flip), `tools/mcts.py:1215-1216`.
- **Detector-advice conditioning at the root** **[OFF]**
  (`advice = False`, `tools/mcts.py:223`). See §3.5 — wired
  end-to-end but **empirically refuted**.

### 1.10 Search correctness guards

- **`SIM_FORK_GUARD=1`** **[OFF, opt-in env var]**. Asserts, around
  every `mcts_search`, that the caller's live state is bit-identical
  before and after, via `deep_state_fingerprint`.
  *Why it exists:* three separate shipped bugs where a search fork
  mutated state shared across forks — the village-ownership bit
  (`fa95da5`), the Aethermaw terrain morph, the `first_time_only`
  event latch — **all three invisible to `state_key`**. Costs two
  full-state hashes per search, so it is a smoke-run/repro flag, free
  when off. `tools/mcts.py:128`, assertion at `:1914-1922`.
- **Live-state deepcopy contract.** `MCTSPolicy.select_action`
  raises if handed the live `sim.gs` rather than a snapshot —
  otherwise `sim.step` mutates the recorded training target and the
  stored action indices stop matching the state's re-encoding.
  `tools/mcts_policy.py:339-355`.

---

## 2. Value learning

### 2.1 Distributional value head (C51)

- **Categorical value head, K=51 atoms on fixed support [−1, +1]**
  **[ON]**. Replaces a tanh-bounded scalar. The scalar the rest of
  the codebase reads is `E[Z(s)]`; no tanh is needed because softmax
  cannot put mass outside the support. *Why:* categorical CE trains
  both the distribution's mean AND its spread — noisy returns at a
  state push the network toward a wider predicted distribution.
  K=51 is the Bellemare et al. (2017) default; at range 2.0 that is
  0.04/bin. `wesnoth_ai/model.py:41-59`, head at `:289-298`,
  forward at `:466-472`.
- **Head reads a dedicated `[CLS]`-style global token**, not a mean
  pool. *Why:* earlier mean-pooled variants diluted the signal
  ~1700× over all token positions. `wesnoth_ai/model.py:278-288`.
- **Categorical-CE loss against a bin-projected target** **[ON]**.
  C51 Algorithm 1 linear interpolation between adjacent bins; targets
  outside the support clamp to the edges (consistent with
  `value_clip`). `_project_returns_to_atoms`, `wesnoth_ai/trainer.py:600`;
  `_categorical_value_loss`, `:635`.
- **`value_clip = 1.0`** **[ON]** — returns/z are clamped to the
  head's support. Without it the loss chases unreachable targets and
  the estimate saturates with infinite gradient pressure.
  `wesnoth_ai/trainer.py:208`.
- **`value_coef = 0.5`** **[ON]** (`wesnoth_ai/trainer.py:150`;
  `--value-coef` at `tools/sim_self_play.py:3293`). Raised in
  campaigns because the value head is the diagnosed bottleneck.

### 2.2 Cliffness — free uncertainty estimate

- **`cliffness = std(Z(s))`** **[ON, always computed]**. A
  heteroscedastic uncertainty estimate that comes for free from the
  categorical head; marks states where small perturbations imply big
  value swings. `wesnoth_ai/model.py:470-472`.
  Root cliffness is logged unconditionally
  (`tools/mcts.py:1857-1861`) even though both consumers (§1.8, §1.9)
  are default-off — the point is to collect the distribution before
  picking a schedule. Normalizer `cliffness_max = 0.577 ≈ 1/√3`
  (std of the continuous uniform on [−1,+1]); derivation in
  `docs/design_constants.md`.

### 2.3 Value targets

- **Terminal `z` distilled onto every visited state** **[ON]** —
  AlphaZero convention: mid-game states train toward the GAME's
  terminal outcome, not a value-net bootstrap.
  `MCTSExperience.z`, `wesnoth_ai/trainer.py:101-108`;
  assignment in `MCTSPolicy.finalize_game`,
  `tools/mcts_policy.py:570-580`.
- **Honest `z = 0` for draws** **[ON]** —
  `train_draw_tiebreak = False` (`tools/mcts_policy.py:125`;
  `--train-draw-tiebreak`, `tools/sim_self_play.py:3021`).
  *Why:* material-z draw labels made "predict material" the dominant
  lesson (~93% of ladder games are draws) and measurably eroded
  win/loss discrimination — human-corpus late-game AUC 0.88 → 0.60 in
  ~80 iterations; `r_material/r_outcome` rose 1.28 → 2.18
  (2026-07-10). The material tiebreak remains a SEARCH preference
  (§6.3) — the two were deliberately decoupled.
- **`draw_value_weight = 1.0`** **[ON = legacy]**
  (`wesnoth_ai/trainer.py:175`; `--draw-value-weight`).
  Setting it to `0` gives decisive-only value learning: draws still
  feed the aux and moves-left heads but stop flattening the value
  head. *Why the knob exists:* ~71% of incoming states were draws;
  even with honest z=0 labels their gradient mass eroded win/loss
  discrimination WITH a 512-state rehearsal anchor in place.
  The loss normalizes by TOTAL WEIGHT, not N, so decisive states keep
  full-strength gradient regardless of the batch's draw share.
  `wesnoth_ai/trainer.py:1154-1166`.
- **Value label smoothing** **[OFF]**
  (`value_label_smoothing = 0.0`, `wesnoth_ai/trainer.py:168`;
  `--value-label-smoothing`). Mixes ε uniform mass into the projected
  target, TRAIN loss only (eval CE stays unsmoothed for
  comparability). *Why it exists:* with hard ±1 targets and many
  replay updates the head collapses toward extreme-atom spikes
  (measured 2026-07-07: Z entropy 1.86 → 1.13, max-atom p 0.39 →
  0.58 in 46 iters) and a confidently-wrong spike makes held-out CE
  explode. `_categorical_value_loss`, `wesnoth_ai/trainer.py:639-655`.

### 2.4 Auxiliary prediction heads (architecture-changing, both OFF)

- **Auxiliary material-margin head (KataGo §3.5)** **[OFF]**
  (`aux_score = False`, `wesnoth_ai/model.py:227`;
  `--mcts-aux-score`). Tanh-bounded predicted final material margin
  from the global token; loss weight `aux_coef = 0.15`
  (KataGo uses ~0.15). *Why:* a denser signal than win/loss z that
  regularizes the shared trunk. Head at `wesnoth_ai/model.py:303-304`;
  loss `wesnoth_ai/trainer.py:1176-1182`.
  **Target refinement (2026-07-12):** the aux target is the NEXT
  recorded state's margin, not the game's final margin — one-step
  material prediction credits captures/kills the moment they happen,
  and since the next recorded state may follow the opponent's reply
  it teaches captures that HOLD. `tools/mcts_policy.py:581-590`.
- **Moves-left head (Lc0-style)** **[OFF]**
  (`moves_left = False`, `wesnoth_ai/model.py:228`;
  `--mcts-moves-left`). Sigmoid-bounded fraction of the turn budget
  remaining, normalized by a fixed `MOVES_LEFT_NORM_TURNS = 200.0`
  (not per-game `max_turns`, so "0.1 left" means the same wall
  distance on every map). Loss weight `moves_left_coef = 0.1`.
  *Why:* a dense TEMPO signal the sparse z cannot provide
  (2026-07-04 action-spam diagnosis).
  `wesnoth_ai/model.py:309-310`; target
  `tools/mcts_policy.py:591-595`; loss `wesnoth_ai/trainer.py:1186-1192`.
  Both heads **stick to a checkpoint**: warm-starting an aux-on
  checkpoint re-enables the head even without the flag, otherwise
  `strict=False` loading would silently DROP the trained head as
  "unexpected keys" (`tools/sim_self_play.py:3250-3251`;
  `tools/supervised_train.py:1024-1059`).

### 2.5 Value-learning diagnostics (these ARE the success metrics)

- **`fresh_value_ce` — the default success metric** **[ON when
  replay is on]**. Value CE on a ≤256-state sample of THIS
  iteration's incoming games, measured BEFORE any gradient step
  touched them. *Why:* distribution-matched (unlike the frozen
  holdout, which drifts off-distribution as play evolves) and never
  seen by the net (unlike the train value loss, measured on replay
  samples the net has taken many steps on) — the gap between the two
  IS the memorization measurement.
  `MCTSPolicy.train_step`, `tools/mcts_policy.py:862-887`;
  `_trainer_eval_value_metrics`, `wesnoth_ai/trainer.py:1391`.
  **Read it FLOOR-RELATIVE** (`fresh_value_ce − fresh_ce_floor`).
- **`fresh_ce_floor` — state-blind CE floor** **[ON]**. CE of the
  best state-blind predictor (the batch's empirical projected-z
  mixture). A learned head must score BELOW it; a HIGH floor means
  the games' outcomes are inherently mixed and caps what any head can
  achieve on that batch. `wesnoth_ai/trainer.py:1478-1480`.
- **`fresh_ce_std`** **[ON]** — game-weighted spread, so an
  iteration-to-iteration move can be told from ~256-state probe
  noise. `wesnoth_ai/trainer.py:1481-1487`.
- **`fresh_pred_entropy`** **[ON]** — mean entropy of predicted
  `Z(s)` in nats (uniform over 51 atoms = ln 51 ≈ 3.93): the
  continuous overconfidence curve. `wesnoth_ai/trainer.py:1473-1474`.
- **`fresh_decisive_ce`** **[ON]** — CE restricted to ±1 incoming
  states; the gate metric under `draw_value_weight=0`, where pooled
  CE is structurally inflated by z=0 states the head is deliberately
  not trained on. `tools/mcts_policy.py:880-887`.
- **`z_*_frac` and `z_*_frac_w`** **[ON]** — target composition in
  TWO normalizations. *Why both:* the raw census is not a gradient
  metric (long games inflate it); the game-weight-weighted version
  matches actual gradient contribution. 2026-07-22: an unweighted
  census of 0.19 draws was misread as 20% of the gradient when the
  weighted share was ~5%. `_attach_z_composition`,
  `tools/mcts_policy.py:928-960`.
- **Frozen holdout probe** **[OFF]** (`--holdout-size 0`,
  `tools/sim_self_play.py:2492`). Diverts WHOLE games out of training
  (states within a game are correlated; splitting one would leak),
  taking ≤`holdout_per_game_cap = 64` randomly-sampled states per
  diverted game. *Why the cap:* the original whole-game fill made a
  512-state holdout out of ~2 games and measured those 2 games'
  idiosyncrasies, not generalization (2026-07-07).
  Persisted beside the checkpoint (`<ckpt>.holdout`) with an
  index-basis stamp, because per-restart resampling made capacity
  trends unreadable (levels jumped 0.44↔0.88 on set changes,
  2026-07-18). `tools/mcts_policy.py:150-168`, `:698-810`.
  **Role:** the holdout is the stall TRIPWIRE, not the success gauge.
- **Boundary-consistency probe `V(s_pre) + V(s_post)`** **[ON]** —
  mean over sampled side-switch pairs of recorded states. Zero-sum
  calibration predicts ~0; the fogged-play WYSIATI bias measured
  +0.4..+0.65 on the 2026-07-28 lineage (fogless ~0). Pairs are
  harvested at `finalize_game` while adjacency is still known
  (replay shuffling destroys it) and only for games that TRAIN.
  `harvest_boundary_pairs`, `tools/mcts_policy.py:962`;
  `TrainStats.boundary_sum`, `wesnoth_ai/trainer.py:257-271`.
- **Value-only training entry points** used by the human-corpus
  work: `Trainer.step_value_from_raw` (`wesnoth_ai/trainer.py:1264`),
  `Trainer.values_from_raw` (`:1329`),
  `Trainer.eval_value_metrics_from_raw` (`:1354`).

---

## 3. Policy learning

### 3.1 MCTS distillation (the production objective)

- **Factored cross-entropy against the search target** **[ON under
  `--mcts`]**. Decomposes joint CE across four heads:
  `log P(actor) + log P(type|actor) + log P(target|actor,type) +
  log P(weapon|actor)`. Mathematically identical to flat joint CE,
  but avoids materializing the `A·T·H·4` joint.
  `_mcts_factored_policy_loss`, `wesnoth_ai/trainer.py:679`.
- **Type-conditional target masks in the loss.** ATTACK and MOVE get
  separate `[A,H]` rows; the union mask is a legacy fallback for
  recruit/end_turn and pre-type-head data.
  `wesnoth_ai/trainer.py:785-812`.
- **Vectorized loss accumulation** **[ON]**
  (`vectorized_mcts_policy_loss = True`, `wesnoth_ai/trainer.py:195`).
  Buckets `(index, count)` pairs per cached log-prob vector and
  reduces each with one `index_select`+sum, collapsing the backward
  graph from O(visit-count tuples) to O(unique vectors)
  (~1.3-2× `step_mcts` at 300-900 tuples/state, the Gumbel regime).
  NOT bit-identical (float summation is reassociated, ~1e-7 rel
  drift), hence the gate; `test_mcts_policy_loss_vectorized` asserts
  grads match the loop within 1e-5. `wesnoth_ai/trainer.py:839-955`.
- **Rollout/loss prior symmetry contract** — the re-forward must
  rebuild the SAME legality masks and the SAME prior biases the
  sampler applied, or the CE fights a target the live priors never
  produced. Enforced by threading `decision_step` from search →
  `MCTSExperience` → loss, and by re-applying
  `prior_bias_end_turn`. `wesnoth_ai/trainer.py:717-724`;
  `MCTSExperience.decision_step` at `:136-141`.
  The same contract is why env-var prior biases and distillation
  damping must be exported to spool workers
  (`tools/sim_self_play.py:1147-1161`).

### 3.2 Per-game and per-side gradient normalization

- **`game_weight = 1 / (2 · max(side_floor, n_side))`** **[ON]** —
  every GAME contributes equally regardless of length, and within a
  game every SIDE contributes an equal half.
  *Why per-game:* a 190-turn ladder draw otherwise outweighed a
  10-turn mini ~19:1 in state count (2026-07-12).
  *Why per-side (2026-08-05):* per-game pooling made gradient mass
  track decision count — the winner (more units alive, more actions)
  out-weighted the loser ~54/46 in value targets, and a
  prior-collapsed passive side, taking few actions, starved its own
  correction. `tools/mcts_policy.py:539-605`; consumed in every loss
  term of `step_mcts` (`wesnoth_ai/trainer.py:1009-1012`).
- **Midgame weight floor `MIDGAME_GW_FLOOR = 8`** (halved to 4
  per-side) **[ON for human-derived midgame starts]**. A
  continuation cut near the end may record 1-3 states whose decisive
  label mostly credits the HUMAN's play; without a floor one such
  state can transiently own most of a minibatch.
  `tools/mcts_policy.py:65`, applied `:565-566`.

### 3.3 REINFORCE path (default CLI, not the campaign path)

- **REINFORCE with a value baseline + entropy bonus**
  (`Trainer.step`, `wesnoth_ai/trainer.py:329`).
- **Re-forward instead of retained graphs** **[ON, architectural]**.
  Transitions store indices + a GameState reference, never tensors;
  the trainer re-forwards at train time. *Why:* the old design pinned
  several GB (4 games × ~200 actions × ~8MB of activations) and
  swapped the machine. Cost is ~2× forwards, but RAM was the actual
  bottleneck. `wesnoth_ai/trainer.py:1-26`.
- **Two-pass advantage computation.** Pass 1 (no_grad) collects
  values; pass 2 builds per-chunk loss and backwards per chunk, with
  one `optimizer.step()` at the end. Both passes run in `eval()` so
  the two value forwards see identical activations — with
  dropout=1e-4 the old asymmetry dragged the value head's signal.
  `wesnoth_ai/trainer.py:380-428`.
- **`normalize_advantages = True`** **[ON]**
  (`wesnoth_ai/trainer.py:196`).
- **`entropy_coef = 0.001`** **[ON]**. Lowered from 0.01 after the
  first 22 train_steps held entropy ~8.3 (near max) — the bonus was
  dominating the tiny shaping gradients and preventing the policy
  from ever committing. **Not exposed by any `sim_self_play` flag.**
  `wesnoth_ai/trainer.py:177-182`.
- **`gamma = 0.99`**, `_compute_returns` walks trajectories in
  reverse with a `done` continuation mask
  (`wesnoth_ai/trainer.py:149`, `:589`).
- **Random (not strided) subsampling to
  `max_transitions_per_step = 4000`** **[ON]** in BOTH paths.
  *Why:* stride-N correlates the kept set with episode position — at
  γ=0.99 over 200 steps, near-terminal returns carry ~1.0× weight
  vs ~0.13× at step 0, and stride keeps near-terminal transitions
  preferentially, biasing the return distribution.
  `wesnoth_ai/trainer.py:369-378`, `:979-991`.
- **Gradient clipping `grad_clip = 1.0`** **[ON, both paths]**,
  over model + encoder parameters
  (`wesnoth_ai/trainer.py:559-562`, `:1221-1224`).
- **AdamW, `lr = 1e-4`, `weight_decay = 1e-4`, NO LR schedule**
  in the self-play path — verified: no scheduler, no warmup, and no
  CLI flag exposes LR. `wesnoth_ai/trainer.py:147-148`, `:323-327`.
  (The supervised path DOES have a cosine schedule — see §4.5.)

### 3.4 Legality masking — the contract

The mask is a **pure function of observable state**. It answers "what
can the policy validly attempt right now, given the information it
has" — NOT "what will the engine accept" (that would need god-view
fog truth) and NOT "what does the model want" (that would let it
infinite-loop). Full contract in `CLAUDE.md` §6 and
`wesnoth_ai/action_sampler.py:42-53`.

- **Masks are rebuilt, never stored on the Transition** **[ON]** —
  deterministic in `game_state`, so sampling and re-forward recompute
  identically and rollout/training symmetry holds by construction.
  `_build_legality_masks`, `wesnoth_ai/action_sampler.py:1030`.
- **TRUE single-turn reachability for MOVE** **[ON]**. Replaced
  crow-flies `dist ≤ moves` with the shared Wesnoth-default planner
  (`tools/pathfind_sim.unit_reach`) on a `ReachContext` built from the
  acting side's OBSERVABLE state: terrain costs, visible-enemy
  blocking, ZoC, ally pass-through. `wesnoth_ai/action_sampler.py:1166-1232`.
- **ATTACK legality = adjacent now, or can land adjacent this turn**
  **[ON]**. `wesnoth_ai/action_sampler.py:1250-1264`.
- **RECRUIT legality = BFS over the leader's connected castle
  network**, leader must be on a keep, plus an **affordability gate**
  so the policy never burns a decision on an unaffordable recruit.
  `_recruit_hex_mask`, `wesnoth_ai/action_sampler.py:1449`;
  affordability `:1344-1396`.
- **Per-turn rejection sets on `gs.global_info`** **[ON]** —
  `_recruit_rejected_hexes` and `_move_rejected_hexes`. Cleared at
  `init_side`. *Why per-turn:* persisting across turns would model
  knowledge a human doesn't have (the enemy may have moved).
  The recruit set is ALSO mirrored into the encoder as a per-hex bit
  (§5.3) so mask and model read the same state; the move set is
  deliberately mask-only. `wesnoth_ai/action_sampler.py:1233-1249`,
  `:1463-1481`.
- **Hidden enemies are treated as empty hexes** **[ON]** — you may
  attempt to MOVE into a fog-hidden hex (the engine reveals on
  contact) but may NOT click-to-attack an unseen unit.
  `wesnoth_ai/action_sampler.py:1097-1136`.
- **Petrified/scenery units are inert (occupancy code 3)** **[ON]** —
  they block movement but are never attack targets, mirroring Wesnoth
  `mouse_events.cpp:753`. `wesnoth_ai/action_sampler.py:1089-1096`.
- **`_NEG_INF = -1e9`, not `-inf`** **[ON]** — `-inf` can NaN a
  downstream softmax. Plus `_safe_softmax`, which detects the
  all-masked slice and zeroes it instead of returning the wrong
  "uniform over equally-illegal" result.
  `wesnoth_ai/action_sampler.py:125-127`, `:822-840`.
- **Gumbel-max sampling instead of `torch.distributions.Categorical`**
  **[ON]** — `argmax(logits + Gumbel(0,1))`. *Why:* multinomial /
  scatter ops are unimplemented on the torch-directml RX-6600 path.
  `wesnoth_ai/action_sampler.py:130-154`.
- **`FORBID_IDLE_END_TURN`** **[OFF]** — would make `end_turn`
  illegal while a recruit is affordable+placeable or a unit still has
  a legal move. *Why retired, with evidence:* it binds on 92.4% of
  decisions (1,680/1,819) yet the free policy picks `end_turn` only
  ~7% when allowed, and it forbids **69% of HUMAN end_turns** in the
  corpus (1,169/1,706 over 60 games) — violating the mask contract
  and distorting the behavior-cloned prior. Raw epoch-3 SL policy
  scored ladder 10/10 and mini 9/10 decisive with the gate OFF.
  `wesnoth_ai/constants.py:88-105`;
  gate `wesnoth_ai/action_sampler.py:1398-1434`.
- **No pruning.** There is no top-k, beam, or threshold pruning of
  the action set anywhere — only the legality mask and skipping
  zero-probability children during enumeration.

### 3.5 Prior hardcoded bias (formerly "combat oracle") — ALL OFF

Standing policy (user order 2026-08-06): **every hand-placed prior
nudge defaults OFF and is activated only on explicit order.**
Pinned by `tests/test_mcts.py::test_prior_hardcoded_bias_defaults_off`.
`wesnoth_ai/constants.py:168-182`.

- **Attack-target bias on hex logits** **[OFF]**
  (`COMBAT_TARGET_ALPHA = 0.0`, `wesnoth_ai/constants.py:180`).
  Would add `alpha · expected_attack_net_damage` per (actor, enemy
  hex), applied BEFORE masking so it can't leak onto illegal hexes.
  `wesnoth_ai/action_sampler.py:1265-1298`.
- **Attack-type bias on `P(ATTACK|actor)`** **[OFF]**
  (`COMBAT_TYPE_ALPHA = 0.0`, `:181`). Fires only when
  `max_j(net) > 0`, so a reachable-but-unfavorable enemy never nudges
  toward attacking. `wesnoth_ai/action_sampler.py:1299-1309`.
- **The oracle model itself** (`wesnoth_ai/combat_oracle.py`) is live
  code with zero effect at default settings. It is a first-order
  analytic estimate: `hit_chance × strikes × dmg × (1−resist)`, with
  overkill capping, asymmetric kill bonus (`+max(8, cost·0.3)`) and
  own-death penalty (`−max(12, cost·0.4)`) — deliberately asymmetric
  because losing your own unit is worse. Hit chances are hardcoded
  0.5 (terrain defense not consulted); ToD, weapon specials, and
  abilities are explicitly punted. `wesnoth_ai/combat_oracle.py:1-136`.
  ⚠ Its docstring still claims "default 0.1" — stale.
- **Anneal schedule `combat_alphas_at(decision_step)`** — machinery
  **[ON]** but multiplies 0.0, so no effect. Linear decay from the
  configured alpha to `0.1 ×` it over
  `COMBAT_ANNEAL_HORIZON = 1_000_000` decisions, flat after.
  `wesnoth_ai/action_sampler.py:94-117`;
  `wesnoth_ai/constants.py:187-197`.
  The `decision_step` plumbing (persisted in checkpoints, threaded
  through search and loss, advanced for spool games, rolled back on
  bounced decisions, resettable via `--reset-decision-step`) remains
  fully wired and is what a future re-activation would ride on.
- **`end_turn` prior bias on mini-category games** **[OFF]** —
  activated per-run via `WESNOTH_PRIOR_BIAS_END_TURN_MINI=<float>`
  (negative = against passing). Env-var by design so spool workers
  and the trainer re-forward read the identical value (symmetry
  contract). `prior_bias_end_turn`,
  `wesnoth_ai/action_sampler.py:686-704`.
- **Detector-advice channel** **[OFF]** (`advice = False`,
  `wesnoth_ai/model.py:229`; `--mcts-advice`). A SEPARATE
  cross-attention block (not part of the main transformer, so
  existing checkpoints load byte-identically): actor tokens attend to
  advice tokens built from swap-detector opportunities; a per-actor
  softplus GATE is the learnable scale; `advice_out` is **zero-init**
  so the graft contributes nothing at load and the model must learn
  the scale up from zero. Fully wired — model
  (`wesnoth_ai/model.py:322-341`, `:423-429`), search root
  (`tools/mcts.py:735-737`), trainer re-forward with grad
  (`wesnoth_ai/trainer.py:1098-1116`), telemetry
  (`advice_grad_share`, `advice_out_norm`,
  `wesnoth_ai/trainer.py:1202-1219`).
  ⚠ **Empirically refuted, not aspirational:** the channel was
  measured to carry no information against a placebo control
  (CLAUDE.md §Current status; `docs/autonomous_run.md` Cycle 32,
  which also retracts `advice_out_norm` as evidence). Kept for
  provenance; do not treat as a working feature.

### 3.6 Model architecture (policy-relevant)

- **Transformer over five concatenated token streams** with additive
  token-kind embeddings so self-attention can tell a hex from a
  recruit. `wesnoth_ai/model.py:68-74`, `:391-405`.
- **Pointer-network target head** — target logits are
  `q(actor) · k(hex)ᵀ / √d`, so the action space scales with the
  board rather than being a fixed output layer.
  `wesnoth_ai/model.py:454-456`.
- **Per-actor type head (ATTACK/MOVE)**, meaningful only for UNIT
  slots. Old checkpoints lacking it initialize fresh under
  `strict=False`. `wesnoth_ai/model.py:259-264`.
- **`dropout = 1e-4`, deliberately not 0** **[ON]** — any dropout>0
  gates off PyTorch's `TransformerEncoderLayer` "better-transformer"
  fast path, whose fused op is unimplemented on torch-directml and
  silently falls back to CPU, shuttling every activation over PCIe
  per layer. 1e-4 is statistically a no-op.
  `wesnoth_ai/model.py:216-225`.
- **`enable_nested_tensor=False`** **[ON]** — PyTorch's nested-tensor
  path is a prototype ~2× slower than dense on CPU with
  `src_key_padding_mask`. `wesnoth_ai/model.py:246-254`.
- **`forward_batch` with key-padding mask** **[ON]** — one padded
  transformer pass over B states returning per-sample outputs with
  the exact single-sample shapes, so downstream code needs no
  changes. Includes an assert that no row is fully masked (an
  all-masked row NaNs the softmax and silently poisons the whole
  batch's gradient through the shared encoder).
  `wesnoth_ai/model.py:506`, assert at `:586-587`.
- **Opt-in bf16 inference autocast** **[OFF]**
  (`--infer-bf16`, `tools/sim_self_play.py:2797`). Trunk+heads run
  bf16 on CUDA at inference only; outputs are cast back to float32 so
  numpy consumers never see bf16. Training math untouched.
  `wesnoth_ai/model.py:360-380`.
- **Lazy `marginal_type_logits`** **[ON]** — was a per-forward field
  costing ~5-9% of every forward with a single test as its only
  reader; now a recompute-on-access property.
  `wesnoth_ai/model.py:191-204`.
- **Inference-weight snapshot** **[ON]** — rollout runs against a
  separate `_inference_model`/`_inference_encoder` pair, refreshed by
  in-place `load_state_dict` under the policy lock after each train
  step. *Why:* workers can call `select_action` while gradient
  compute is in flight; the copy is the atomicity boundary.
  `wesnoth_ai/transformer_policy.py:388-405`.

---

## 4. Imitation learning / warm-start

Entry points: `tools/build_imitation_dataset.py` (dataset),
`tools/supervised_train.py` (trainer), `configs/imitation.json`,
`tools/net2net.py` (growth).

### 4.1 Behavior cloning objective

- **Four-head cross-entropy on human actions** **[ON — it is the
  objective]**: actor + type + target + weapon.
  `_loss_parts_for_output`, `tools/supervised_train.py:512-666`.
- **No legality mask during SL** **[ON, deliberate]** — the observed
  human action is legal by construction, and applying the approximate
  mask risked `-inf`-ing the ground truth and blowing the loss up to
  ~1e9. The mask applies only at rollout.
  `tools/supervised_train.py:549-557`.
- **Label smoothing 0.05** **[ON, hardcoded, no flag]** on all four
  head CEs. `_LABEL_SMOOTHING`, `tools/supervised_train.py:509`.
- **Per-head "fired" accounting** **[ON]** — running averages take
  only the pairs where a head actually fired, so `actor = 2.5` is
  comparable to `ln(num_classes)`.
  `tools/supervised_train.py:448-453`.

### 4.2 Imitation-mode data shaping (`--imitation-config`) — [OFF] by default

The flag is `--imitation-config`, default `None`
(`tools/supervised_train.py:1884`). `configs/imitation.json` supplies
the values below.

- **Winners-only policy filter** **[ON when imitation mode is on]** —
  `policy_winners_only: true`. Policy CE trains on the winning side's
  actions only (`p_w = 0.0` zeroes all four heads for loser pairs);
  the loser's states still contribute VALUE supervision, "which is
  where loser data earns its keep". Loser pairs not drawn as value
  states are skipped before encoding — a real compute saving.
  `tools/supervised_train.py:1517-1538`.
- **Per-game equal weighting** **[ON when imitation mode is on]** —
  `per_game_weight: true`; `w = median_actions / winner_actions`,
  clipped to `[0.25, 4.0]` to bound `E[w²]` (unbounded per-game
  weights destabilize file-sequential batches).
  `tools/supervised_train.py:1244-1254`.
  ⚠ Weighting is by ACTION COUNT only. **There is no Elo weighting
  and no outcome-magnitude weighting anywhere** — verified by grep.
- **Deterministic hash-based holdout** **[ON when imitation mode is
  on]** — `sha1(source_path) % 10000 < holdout_fraction·10000`
  (`holdout_fraction: 0.02`), stable across rebuilds.
  `tools/build_imitation_dataset.py:117-120`.
- **Outcome-class selection** — `outcome_classes: ["explicit"]`
  (leader death / surrender only); `inferred` games gated on a
  material ratio can be added without rebuilding stats.
  `tools/build_imitation_dataset.py:60-61`.

### 4.3 Value supervision in SL

- **Value-from-outcome C51 CE** **[ON]** — `--value-loss-weight`
  default `1.0`. Corpus games are decisive by construction so the
  projected target is one-hot on an edge atom and CE reduces to
  `-log p(edge)`. Auto-disabled with a warning if
  `value_corpus_index.jsonl` is missing.
  `tools/supervised_train.py:641-649`, `:1309-1311`.
  *Why value trains alongside policy:* the value-frozen epoch-0 pass
  let policy gradients reshape the trunk under the head — late AUC
  0.79 → 0.63 (user 2026-07-16).
- **Value-state subsampling, k=16/game** **[ON]** —
  `--value-states-per-game` default 16, selection probability
  `min(1, k/n_commands)`. *Why:* the pair stream is FILE-SEQUENTIAL,
  so a batch used to fill with 64 same-label states of one game —
  "the actual instability mechanism behind `value_auc` oscillating
  0.36↔0.76". `tools/supervised_train.py:1276-1308`.
- **`--reinit-value-head`** **[OFF]** — deletes all `value_head.*`
  keys AND skips the optimizer-state restore (stale Adam moments
  would otherwise land on the re-initialized tensors).
  *Provenance (imitation A/B, 2026-08-08):* the warm trunk+policy
  dominate everywhere (holdout CE 3.107 vs 3.449), but the warm VALUE
  head — trained on search-backed z targets — fights outcome
  supervision all run (AUC oscillating 0.52-0.89, final 0.538) while
  a fresh head climbs cleanly to 0.951.
  `tools/supervised_train.py:1108-1151`.
- **Value AUC probe on every held-out eval** **[ON]** —
  `P(E[V]_winner-to-move > E[V]_loser-to-move)` with tie-halving;
  cheap enough to ride every eval so trunk-drift damage shows in the
  CURVE. `tools/supervised_train.py:901-913`.

### 4.4 Class balancing

- **Per-action-type loss weights** **[ON, baked-in defaults]** —
  `move 0.189, attack 0.628, recruit 1.748, recall 0.0,
  end_turn 1.435`, inverse-frequency from a 5,000-replay scan
  (`tools/compute_action_type_weights.py`), scaled so the mean of
  nonzero weights is 1.0. *Why:* without upweighting, moves dominate
  (~65% of corpus actions) and the model learns to never recruit.
  `_DEFAULT_ACTION_TYPE_LOSS_WEIGHT`, `tools/supervised_train.py:476-484`;
  override via `--action-type-weights` /
  `configs/action_type_weights.json`.
  ⚠ Applied to the actor and type heads only — target and weapon
  losses are unweighted, so a `recall` pair (actor weight 0.0) still
  trains the target head at full strength.
  **Not used by the self-play path.**

### 4.5 SL optimization and plumbing

- **AdamW `lr=1e-4`, `weight_decay=1e-4`** +
  **`CosineAnnealingLR(T_max=epochs, eta_min=0.05·lr)`** **[ON]** —
  stepped once per epoch AFTER the checkpoint save, so the saved
  optimizer state carries the lr that produced its gradients.
  Scheduler state is not checkpointed; it is reconstructed by
  replaying `resumed_epoch` steps.
  `tools/supervised_train.py:1071-1090`, `:1180-1186`.
  (This is the ONLY LR schedule in the codebase.)
- **Gradient clipping `max_norm = 1.0`** **[ON, hardcoded]**, at all
  four flush sites. `tools/supervised_train.py:808`, `:1475`,
  `:1628`, `:1751`.
- **Parallel dataset streaming, one message per replay** **[OFF by
  default: `--workers 0`]** — spawn-context workers; batching whole
  replays instead of per-pair pickling took throughput 113 → 86
  pairs/s in the wrong direction per-pair, so the message granularity
  is the replay. `tools/supervised_train.py:213-393`.
- **`--batched-forward auto`** **[ON iff device ≠ cpu]** — on CPU,
  padding to max sequence over 1700-hex states regressed 27/s to
  3-6/s. `tools/supervised_train.py:1390-1404`.
- **Periodic held-out eval → a CURVE** **[ON]** — every
  `--eval-every 50000` pairs over `--eval-pairs 1200`, appended to
  `<ckpt_stem>_eval.jsonl`. House convention: measure mid-epoch, not
  only at endpoints. `tools/supervised_train.py:917-943`, `:1695-1712`.
- **Corpus filters** **[ON]** — competitive-2p only (drops ~97% of
  the raw dataset), `--max-replay-commands 1500` (p99 ≈ 1450; four
  outliers at 2000-3300 drove RAM to 65% and caused swap thrashing).
  `tools/supervised_train.py:1188-1205`.
- **Stage profiling `WESNOTH_PROF=1`** **[OFF]** — splits wall-clock
  into `wait` (producer stall) / `encode` (phase-2 on the main
  thread) / `flush` (fwd+bwd+step). Purpose: the CPU-encode vs
  GPU-forward balance readout for hardware selection.
  `tools/supervised_train.py:1374-1386`.
- **Epoch accounting line** **[ON]** — `files_seen / file_errors /
  pairs` per epoch, plus WARNING-level per-file error logs.
  *Why:* an epoch that "completes" with a large error count or far
  fewer pairs than the corpus holds is a broken run, not a fast one
  (2026-08-08 random-arm underrun: "done" at 171k of 2.5M pairs,
  undiagnosable post-hoc). `tools/supervised_train.py:1483-1494`,
  `:1777-1781`.
- **No data augmentation** — no mirroring, rotation, or colour swap
  anywhere in the SL path (verified by grep).

### 4.6 Net2Net growth

`tools/net2net.py`. Used to grow the 471K net to the 5.0M Tier-a net
(`d_model 256, layers 6, heads 8, d_ff 1024`).

- **Leading-block weight transfer.** The trained `[out_old, in_old]`
  matrix lands in the top-left of the wider matrix; new rows/cols
  keep the destination's FRESH random init (symmetry already broken).
  **No noise is added.** `_copy_leading_block`, `tools/net2net.py:57-61`.
- **Q/K/V-aware attention transfer.** `in_proj_weight` is `[3E, ·]`
  stacked as (Q;K;V); the leading dim is split into three blocks
  copied separately "so widening doesn't shear Q's rows into K".
  `_transfer_param`, `tools/net2net.py:64-75`.
- **Honest scope:** an APPROXIMATE warm start, not an exactly
  function-preserving Net2WiderNet (exact preservation through
  LayerNorm + MHA is substantially more involved). The identity case
  (same arch) IS bit-exact, asserted by `test_net2net`.
  `tools/net2net.py:12-19`.
- **Measured quality: value MAE ≈ 0.017** for the 128/4 → 256/8 grow
  — good enough to warm-start the value head. The gate tool
  `tools/measure_warm_start.py` hardcodes both precedents
  (`MAE_ACCEPTED_PRECEDENT = 0.017`,
  `MAE_REJECTED_PRECEDENT = 0.217`) and emits a
  `DROP-IN` / `NEEDS-FINETUNE` / `NOT-A-WARM-START` verdict alongside
  `c51_kl_nats` and `actor_kl_nats`.
  `tools/measure_warm_start.py:66-67`, `:193-196`.
  ⚠ Constraint: the copy is head-aligned **only when
  `d_head = d_model/num_heads` is unchanged** (32 in the accepted
  case). `tools/measure_warm_start.py:22-27`.
- **Refuses to write a mutilated checkpoint.** Transfer walks the
  DESTINATION's params, so a source tensor with no destination slot
  is silently discarded — which is how an optional head disappears.
  `grow_checkpoint` turns any dropped tensor into a hard `SystemExit`,
  and optional-head flags default to "carry from source" rather than
  off. `tools/net2net.py:111-117`, `:177-185`.
- **Vocab + `decision_step` carried across the grow** so transferred
  embedding ROWS stay aligned with their type/faction ids.
  `tools/net2net.py:187-197`.

### 4.7 Human-corpus value work (separate scripts)

- **Frozen-trunk value pre-train** (`tools/value_pretrain.py`) and
  **unfrozen full-trunk value fine-tune** (`tools/value_finetune.py`).
  The frozen experiment cured confident-wrongness but plateaued at
  the floor with weak board reading (late AUC ~0.68) — the frozen
  self-play trunk's features cap it; the unfrozen version tests
  whether learning win-predictive features lifts discrimination.
  Policy heads get no gradient either way (human data has no visit
  counts).

---

## 5. Environment and encoding

### 5.1 Simulator fidelity (the substrate everything rests on)

- **Pure-Python reimplementation of Wesnoth 1.18.4 game logic**,
  ~1000× faster than driving a Wesnoth subprocess.
  `tools/wesnoth_sim.py`.
- **Bit-exact combat**, verified 731/731 strikes against Wesnoth's
  own `[mp_checkup]` oracle on strict-sync replays; full-replay
  reconstruction at 99.93% clean (5,482/5,486).
  Regression check: `tools/diff_replay.py`.
  *Why this matters for learning:* chance nodes (§1.5) sample from
  the TRUE outcome distribution because the sim IS the distribution,
  and the exact-enumeration DP can share parameters with the resolver
  so drift is impossible by construction.
- **Uniform advancement** **[ON, unconditional]** in training games
  (`sim.enable_uniform_advancement()`,
  `tools/sim_self_play.py:806`, `:849`).
- **Engagement telemetry** **[ON]** on the real sim only — MCTS forks
  never carry it, so search pays nothing
  (`tools/sim_self_play.py:318`).

### 5.2 Token streams

- **Five streams → one `d_model`** **[ON]**: hexes, units, recruits,
  a global token, and a learned `end_turn` sentinel.
  `wesnoth_ai/encoder.py:642-788`.
- **Recruit "phantom unit" tokens** **[ON]** — each recruit offer is
  encoded as a would-be unit standing at its leader's keep, sharing
  the unit type embedding, feature projection, and coordinate
  embeddings. *Why:* closes the under-specification gap that drove
  "the model never recruits" — the actor head can now discriminate
  recruits by cost/HP/alignment, not just type+side.
  `wesnoth_ai/encoder.py:734-750`, `:1190-1244`.
  The phantom slot order **is** the recruit action index basis
  (`wesnoth_ai/action_sampler.py:277`).
- **Learned per-axis coordinate embeddings, shared across streams**
  **[ON]** — `pos_x_embed` / `pos_y_embed`, `nn.Embedding(128, d)`.
  Not sinusoidal and not sequence-position: these are board
  coordinates, and the sharing is what lets the model relate "unit at
  (x,y)" to "hex (x,y)". `wesnoth_ai/encoder.py:440-442`.
- **Two-phase encode split** **[ON]** — `encode_raw` is pure
  Python/numpy with a read-only vocab (picklable, worker-side);
  `encode_from_raw` touches learned parameters.
  *Why:* it is the process boundary for parallel rollout AND it lets
  the trainer cache the expensive half once per train_step instead of
  once per pass (~30-50% speedup).
  `wesnoth_ai/encoder.py:297-324`; caching in
  `wesnoth_ai/trainer.py:429-458`.
- **Batched finalization `encode_from_raw_batch`** **[ON]** — 19
  kernel launches per chunk instead of 19×B.
  `wesnoth_ai/encoder.py:790-960`.

### 5.3 Feature detail

- **13-dim unit features** = 9 numerics + 4-way alignment one-hot,
  with both absolute and ratio forms (`max_hp/HP_NORM` AND
  `hp/max_hp`). Order is load-bearing.
  `wesnoth_ai/encoder.py:1325-1343`.
- **Divisor normalization, config-overridable in one place** —
  `HP_NORM 80, MOVES_NORM 10, EXP_NORM 150, COST_NORM 80,
  GOLD_NORM 500, INCOME_NORM 50, VILLAGES_NORM 30, TURN_NORM 60`,
  chosen so default-era values land in ~[0,1]; era mods override in
  `constants.py`. `wesnoth_ai/constants.py:125-144`.
- **Per-hex static modifiers (3)**: owned_village, keep, castle.
  **Per-hex dynamic flags (3)**: `recruit_rejected`, `village_ours`,
  `village_theirs` — in a SEPARATE `dynamic_flag_proj` so old
  checkpoints' 3-input `modifier_proj` loads unchanged.
  `wesnoth_ai/encoder.py:133-174`.
- **`recruit_rejected` bit mirrors the mask's rejection set**
  **[ON]** — the mask consults the set, the model sees the bit; both
  read the same state (the contract in §3.4).
  `wesnoth_ai/encoder.py:1122-1124`.
- **Village ownership sourced from a per-fork `_village_owner` map,
  not a terrain stamp** **[ON]**. *Why — this was the single most
  damaging bug found:* `Hex` objects are ALIASED across MCTS forks,
  so stamping `VILLAGE` on capture let a search-imagined capture
  permanently rewrite the REAL game's encoder input. Since stored
  transitions are re-encoded at train time, a state at turn 3 showed
  ownership for villages captured by turn 30 — **the old encoding
  leaked the future into training inputs** (`fa95da5`, 2026-07-29).
  `wesnoth_ai/encoder.py:1079-1080`, `:1107-1115`.
- **Separate our-faction / their-faction embedding tables** —
  "Drakes as us" ≠ "Drakes as them".
  `wesnoth_ai/encoder.py:452-460`.

### 5.4 Fog of war in the observation

- **Fog hexes RETAINED in the hex stream** **[ON]** — terrain is
  public (the player saw the map at scenario start), and dropping
  them would silently make fog castle hexes ineligible for the
  recruit mask. `wesnoth_ai/encoder.py:1008-1024`.
- **Unit stream fog-filtered** **[ON]** — own units always; enemies
  outside sight discs hidden; ambush/submerge/nightstalk units hidden
  until uncovered. `wesnoth_ai/encoder.py:1137-1158`.
- **Recruit phantoms OWN SIDE ONLY** **[ON]** — emitting them for
  every side was a double fog leak (enemy faction identity + enemy
  keep coordinates). `wesnoth_ai/encoder.py:1190-1235`.
- **Enemy village ownership vision-gated** **[ON]** — your own
  villages show through fog; enemy villages only inside vision. A
  fogged enemy village reads as neutral (a deliberate deviation from
  Wesnoth's stale last-seen display).
  `wesnoth_ai/encoder.py:1125-1135`.
  ⚠ **Known residual:** static modifier bit 0 (`owned_village`) is
  NOT fog-gated (`wesnoth_ai/encoder.py:1115-1117`), so a fogged
  captured village reads "owned by someone" on the static bit while
  the dynamic bits read neutral — a small residual signal. Worth
  confirming against the intent stated at `:155-162`.
- **Neutral side code (2) for petrified units and scenery** —
  encoded as `is_ours=0` even if nominally on our side, matching the
  mask, where they are inert and never actors.
  `wesnoth_ai/encoder.py:1170-1181`.

### 5.5 Relevant-set (subset-of-hexes) encoding — [OFF]

- `relevant_set_hexes = False` (`wesnoth_ai/encoder.py:397-409`;
  `--relevant-set-hexes`). **Default is the WHOLE BOARD.**
  When on, the hex stream is the union of own-unit single-turn reach,
  visible-unit hexes, leader castle network + leader hex, all
  villages, and all castles/keeps. *Measured:* mean 0.30 of the
  board, zero superset violations over 1,840 decisions, ~4.3-4.8×
  rollout forward speedup. `wesnoth_ai/visibility.py:470-534`.
- **Why it is off:** it changes the action space's INDEX BASIS, so
  checkpoints and replay buffers are not interchangeable across the
  flag. Guarded three ways: a `hex_subset` marker making a slot miss
  a hard error rather than a silent action-space shrink
  (`wesnoth_ai/action_sampler.py:1227-1230`), and a spool-ingest
  basis check that drops mismatched games and escalates to
  `SystemExit(6)` on a 2-iteration streak
  (`tools/sim_self_play.py:1310-1371`).
- **Deterministic slot contract** **[ON in both modes]** — both
  derive from the same canonical row-major `(y,x)` sort; the relevant
  set FILTERS rather than re-sorts, so slot indices are reproducible
  when the trainer re-encodes stored states and replays `target_idx`.
  `wesnoth_ai/visibility.py:523-534`.

### 5.6 Vocabulary handling

- **Encoder-owned name→id vocab** **[ON]**, deliberately not the
  state converter's, so a recruit string and an on-board unit of the
  same name hit the same embedding row regardless of discovery order.
  `wesnoth_ai/encoder.py:411-420`.
- **Append-only growth under a process-wide lock** **[ON]** —
  training and inference encoders share the same dict OBJECTS by
  reference, and MCTS leaf expansion registers from worker threads.
  `wesnoth_ai/encoder.py:212-221`, `:503-533`.
  ⚠ **This sharing was the site of a warm-start-corrupting bug:**
  `load_checkpoint` rebinding the dicts orphaned the shared
  references, so after ANY warm start MCTS rollouts ran with an empty
  vocab and scrambled unit-type embeddings (2026-07-02). Fixed
  in-place with a regression test.
- **Unknown types → overflow bucket, silent alias** **[ON]** — the
  201st distinct type aliases onto id 199, treated as a data-quality
  issue rather than an error; surfaced at REGISTRATION time via a
  warn-once log. `_lookup_id`, `wesnoth_ai/encoder.py:971-978`.
- **`freeze_vocab()`** **[OFF, caller-invoked]** and
  **`watch_vocab_growth()`** **[OFF by default, but auto-armed on any
  warm start]** (`tools/sim_self_play.py:3348-3349`) — makes a new
  unit type appearing mid-run a visible breadcrumb.
- **Pre-seeded faction vocab** (`DEFAULT_FACTIONS`) so cross-replay
  supervised training stays consistent. Order matters for checkpoint
  compatibility: appending is safe, reordering is not.
  `wesnoth_ai/constants.py:199-215`.
- **Checkpoint-compat shims** **[ON]** —
  `pad_legacy_encoder_state` (zero-pads `dynamic_flag_proj [d,1]→[d,3]`
  and `side_embed [2,d]→[3,d]`; `strict=False` does NOT tolerate shape
  mismatches) and `repair_optimizer_state_shapes` (pads Adam moments
  to match). `wesnoth_ai/encoder.py:76-131`.

### 5.7 Encoder-level performance caches (all [ON])

`pos_to_hex` map, `recruit_is_ours_np` zero-copy view,
`visible_unit_ids` frozenset (keyed by stable `u.id` so it survives
deep-copied GameStates), `_RECRUIT_STATS_CACHE`, and a lazily
computed vision disc shared between the village fog gate and unit
visibility. Plus pinned-memory H2D transfers on CUDA.
`wesnoth_ai/encoder.py:236-278`, `:666-683`, `:1052-1066`, `:1357`.

---

## 6. Reward shaping

⚠ **All of §6.1 and §6.2 is INERT under `--mcts`** — see the framing
note at the top. It applies to the REINFORCE path only.

### 6.1 Config-driven weighted shaping reward

- **`WeightedReward`: one weighted sum, every term local, all knobs
  in JSON/YAML.** *Design contract:* "editing a dataclass field, not
  touching the trainer". `wesnoth_ai/rewards.py:1-33`, `:302-750`.
- **Signed-contribution convention** **[ON]** — every field is the
  SIGNED contribution per fire; `__call__` is purely additive with no
  per-field sign flipping. `_penalty` suffixes are readability only.
  `wesnoth_ai/rewards.py:306-312`.
- **`StepDelta` is computed by the trainer, not the reward fn**
  **[ON]** — reward functions are pure `StepDelta → float`, so a
  modder writes one function. `wesnoth_ai/rewards.py:51-186`.
- **Config validation at load time** **[ON]** — every scalar key is
  checked against `dataclasses.fields(WeightedReward)`, so a typo'd
  key raises instead of being silently ignored; `_`-prefixed keys are
  inert documentation. `load_reward_config`,
  `wesnoth_ai/rewards.py:1499-1516`.
- **Active terms in `configs/reward_selfplay.json`**:
  `terminal_win +1.0`, `terminal_loss −1.0`,
  `terminal_draw/timeout 0.0`, `gold_killed_delta 0.01`,
  `village_delta 0.05`, `damage_dealt 0.0005`,
  `unit_recruited_cost 0.001`, `per_turn_penalty −0.001`,
  `leader_move_penalty −0.01`, `invalid_action_penalty −0.001`,
  `min_enemy_distance_penalty 0.0`.
  *Why gold-killed is the main dense term:* it is a better proxy than
  raw HP because it prices in hitpoints, resistances, and trait rolls.
- **Default-OFF shaping terms** (present, weight 0):
  `approach_enemy_leader_per_mp` (terrain-aware Dijkstra closing —
  and a **deliberate documented god-view breach**, inert at weight 0,
  `wesnoth_ai/rewards.py:465-467`), `unused_mp_penalty`,
  `fog_reveal_weight`, `attack_attempt_bonus`,
  `gold_killed_one_sided`, and both bonus lists.
- **Unit-type and turn-conditional bonuses with a predicate
  registry** **[available, empty by default]** — `register_predicate`
  / `get_predicate`, built-ins `leader_on_village`, `leader_on_keep`,
  `controls_majority_villages`, `no_units_lost`; turn-conditional
  bonuses support `once`-per-game gating. This is the main
  "incentivize an unorthodox strategy without retraining" seam.
  `wesnoth_ai/rewards.py:198-298`, `:1317-1400`.
- **Per-component reward telemetry** **[ON when the accumulator is
  attached]** — 15 stable `r_*` CSV columns.
  `wesnoth_ai/rewards.py:542-555`.

### 6.2 Scripted openers

- **`OpenerPolicy` wrapper** **[OFF]** (`--opener-spec`,
  `tools/sim_self_play.py:2592`). A scripted opener delegates to the
  learned policy once it finishes. *Design goal:* forcing a specific
  opener (rush, defensive, village-grab) is a config flip, not a
  retrained model. `tools/openers.py:1-20`.
  Incompatible with the actor pool and spool workers.

### 6.3 Draw tiebreak — the one "shaping" signal that IS live under MCTS

- **Material tiebreak at search turn-cap terminals** **[ON]** —
  `--draw-tiebreak-cap 0.3` (`tools/sim_self_play.py:3044`);
  a draw scores `cap·tanh((w_v·ΔV_frac + w_g·Δgold + w_u·ΔunitValue)
  / score_scale)` instead of a flat 0.
  *Why:* while the policy is too weak to ever kill a leader, 100% of
  games end as turn-cap draws, every z is exactly 0, the value head
  gets no gradient at all, and PUCT is starved of meaningful Q. The
  SAME function scores the search terminal and (optionally) the
  trainer's z, so what the search optimizes at the horizon is exactly
  what the head learns. `tools/draw_tiebreak.py:1-43`;
  `_terminal_value`, `tools/mcts.py:680`.
- **`cap = 0.3`** keeps the best possible draw far below a real win
  (+1) — the search must always prefer a leader kill.
- **`weight_gold = 0.0`** **[ON since 2026-07-20]** — MP Wesnoth has
  no gold carryover, so end-of-game gold is worthless in the real
  game, and pricing it at par with units taught the prior 2.8×
  hoarding vs the SL baseline. Only units score, so converting gold
  into army is strictly rewarded.
- **Village weight normalized to the MAP FRACTION** — village counts
  vary ~10-30 per ladder map, so a per-village term made one village
  worth 3× more on a small map. `configs/draw_tiebreak.json`;
  derivation in `docs/design_constants.md`.
  ⚠ `configs/draw_tiebreak.json` is authoritative for the current
  weights (`weight_village 2.0`, `weight_unit_value 0.016667`,
  `score_scale 5.0`) — `docs/design_constants.md` still quotes the
  pre-2026-07-21 calibration.
- **Training z uses honest 0, not the tiebreak** — see §2.3.

---

## 7. Infrastructure that affects learning

### 7.1 Experience replay

- **Bounded FIFO replay + multi-epoch minibatch updates** **[OFF]**
  (`ReplayConfig.enabled = False`, `tools/mcts_policy.py:86`;
  `--replay-buffer`). `capacity 4000`, `updates_per_iter 8`,
  `minibatch 128`, `min_size 512` (warm-up on the legacy one-pass
  until the buffer fills).
  *Why it exists — measured:* the default one-gradient-step-per-fresh-
  batch-then-discard schedule is severely sample-inefficient. Overfit
  probes showed the value head needs ~80-100 gradient steps to
  converge on a batch, but live training gave it ONE step per
  shifting, high-variance batch → the value head never left its
  ~uniform floor (val loss ~3.56 vs ln 51 = 3.93), MCTS produced
  near-uniform visit targets, and the policy plateaued.
  `tools/mcts_policy.py:68-91`, `train_step` at `:847-918`.
  Sampling uses a dedicated seeded RNG so replay runs are
  reproducible. MCTS-only.

### 7.2 Anti-forgetting rehearsal

- **Human-corpus value anchor** **[OFF]** (`--human-anchor-file`,
  `tools/sim_self_play.py:3028`; `updates 4`, `batch 128`).
  Value-only gradient steps on pre-encoded human states with clean ±1
  labels, run BEFORE `train_step` so its inference-weight sync
  captures them.
  *Why:* self-play alone eroded human-corpus late-game AUC 0.88 →
  0.60 in ~80 iterations. `tools/sim_self_play.py:1860-1888`;
  cache built by `tools/build_human_anchor.py` (checkpoint-independent
  — `RawEncoded` depends only on the frozen vocab).

### 7.3 Curriculum and scenario randomization

- **Absolute-fraction training mix, one categorical roll per game**
  **[default = 100% fogged ladder]** —
  `--ladder-ratio 1.0`, `--mini-ratio 0.0`, `--drill-ratio 0.0`,
  `--midgame-ratio 0.0`, `--fogless-ratio 0.0`, validated to sum to 1
  at startup. *Why absolute (2026-07-20 redesign):* cascading
  remainders made the realized mix unreadable.
  `tools/sim_self_play.py:2696-2724`, `roll_mix` at `:895`.
  **Every curriculum component is therefore default-OFF.**
- **Scenario + faction randomization** **[ON]** — random Ladder-Era
  map, random faction + leader per side, fresh GameState from the
  scenario WML with prestart events fired.
  *Why not replay-as-seed:* replays carry idiosyncratic starting
  states, and self-play wants a clean canonical start with random
  matchups so the policy doesn't overfit to popular pairings.
  `tools/scenario_pool.py:1-27`.
- **Forced faction** **[ON — Knalgan Alliance]** ⚠ surprising
  default. `FORCED_FACTION = "Knalgan Alliance"`
  (`tools/scenario_pool.py:360`, user request 2026-04-30): every
  game has at least one Knalgan side; the other samples uniformly, so
  mirrors still occur ~16.7% of the time. Disable with
  `--forced-faction none`; `--forced-faction <name>` locks another.
- **Mini-map curriculum** **[OFF]** (`--mini-maps`) — the 5 smallest
  ladder maps, leaders ~12-15 hexes apart, so the policy can discover
  engagement before the long-march cost dominates.
- **Backward curriculum from human mid-game positions** **[OFF]**
  (`--midgame-ratio`). A fraction of games starts from a human game's
  position at a uniform-random turn and is played out by self-play.
  *Why:* the stalemate is an equilibrium of self-play at current
  skill — fresh ladder games never reach contact, so decisive
  terminals never occur. Rather than relabel draws (which distorts
  the objective), change WHICH STATES get experienced; the learnable
  frontier then walks backward from contact toward the opening.
  `tools/midgame_starts.py:1-20`.
- **Fogless mixing** **[OFF]** (`--fogless-ratio`) — full-information
  games give the value head mutually-visible armies.
- **Turn-cap jitter** **[OFF]** (`--max-turns-min`, default `None` =
  fixed `--max-turns 200`). *Why it exists (user directive
  2026-07-20):* a FIXED cap let the policy learn to bank gold until a
  known last turn. Training paths only; eval/demo keep fixed caps.
  `_roll_max_turns`, `tools/sim_self_play.py:1020-1029`.
- **No-progress (stalemate) rule** **[OFF]** (`--no-progress-turns 0`)
  — but the tracker RECORDS would-fire statistics on every game, so
  candidate K values can be evaluated offline before enforcement
  (2026-07-21). `tools/sim_self_play.py:2474`, `:632`.
- **PvP economy normalization** **[ON]** (`use_map_settings=True`;
  starting gold 100, village gold 2, support 1, exp modifier 70).
  ⚠ `starting_gold` is deliberately NOT mapped through since
  2026-07-21 — scenario `[side] gold=` is ground truth, and the PvP
  default was silently overriding minis/drills/Arcanclave.
  `tools/sim_self_play.py:826-833`.
- **Random start-ToD on fixed-ToD mini templates** **[OFF]**
  (`--mini-random-tod`, implemented as `WESNOTH_MINI_RANDOM_TOD=1` so
  spool workers inherit it). A de-confound lever.

### 7.4 Parallelism (all three modes default OFF)

- **Thread workers** **[OFF]** (`--workers 0`). Safe via the policy's
  snapshot+lock design; each worker gets its own RNG seeded from the
  master so games stay deterministic given the seed.
- **Actor pool (SEED-RL / MonoBeast)** **[OFF → tier-b production,
  user ruling F3 2026-08-10]** (`--actor-pool 0`).
  N weightless actor processes; every leaf forward ships as a
  `RawEncoded` to a central batching server in the main process, so
  there is NO weight sync. `tools/actor_pool.py:1-19`.
  ⚠ The "measured losing design" verdict (~200 req/s cap, GPU idle)
  was TIER-A-SPECIFIC: at 5M params the spool workers' CPU forwards
  were cheap enough to beat the central server. At tier-b (15M) the
  arithmetic flips — required leaf throughput at 4-7k steps/hr ×
  ≤32 sims is 20-62 req/s, well under the measured 200 req/s
  ceiling, while the spool projects to ~2k steps/hr on 15M CPU
  forwards. Activated for tier-b without a fresh A/B (user accepted
  the arithmetic). Caveat: cross-actor dynamic batching breaks
  bit-determinism of training.
- **Spool workers** **[OFF]** (`--spool-workers 0`) — **the measured
  winner.** N independent processes each play whole games with their
  own in-process GPU forwards and atomically spool one pickle per
  finished game; the learner consumes them and saves checkpoints the
  workers hot-reload on mtime change. Saturated a 4090 at 99%.
  `tools/selfplay_worker.py:1-20`; ingest `tools/sim_self_play.py:1080-1379`.
  Warning: training is no longer bit-deterministic under the actor
  pool (dynamic cross-actor batching).
- **VRAM-budgeted device split + reactive demotion** **[ON when spool
  is used]** — `SPOOL_WORKER_VRAM_BYTES = 640 MiB`,
  `TRAINER_VRAM_RESERVE_BYTES = 15 GiB`; a one-way per-iteration
  demotion ratchet driven by the iteration's true backward peak, plus
  an OOM emergency retry (empty_cache → demote one worker → retry
  once). *Why:* the 2026-07-18 incident — 56 auto-CUDA workers on a
  24GB 4090 left the trainer 318MB short and crash-looped it through
  3 OOM deaths; the learner's backward peak GROWS with play quality
  (7.1 → 12.6 GiB across two days at unchanged settings).
  `tools/sim_self_play.py:936-945`, `:1245-1277`, `:1892-1910`;
  derivation in `docs/design_constants.md`.
- **Device-aware `train_batch_size`** — `1` in `TrainerConfig`
  (`wesnoth_ai/trainer.py:226`), **128 on CUDA at the CLI**
  (`tools/sim_self_play.py:3284-3292`). *Why:* on CPU, batching the
  transformer ran 1.7-2.8× SLOWER at ~1600-hex sequences (padded
  activations spill past L2/L3); on GPU it is THE key knob, so the
  replay minibatch forwards as one batched call instead of 128
  batch-1 calls.
- **CPU thread cap** **[ON, CPU only]** — auto-capped to 4 threads
  (measured ~1.3-2.3× on CPU); not capped on GPU devices.
  `tools/sim_self_play.py:3129-3139`.
- **Module-identity canonicalization** **[ON]** —
  `sys.modules.setdefault("tools.sim_self_play", ...)` before any
  spool payload unpickles. *Why:* nine `tools/` modules were being
  imported both as `tools.X` and bare `X`, producing two module
  objects with duplicated module state — including a bug that
  **cannot exist single-flavour** (`8b68a25`).
  `tools/sim_self_play.py:3896-3913`.

### 7.5 Metrics, tripwires and run control

- **Trainer-history CSV** **[ON]** — ~140 stable columns, written
  line-buffered so a walltime kill leaves a recoverable file, with a
  header-mismatch guard that rotates a stale file to `.oldschema`
  (exactly what hid the holdout column on the 2026-07-06 Vast run).
  Default path `training/logs/trainer_history_local.csv`.
  `_TrainerHistoryCSV`, `tools/sim_self_play.py:2173-2388`.
- **Per-net-size metric archives** — `training/metrics/history_5m.csv`
  and `history_15m.csv`, schema `net, run, <trainer_history columns>`.
  ⚠ **This is a standing HUMAN rule, not code** (user order
  2026-08-06): nothing tags rows with `net`/`run`; the archives are
  appended by hand at each harvest, and the CSVs are gitignored.
  *Why the rule exists:* the 15M "benign plateau" was only exposed as
  a floor-relative regression by comparison against the 5M campaigns.
  `training/metrics/README.md`.
- **Per-game JSONL telemetry** **[ON]** — one directory per
  iteration, with a claimed run-id because the in-process iteration
  counter restarts at 0 on every relaunch.
  `tools/sim_self_play.py:1691-1729`, `:951-969`.
- **Diagnostic breakdowns** **[ON]** — closest-approach to the enemy
  leader ("the headline no-kills metric"), action-type histogram,
  per-map-class decisive split, fog vs fogless split, actions per
  side-turn, `search_q_spread` / `search_overturn_frac`, ~25 `eng_*`
  engagement columns (two of them zero-expected tripwires), and
  distillation-target telemetry (`distill_sharpen_top` is the
  prior-ratchet gauge). *Why the per-class split:* the AGGREGATE
  decisive rate is misleading over a mixed curriculum — 2026-07-03
  showed ~50% aggregate with 0/8 decisive on ladder maps.
  `tools/sim_self_play.py:1596-1856`, `:2045-2064`.
- **All-draws abort tripwire** **[OFF]** (`--abort-decisive-rate`,
  window 20) — saves a checkpoint and exits code 4. Guards paid GPU
  runs against the known all-draws failure shape.
- **Memorization / holdout-stall tripwire** **[OFF]**
  (`--abort-holdout-stall`, min-delta 0.01) — exits code 5.
  *Why:* the 2026-07-02 Kaggle run had train value loss fall
  3.8 → 1.15 while holdout CE sat flat at ~3.1 — pure buffer
  memorization.
- **Dead-iteration guard** **[ON, hardcoded 5]** — exits code 3.
  Observed 2026-05-09: a job "completed" 999 iterations of 0 games.
- **Atomic checkpoint save + rolling `.bak`** **[ON]** —
  `torch.save` → `.tmp`, `os.replace(path, path.bak)`,
  `os.replace(tmp, path)`; the loader falls back to `.bak` and
  rewrites the arg so the arch peek and the weight load agree.
  *Why:* on rented SPOT GPUs a kill mid-`torch.save` would otherwise
  truncate the ONLY checkpoint and lose the whole paid run; a
  preemption now costs at most the last save interval.
  `wesnoth_ai/transformer_policy.py:612-652`;
  consumer `tools/sim_self_play.py:3175-3217`.
- **Arch mismatch on load = start fresh, not fatal**, so chain links
  survive model-size changes (`tools/sim_self_play.py:3315-3328`).
- **Wall-clock `--time-budget`** **[OFF]**, checked AFTER each
  iteration so no rollout work is wasted; and a **graceful-cancel
  sentinel** **[ON]** at a fixed path, cleared at startup and polled
  between iterations (saves, unlinks, breaks).
- **Validation replay exports** **[ON, every 100th game per
  category]** (`--validate-export-every 100`) — strict-sync
  Wesnoth-loadable exports, tarred per iteration because ~2k
  replays/day as loose files blows HF's per-folder guidance within
  weeks. `tools/validation_exports.py`;
  `tools/sim_self_play.py:974-1017`.
- **Recruit-bounce retry loop** **[ON]** — when the model picks a
  fog-hidden occupied castle hex, the decision is re-taken without
  consuming the turn; the hex is blacklisted for the turn, and the
  bounced pick is un-recorded (`drop_last_pending`) so it is not
  trained on with terminal z and does not over-advance the
  decision-step counter. The mask shrinks monotonically, so no K-cap
  is needed. Move targets are deliberately NOT pre-bounced since
  2026-07-17 — the sim resolves them Wesnoth-faithfully.
  `tools/sim_self_play.py:380-414`.

### 7.6 Strength measurement (outside the training loop)

- **Whole-History Rating (Coulom 2008)** — `tools/whr.py`. Fits a
  time-varying strength curve jointly over the entire training
  history with a Brownian-motion prior tying consecutive
  checkpoints, so a checkpoint that played few games borrows strength
  from its neighbors. *Why over static Bradley-Terry:* one coherent
  strength-vs-time curve, no moving window, no per-window gauge
  drift.
- **Static Bradley-Terry Elo ladder** — `tools/elo_ladder.py`,
  round-robin over the simulator; one jointly-fit number per player
  with a principled standard error.
- **External baseline** — `tools/eval_vs_builtin.py` +
  `tools/eval_runner.py`, the only remaining live-Wesnoth consumer,
  pitting the model against the built-in RCA AI over a
  (map × matchup × side-swap) matrix.
- ⚠ **Never merge the two numbers.** The in-lineage Elo is measured
  WITH search; the RCA eval is RAW policy. They are different
  objects. See CLAUDE.md §Current status.
- **No opponent pool, no league, no snapshot opponents in the
  training loop.** Both sides always share the current policy
  (`tools/sim_self_play.py:1582-1585`). This is a genuine gap, not
  an omission from this document.

---

## Appendix: known-stale or dead items encountered while writing this

Listed so the next reader does not re-derive them. None of these
change behavior today.

- `configs/imitation.json` `value_from_outcome_weight` is a **dead
  key** — nothing reads it; value supervision is controlled solely by
  `--value-loss-weight`. Editing it to `0.0` would NOT disable value
  supervision, contrary to the file's own `_doc`.
- `configs/imitation.json` `_doc` names a flag `--imitation-manifest`
  that does not exist; the real flag is `--imitation-config`.
- `--max-pairs-per-replay` is silently ignored when `--workers > 0`
  (`tools/supervised_train.py:236-238`). Harmless at the default 0.
- `--eval-only` never computes value AUC (it omits `winner_map=`).
- SL held-out eval is neither winners-only nor per-game weighted, so
  in imitation mode the eval CE measures a different objective than
  the one being trained; it also includes actor-head action-type
  weighting while the training log's per-head numbers are raw.
- `--replay-pool` is parsed but ignored — `main()` hard-sets
  `pool_files = None` (`tools/sim_self_play.py:3141-3146`).
- `--holdout-size` help says "not persisted across resumes";
  persistence was added at `tools/sim_self_play.py:3489-3492`.
- `--mcts-advice` help says the trainer re-forward is a follow-up; it
  is implemented (`wesnoth_ai/trainer.py:1098-1116`). The channel is
  separately refuted on evidence (§3.5).
- `wesnoth_ai/combat_oracle.py:22-23` still says the sampler scales
  by "default 0.1"; the alphas have been 0.0 since 2026-07-16.
- `wesnoth_ai/encoder.py:170-174` says keeping
  `NUM_HEX_DYNAMIC_FLAGS = 1` preserves compatibility, directly above
  `NUM_HEX_DYNAMIC_FLAGS = 3`.
- `wesnoth_ai/action_sampler.py:963-969` `_enemy_unit_at` is dead
  code (no callers anywhere).
- `docs/design_constants.md` quotes `weight_village = 10.0` for the
  material margin; `configs/draw_tiebreak.json` is authoritative at
  `2.0` after the 2026-07-21 de-saturation recalibration.
- CLAUDE.md's superseded 2026-06-11 block claims self-play
  auto-warm-starts from the highest `supervised_epoch*.pt`; no such
  glob exists — `--checkpoint-in` defaults to `None`.
