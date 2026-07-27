# Detector → trainable signal — approved design (2026-07-24)

Turn the swap detector's findings into a learning signal the model can USE
but also **learn to ignore** where it deviates deliberately (experience
management, baiting, a bigger positional play). The naive version — reward
the model for matching the detector — is exactly what would cripple a
strong model, so we do not do that.

## Principle: the detector PROPOSES, the value net JUDGES

The detector is cheap and exhaustive at finding *locally* dominant
reorderings under a *fixed* valuation; the true objective (winning) values
things the detector's dimensions don't (long-horizon XP payoff, tempo,
information, board control beyond the (pos,MP) criterion, baiting). So the
detector says "here's a candidate you didn't try"; the model's OWN value
network says "…and here's whether it actually helps *me*." That single
choice makes "learn to ignore" intrinsic, not bolted on.

### Self-calibration property (why this is safe)

Verdict tier already correlates with safety:

- **Tier-1 product-order certificates** (`backstab_setup`,
  `leadership_setup`) dominate on *every* tracked dimension — and XP is a
  tracked dimension — so a reorder that traded XP away is never Tier-1. Safe
  to weight heavily.
- **Banking opportunities** (`attacks_before_commit`,
  `strong_attacker_first`) explicitly trade off (MP/position vs XP
  allocation) → already flagged product-*incomparable*. Deliberate
  exp-management lives precisely in this weak-signal tier.

So the signal's strongest form is also its safest form. Tier maps to
coupling strength.

Caveat: even Tier-1 is not dominant on UN-tracked dimensions (information,
tempo, baiting), so even Tier-1 stays overridable — just with a higher
default prior.

## Approved fork decisions

1. **Channel** — input-feature + ΔV-weighted **distillation target**; NO new
   reward term. (Alternative: potential-based reward shaping with a learned
   potential — deferred, see BACKLOG.)
2. **Arbiter** — cheap: the model's value net evaluated on the
   *reconstructed* end-state distribution. (Upgrade: a short MCTS from the
   reordered position — deferred until ΔV proves noisy.)
3. **MVP scope** — Tier-1 certificates only first; banking tier once the
   gate is trained.

## Signal schema (what / where / proposal)

Each `Finding` becomes a structured, board-localized record:

- **what** — `motif` id (embedding) + `verdict` tier + guaranteed-gain
  vector (the per-dimension deltas the detector claims);
- **where** — pointers into the model's own unit/hex token sequence
  (attacker, mover/flanker, target), anchoring the advice on the board the
  model already sees;
- **proposed instead** — the reorder as a delta on the action sequence: "at
  this decision, do M (move flanker) rather than A (attack)."

## Pipeline: Propose → Dispose → Distill

```
played side-turn ─► detector.generators ─► findings {motif, board-refs, reorder, tier, gain}
                                                  │
                  ┌───────────────────────────────┼──────────────────────────────┐
                  ▼                                ▼                               ▼
        (1) advice tokens in encoder     (2) reconstruct BOTH orderings'   (3) tier+gain ─► prior-boost
            — the model SEES it              end-state dists, score each        weight, ANNEALED like the
                  │                          with the model's OWN value         existing combat oracle
                  │                          net ─► ΔV                          │
                  └──────────►  policy-distillation target  ◄──────────────────┘
                               weighted by  gate(state, finding) · max(0, ΔV)
                                                  │
                              policy/value trained ONLY toward winning
                              ⇒ "ignore when ΔV ≤ 0" is automatic
```

Per played side-turn in self-play: run the cheap DP-based generators; for
findings above a gain threshold, use `reconstruct_side_turn_dist` to get
both orderings' end-state distributions and score them with the current
value net → ΔV. Where ΔV>0 by the model's *own* judgement, add the proposed
action as an extra soft target at the divergence decision, weighted by the
gate. Where ΔV≤0, no push. **Anneal from "trust the detector's fixed
valuation" → "trust the value net"** over training — the pattern the
combat-oracle attack-bias already uses and `--reset-decision-step` already
manages.

## Revised coupling: a learnable scale on the ACTING path (user, 2026-07-25)

The MVP's "sign(ΔV) gate + fixed per-tier strength" was rejected: it throws
away ΔV's magnitude (offline validation saw +0.0107 vs +0.0001 -- a real
confidence difference). The replacement is a LEARNABLE, magnitude-aware
scale. Where it lives is forced by one fact:

- **A learnable multiplier on the distillation target is self-defeating.**
  If `weight = λ·max(0,ΔV)` blends the proposal into the visit-count target
  and λ is trained by the distillation loss `CE(policy, π')`, then
  `∂loss/∂λ` moves λ to make π' match the CURRENT policy. When the policy
  doesn't already favour the proposal (exactly when the advice is novel),
  `log p_proposed` is very negative, so the gradient drives **λ → 0**: the
  scale learns to switch the advice off. So the scale's gradient must come
  from the TRUE objective (winning), not the imitation loss -- i.e. it must
  sit on the ACTING path, not the training-target path.

So the coupling is:

1. **Prospective advice tokens (decision-time, acting path).** A cheap
   pre-filter detects setup opportunities among the AVAILABLE actions
   (e.g. a backstab-weapon unit adjacent to an enemy with an available
   flanker move to the opposite hex). Each becomes an encoder ADVICE TOKEN
   carrying {motif, tier, guaranteed-gain features, ref to the setup move,
   optional prospective ΔV}. The model attends to it.
2. **A gate head** emits `s = softplus(gate(state, advice_features))`; the
   advice token's contribution to the trunk is `s`-scaled. The policy
   learns its response -- including how to weight the magnitude -- from the
   TRUE reward. Learnable, magnitude-aware, non-circular. This IS the
   learnable scaling factor, and it unifies scaling with "learn to ignore":
   the gate shrinks toward 0 exactly where deviating wins (exp management).
3. **Zero-init graft.** The gate/advice output projection inits to zero, so
   the advice contributes nothing at load: `load_state_dict(strict=False)`
   fills the rest from an existing checkpoint (tier_a loads cleanly) and
   the model LEARNS the scale up from zero. No checkpoint is invalidated.
4. **ΔV stays as a retrospective exploration SEED** (optional, later): a
   small annealed distillation push toward the reordered action so it
   appears in the training data for the gate to learn from. This push is
   fixed/annealed (NOT the learnable scale) -- it only seeds exploration.

MVP order: model-side plumbing (advice tokens + gate head + zero-init,
config-gated OFF) first; prospective advisor + self-play wiring next.
Prospective ΔV in the token (needs decision-time reconstruction + value) is
an enhancement over gain-vector-only tokens.

## Why "learnable to ignore" holds — three independent mechanisms

1. **Value-net-as-judge (ΔV):** a stronger value net → better ΔV → a
   stronger model automatically discounts bad proposals. Anchored to real
   game outcomes, so it can't trivially be gamed.
2. **Advice-as-input + true-reward-only:** findings are also encoder
   features and the only gradient is winning, so the policy learns
   (state, advice) → best action, which includes "ignore the advice here."
3. **A learnable gate** `gate(state, finding) ∈ [0,1]`, trained on realized
   outcomes, makes trust explicit and *readable* (log: "deviated from
   backstab-setup; gate=0.1, ΔV=−0.04 ⇒ setting up a bigger play"). Direct
   hit on the "study its strategies" goal.

## Where it plugs into the code

- `wesnoth_ai/encoder.py` — advice tokens, same shape as the recruit-phantom
  tokens already there.
- `wesnoth_ai/model.py` — advice-attention path + optional gate head beside
  the C51 value head.
- `wesnoth_ai/action_sampler.py` — annealed prior boost beside the existing
  combat-oracle attack-bias.
- `wesnoth_ai/trainer.py` — ΔV-weighted proposal as an extra distillation
  target in `step_mcts` (no new reward term for the MVP).
- `tools/swap_detector.py` — prospective advisor + ΔV scorer (reuses this
  session's `reconstruct_side_turn_dist` / `compare_state_distributions`).
- `configs/` — anneal schedule, gain threshold, tier weights (modder-
  flippable, per the "config over weights" principle).

## MVP (phase 1)

1. **Prospective advisor** in `swap_detector.py`: given (state, committed
   actions this turn, available actions), return Tier-1 findings as the
   structured signal above. (The current generators are retrospective over a
   recorded side-turn; the advisor runs them over committed+available.)
2. **ΔV scorer**: value net over the reconstructed end-state distribution of
   played vs proposed.
3. **Encoder advice tokens** + **trainer distillation target** weighted by
   `max(0, ΔV)` (learned gate deferred), config-gated + annealed.
4. **Readability trace**: per-game log of findings / ΔV / followed.

## Offline validation results (2026-07-25, tier_a_campaign_final)

`tools/validate_advisor.py` over the 19 HF ladder games with the tier_a
value net. Findings that shaped the design:

- **Full-turn reconstruction has ~0 coverage.** All 10 Tier-1 findings
  scored `delta_v = None` -- the joint over every combat in a real
  side-turn blows up. Fixed by WINDOW reconstruction (see
  `delta_v_for_finding`, `window=True`): reconstruct only
  [min(attack,move)..max], conditioning on the recorded prefix.
- **Backstab certificates judge dv > 0.** With windowing the three
  single-combat backstab windows judged +0.0001 / +0.0015 / +0.0107 -- the
  value net AGREES with the product-order certificate, as expected. No
  negatives in this (certificate-tier) sample; the "ignore" case (dv<=0)
  is expected to surface in the banking tier (XP trades), which is where
  deliberate exp-management lives.
- **delta_v magnitudes are small** (1e-4 .. 1e-2 win-prob): one reorder
  barely moves the game. So the distillation weight CANNOT be raw
  `max(0, delta_v)` -- it needs a **scale/temperature** (or use
  `sign(delta_v)` as a gate and the detector's guaranteed-gain magnitude
  as the weight). A config knob, tracked in BACKLOG.
- **Leadership windows still bail** (2/2): both were two-attackers-on-one-
  hex turns, so the window spans two combats and the joint blows up.
  Window TRIMMING (drop combats not involving the reorder's units -- they
  are independent, apply them deterministically like the prefix) is the
  fix. Tracked in BACKLOG.

## Build status + local iteration (2026-07-25)

**Acting side COMPLETE + validated + profiled.** Wired end-to-end behind
`--mcts-advice` (OFF by default): model advice path (gated learnable scale,
zero-init graft) -> advice-token builder -> prospective advisor (decision-
time) -> MCTS root `_expand(advice=)` -> config/CLI/checkpoint round-trip.

- **Profiling** (micro-benchmark, 256-d/6-layer): the prospective advisor
  adds **~0 ms** on a plain state (no backstab-weapon unit -> the pre-check
  is free) and **~3 ms** on a state with a setup (the DP verification).
  Root-only + rare -> negligible next to a decision's 16-200 leaf forwards.
- **Validation** (short MCTS self-play, warm-start tier_a_campaign_final):
  the graft loaded with 12 missing `advice_*` keys / 0 unexpected (clean),
  advice path ON, a full game rolled + a train_step ran + the checkpoint
  saved. No crashes.

**Learning side DONE -- the gate learns.** The trainer's MCTS policy-loss
reforward rebuilds each sample's prospective advice tokens (grad ON) from
its stored `game_state` and attaches them, so the advice cross-attn gets a
policy-loss gradient: `advice_out` bootstraps first (its grad
= gate*attn_out is non-zero even at zero-init), the gate follows once
`advice_out != 0`. Deterministic from the state (same advisor as acting)
and re-resolved against the reforward's own encoding frame, so grounding
indices are self-consistent (no reliance on unstable `gs.map.units` set
order). `has_advice`-gated -> advice-free training is byte-unchanged.
Validated: `advice_out` bootstraps (non-zero grad from zero-init) and the
full reforward path reaches the advice params (tests); an end-to-end local
MCTS iteration ran a train_step without crashing.

**Follow-ups (not blockers):**
- **Batched advice in `forward_batch`.** For B>1, chunks that carry advice
  fall back to per-sample forward (correct, just slower). Advice is rare so
  most chunks stay batched; a padded-advice + key-mask kernel is the perf
  optimization for advice-dense CUDA training.
- **Full slow tier before a real advice campaign.** The fast tier is green
  (563); the slow e2e tier hasn't been run clean-through (the dev machine
  kept sleeping mid-run). Run `pytest -m ""` before launching an advice
  training run.

## Risks / open questions

- **Off-distribution value-net eval** on reordered states it wasn't trained
  on → mitigation: train value net on some reordered states, or lean on
  MCTS-arbitration (deferred).
- **Reconstruction bails** (advancement past cap / blow-up) → ΔV
  uncomputable → fall back to advice-token-only (no distillation push);
  track coverage.
- **Self-play cost** of the prospective detector + ΔV evals → gain-
  threshold, cache, batch value-net evals.
- **Circular eval** (value net judging its own proposal) → anchored to real
  returns + anneal; watch for feedback loops.

Deferred improvement items are tracked in BACKLOG.md.
