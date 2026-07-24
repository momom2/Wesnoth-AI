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
