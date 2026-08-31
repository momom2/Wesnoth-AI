# signal_profiler — gradient-amplitude tree (v1, 2026-08-31)

Standalone instrumentation (user order): decompose the training
signal from played games down to the gradient, as a TREE of
amplitudes with attribution per part of the training system.
Correlational at the Elo hop (v1 contract); everything upstream of
the optimizer is exact.

## What it measures

One profiling round = play N games through the PRODUCTION pipeline
(TurnCommitPolicy, the arm-V3 config by default), drain the
experience batch, then re-run the production `trainer.step_mcts`
once per ISOLATED loss term with the optimizer stubbed to no-op and
read the accumulated post-clip gradients. Isolation uses the
pipeline's own per-experience channels (policy_weight /
value_weight / aux_target / gbc_labels / moves_left_target), so no
new loss code runs — the tests-drive-real-code rule.

Fresh-Adam parameter deltas are deliberately NOT the metric: on a
cold optimizer the bias correction makes step size nearly
gradient-scale-invariant, which would erase exactly the amplitude
information this tool exists to expose.

## The tree

    total (all terms, one production step)
    ├─ policy_distill      ── per parameter group
    ├─ value_inbatch       ── per parameter group
    ├─ gbc                 ── per parameter group
    ├─ aux_margin          ── per parameter group
    ├─ value_memory        (head-only step, own path)
    └─ [anchor_policy]     (when --anchor-file given)

Parameter groups: encoder / trunk / actor_head / type_head /
target_proj / weapon_head / value_head / gbc_heads / aux+ml heads.

Node fields:
  norm      — L2 of that term's (clipped) gradient over the group;
  frac      — norm / total-node norm (same group);
  cos_total — cosine vs the all-terms gradient (same group):
              NEGATIVE = this term FIGHTS the combined update there;
  proj_frac — signed projection onto the total, / |total|²: the
              share of the realized update this term is responsible
              for (sums to ~1 across terms, unlike norms).

## Run

    python signal_profiler/run_profile.py \
        --checkpoint training/checkpoints/seed_imit_tierb_start.pt \
        --games 8 --out profile.json
    # arm-V3 config is the default; --no-turn-search etc. available.

Smoke test (tiny net, CPU): pytest signal_profiler/tests -q
(deliberately outside the main suite's testpaths).
