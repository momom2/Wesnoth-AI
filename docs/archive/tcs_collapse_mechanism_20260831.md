# The K-collapse mechanism (2026-08-31, user order: "probe TCS's
# accept/reject stream and find the collapse mechanism")

## Verdict

**Boundary-only grading cannot price tempo.** Ending the turn early
costs nothing AT the boundary, because the boundary never shows the
opponent's free reply. Whenever a value head's per-action deltas dip
(noise, drift, or honest indifference), TCS's gate — working as
designed — accepts an end_turn alternative as an "improvement", and
committed turns shrink. The evaluation frame only modulates how
loudly this expresses; the disease is in WHAT is graded, not how.

The antidote already exists in-tree: multi-turn projection
(TurnSearchConfig.project, built 2026-08-17 after leg-3 for exactly
this, default OFF ever since). With it on, "stop early" pays the
reply, truncation dies, and search stays discriminating.

## The evidence chain (all runs: 40 identical dummy-harvested
## midstates, per-gate TRACE streams; artifacts in
## eval_games/tcs_collapse_probe/)

Checkpoints: seed (+223 board), armV1/armV2 (value-memory arms that
K-collapsed in ~5/~3 iterations in vivo), armT pin 2,922,263 (the
-478 oscillation pin, K-healthy in vivo).

1. **Mover frame, no projection** (the leg-5+ production setting):
   end_turn alternatives WIN accepted gates — seed 30, armV1 33 of
   ~40 plans; committed medians 1 and 3 vs spine 5. NOT
   accept-starvation (accept rates 0.63-0.75), NOT short spines
   (equal at ~5.1). Truncation is graded as improvement.
2. **Ruled out on the way:** the combat-oracle anneal (both alphas
   are 0 at every decision_step — dead code in this era); a simple
   "quit while ahead" value-sign law (seed truncates when ahead,
   armV1 when BEHIND, armV2 barely at all in vitro); and the
   unspent-MP visibility story (neutralizing MP/acted flags at the
   boundary made truncation WORSE — all heads to median 1).
3. **Opponent frame** (pre-leg-5): zero truncation — but
   best_delta median 0.0000 and accept rates 0.05-0.17: the frame
   is blind (the leg-4 finding), so nothing wins, including
   truncation. Inert search, not health.
4. **Mover frame + projection (project=all, 1 half-turn)**: seed
   et-accepts 30 -> 3, armV1 33 -> 7, armV2 -> 0; committed len ~=
   spine for all; deltas informative (0.02-0.10); accept rates
   0.59-0.70. Truncation dead, search alive. CONFIRMED.

## How this unifies the history

- Leg-3 K-collapse: boundary-only grading + force-included end_turn
  (the projection option was designed in response, then parked).
- Arm T's oscillation: mover frame, project=none — living at the
  door without falling through; the value drift the search consumes
  swings play quality (the -478/-28 knife edge).
- Arms V1/V2: the value-memory fit converged the head fast; its
  shifting per-action deltas opened the door within 3-5 iterations
  (head-only vs full-unfreeze made no difference — it never was a
  trunk-gradient problem).
- Arm M (plain Gumbel): sibling disease at the per-DECISION scale
  (end_turn mass ratchet in distill targets) — projection does not
  speak to that path.

## Recommendation (user ruling pending; nothing launched)

Arm V3 = value memory (head-only, as committed) + boundary_frame
mover + project=reval (gate re-grading only; cheapest placement) or
=all. Prediction: no K-collapse AND the value-memory de-noising
finally gets its clean read on the oscillation. The mover_mp0 frame
variant stays in-tree as probe apparatus with its falsification
recorded — not a production candidate.

Cost of the whole mechanism hunt: ~$0.60 of box time, five probe
runs, one afternoon.
