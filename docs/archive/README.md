# Archive (moved 2026-09-04)

Documents that describe superseded plans, finished legs, or
mechanisms now in quarantine. Kept verbatim for provenance; nothing
here is current. The live entry points are `CLAUDE.md`, `BACKLOG.md`
and `docs/plan_20260904.md`.

## Status and backlog history

| file | what it is |
|---|---|
| claude_status_history.md | every "Current status" block CLAUDE.md carried from 2026-06-11 to 2026-09-04 |
| backlog_20260904.md | the pre-restart BACKLOG.md, 1,055 lines of rulings and open items |
| backlog_closed_20260926.md | BACKLOG.md's sections closed between 2026-09-12 and 2026-09-24 (fixes that landed, phase 1's record, the time of day, the scenario builder, the 2026-09-14 review), verbatim |
| claude_status_20260904_20261007.md | CLAUDE.md's "Current status" log from 2026-09-04 to 2026-10-07 (references seed2 to `parity3`, phase 1, the Rust core, the parity-memory retrain, live games), verbatim |
| backlog_closed_20261008.md | BACKLOG.md's historical sections as of 2026-10-08 (value head study, phase 2's prerequisite, training-signal panel, corpus census, the 2026-09-13 audits, the hide-cover review), verbatim |

## Plans that were followed and superseded

| file | period | outcome |
|---|---|---|
| superhuman_training_plan.md | 2026-06/07 | tier-a/tier-b staging; superseded by plan_20260904.md |
| tier_a_runbook.md, tier_a_remeasure_20260720.md, eval_20260728.md | 2026-07 | tier-a 5M campaign: +133 in-lineage with search, 0-30 vs the built-in AI |
| tier_b_brief.md, tier_b_runbook.md | 2026-08 | tier-b 15M handoff legs; launcher ops now in scripts/ and docs/box_specs.md |
| autonomous_run.md | 2026-07-28..31 | 72-hour run log: four signal defects fixed, external gap unchanged |
| literature_scan_20260810.md, technique_review_20260810.md | 2026-08-10 | techniques adopted for the handoff legs; every one later quarantined |
| planning_abstractions_litreview_20260812.md | 2026-08-12 | review that led to TCS |
| redesign_1000x_20260828.md, procedure_propositions_20260826.md | 2026-08 | first 1000x program; measured against a sampling reference, so its baseline numbers are wrong (see raw_argmax_control_20260904.md) |
| az_minimal_spec.md, az_leg_20260903.md | 2026-09-03..04 | the minimal AlphaZero loop; raw pin 11-29, and the pins compared sampling players on both sides |

## Runbooks superseded by the current procedure (moved 2026-09-26)

| file | period | why it is here |
|---|---|---|
| eval_box.md | 2026-08-03 | the CPU eval box for the sampling player (`raw`); matches now run on a 4090 through persistent workers and a shared inference server (CLAUDE.md, "Eval procedure") |
| running_on_gpu.md | 2026-07 | launching `tools/sim_self_play.py` on a CUDA node; its command passes `--drill-ratio`, which no parser defines, and self-play runs through `tools/az_loop.py` now |
| gpu_perf_patches.md | 2026-07-02 | CUDA-stall patches for in-process MCTS rollouts: B3, #1 and #2 are in the code (`tools/mcts.py`, `wesnoth_ai/encoder.py`), and the note lists their CUDA checks as still required |

## Mechanism specs (all in quarantine/INVENTORY.md)

| file | mechanism |
|---|---|
| tcs_spec.md, tcs_collapse_mechanism_20260831.md | turn-commitment search and why it ends turns early |
| gbc_spec.md | event-prediction auxiliary heads |
| cgr32_spec.md | certified Gumbel re-decision |
| swap_detector_design.md | swap detector |
| credit_assignment_design_20260817.md | value-head restart credit assignment; records that common random numbers were measured dead |
| cliffness_calibration.md | value-head spread calibration |
| arm_w_spec_20260901.md | signal rebalancing arm, never launched |

## Leg records and postmortems

| file | leg | one-line result |
|---|---|---|
| mcts_vs_reinforce_eval.md | 2026-05 | MCTS vs REINFORCE on the 0.5M net |
| leg3_passivity_rootcause_20260817.md | tier-b leg 3 | turn length collapsed to 2 actions |
| leg4_erosion_rootcause_20260820.md | tier-b leg 4 | distill-prior discount flattened the policy; -309 vs seed |
| leg5_value_inversion_20260825.md, leg5_resume_verdict_20260826.md | tier-b leg 5 | proxies green, 9-31 vs seed; TCS at play time costs ~200 Elo |
| teacher_arms_20260829.md, teacher_arms_findings_20260829.md | arms T/M/G/V | K-collapse under MCTS teaching; value drift on imagined states; the signal-profiler rounds |
| arm_vg_leg_20260901.md, arm_vg2_leg_20260902.md, arm_vg3_leg_20260902.md | arms VG1-3 | value grounding: three collapses, consistency term took the update |

## Archived branches (git tags)

A branch whose code does not land is deleted once its tip is tagged
`archive/<branch>` (slashes as hyphens) and the tag is pushed. Records that
name such a branch or one of its commits resolve through these tags;
`git tag -l 'archive/*'` lists every one.

| tag | what it holds | its record on `main` |
|---|---|---|
| archive/exp-xod-dominance | XOD, parked by user ruling 2026-09-24 | the tip commit's message; docs/selfplay_redesign_20260904.md |
| archive/exp-turn-value | the turn-ranking value function: generator, features, arms | docs/turn_value_prereg_20260925.md |
| archive/refactor-step4a-sim | step 4a of the refactor, dropped when the Rust core took over the rules | docs/refactor_plan_20260925.md |
| archive/exp-memory-in-play | `parity2`'s memory in play: match tempo, a counterfactual reader | BACKLOG.md, "The memory's cost in play" |
| archive/exp-cpu-turn-planner | the CPU turn planner's design (its files are on `main`) | docs/selfplay_algorithm_design_20261007.md |
| archive/feature-critic-step1 | step 1's critics: data, training, the turn benchmark | docs/selfplay_program_20261008.md, "Step 1, measured" |
| archive/feature-lookahead-player | the one-step look-ahead player | docs/lookahead_material_gate_prereg_20261009.md |
| archive/feature-lookahead-gate-box | the look-ahead's gate script | docs/lookahead_material_gate_prereg_20261009.md |
| archive/exp-value-policy-iteration | the two above and the critics, with main merged in | docs/selfplay_program_20261008.md |
| archive/exp-memory-in-play-parity3 | the per-side-turn memory reset, the counterfactual reader, turn tempo | docs/memory_in_play_parity3_prereg_20261009.md |
| archive/exp-exposure-evaluator | the material evaluator with gold and exposure, the core's threats | docs/exposure_evaluator_prereg_20261012.md |
| archive/exp-rollout-regrade | the turn benchmark re-graded by rollouts from the mover's view | docs/rollout_regrade_prereg_20261012.md |
| archive/exp-belief-worlds | the re-grade in belief worlds | docs/belief_worlds_prereg_20261012.md |
| archive/feature-decode-knobs | the attack, recruit and time-of-day decode knobs, the knob search | docs/decode_knobs_prereg_20261012.md |
| archive/land-shutdown-drains-results | a second landing of fix/shutdown-drains-results; the other is on `main` | none needed |
