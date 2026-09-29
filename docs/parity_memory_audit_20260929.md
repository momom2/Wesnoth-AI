# The audit before the parity-memory retrain (2026-09-29)

The retrain runs only when this audit has nothing open
(docs/parity_memory_prereg_20260929.md "Audit gate"). Eight independent
reviewers, each given a scope and no hypotheses: four on the data path
(the labels; what the network observes against a player; training
against play; capacities and silent failures), and four targeted at the
parts most at risk for the coming boxes, at the user's request (box
operations; whether the certification can fail when the core is wrong;
the binding of Python views to the Rust core; the match harness). Their
scripts and outputs are in the session's scratchpad (`review_*`,
`audit_*`); the findings below are verified by their reviewers unless
marked otherwise.

Dispositions: **fixed** (with the commit), **retrain** (built into the
retrain, owner named), **queued** (a fix to land before the box), and
**accepted** (recorded; the user's word needed).

## Found while running the certification box, before the audit

| id | finding | disposition |
|---|---|---|
| B1 | `core_certify_box.sh` read the corpus tarball's member count after the `with` block closed it: the box stopped at its corpus step | fixed, 0.10.1 |
| B2 | `nproc` reads 1 on the pytorch images (it honours `OMP_NUM_THREADS=1`): the sweep would have run as one shard and been cut at 90 minutes; `corpus_v2_rebuild_box.sh` had the same default | fixed, 0.10.1 (`box_cores`, the cgroup quota) |
| B3 | Vast stopped answering offer searches on `id=`: `rent_box.py create` refused every offer | fixed, 0.10.2 (`ask_contract_id=`) |

## The labels

| id | finding | disposition |
|---|---|---|
| L1 | Moves the engine runs at a side's turn start for multi-turn orders (`execute_gotos`, before the player's input) are trained as decisions: 494 of 86,650 player moves in 500 games (0.57%) | retrain: corpus v3 marks them as the engine's, not the player's; applied, never paired (owner: main session) |
| L2 | A turn that ran out of time is trained as an end_turn decision: 17 of 1,430 side turns (1.2%) in 350 games, where the timer settings make it detectable; undetectable when the reservoir caps the recorded time (the most common setting) | retrain: corpus v3 marks the detectable ones; the rest is **accepted** pending the user |

## Training against play

| id | finding | disposition |
|---|---|---|
| I1 | The core's reach and zone-of-control context were mapped to hexes through the view's hex-set order, which differs from the core's after a terrain change (Aethermaw from turn 4): in play the simulator refused 64 of 65 moves the mask offered and raised on one; pre-encoding raised on all 578 Aethermaw games | fixed, `fix/core-hex-order` e9eb76d, with a test that crosses the terrain change |
| I2 | `unit_vocab_retrain_box.sh` asserts 190 vocabulary entries; the vocabulary holds 237 (47 variation aliases on their base rows): the box would stop before pre-encoding | queued: check distinct ids; the retrain's own script does so |
| I3 | The recruit-rejected hex flag is set in play and never in the corpus (a replay records where the recruit landed), so its weights are untrained | retrain: the input column is 0 under `observation_parity`; the mask keeps the rejection set (owner: observation builder) |
| I4 | `load_checkpoint` restores the fog gate and the terrain view but not the relevant-set basis, silently: the value tools rebuild a relevant-set checkpoint on the full board and save it that way (not on the retrain or match path) | queued |

## Capacities and silent failures

| id | finding | disposition |
|---|---|---|
| C1 | `core_compare.observation_differences` compares hex-indexed arrays position by position: after Aethermaw's terrain change the applier's hex order differs from the core's, so the certification would flag all 578 Aethermaw games while every model input is identical | queued, before the certification: compare by hex position |
| C2 | The Rust unit database falls back to generic stats for an unknown type without a warning (0 unknown types in 2,762 games) | retrain: count and warn (owner: observation builder) |
| C3 | An unknown type name takes the overflow embedding row without a warning on three encode paths (0 in 2,762 games) | retrain: count and warn (owner: observation builder) |
| C4 | A command whose hex holds no unit is skipped without a warning or a count, and its label is still trained (0 cases in 16 games checked; 4 engine-aborted attacks in 294,276, correctly skipped) | queued: count, and warn once per game |
| C5 | An out-of-range advancement choice becomes the first option silently; the extractor's `choose_queue` is filled and never read | queued with C4 |
| C6 | A label whose actor index is out of range adds nothing to the loss, value included, uncounted (the serial path and the holdout probe; the retrain uses checked pre-encoded records) | retrain: the sequence trainer refuses a slot mismatch |
| C7 | Two default filters (competitive 2p, 1,500 commands) do nothing only because the corpus has no `index.jsonl`; with one they would drop the 4,936 mini games at INFO | queued: make the filters explicit options, off by default |
| C8 | A recruit option's max experience is the type's unscaled value (1.43 times the recruited unit's at the 70% modifier 295 of 300 games use) | retrain (owner: observation builder) |
| C9 | The pre-encoded cache's fingerprint covers neither the label builder's version nor the core's phase | retrain: the new pre-encoding records both |

## What the network observes

| id | finding | disposition |
|---|---|---|
| O1 | Delayed shroud updates (`[auto_shroud] active=no`, then `[update_shroud]`) are not modelled: reconstruction clears fog at every move, so the network sees what the player's client had not revealed. 59 of 700 sampled games use it; 3,077 of 25,728 of their decisions hold uncommitted vision and 550 show an enemy the player had not been shown (about 2% and 0.3% of training decisions) | queued: the engine's rule in the core and the extractor, after the observation builder lands (the same Rust files) |
| O2 | Nothing tells the network whether fog is on, and with fog off `observation.seen` covers only the units' vision (3,229 fog-off games) | retrain: the fog-on global; "sees the hex now" is seen or fog off (owner: observation builder) |
| O3 | Recruit rows read has-attacked 0 where a fresh recruit has no attacks left | retrain (owner: observation builder) |
| O4 | = I3 | |
| O5 | The village bit comes from a modifier no Ladder map sets: an unowned water village reads as water in every state (wider than the parity census's gap 8) | retrain: the static bit from the terrain (owner: observation builder) |
| O6 | Shroud is treated as fog, so the terrain of never-explored hexes shows (89 games with shroud) | **accepted** pending the user: public maps, the reasoning of the statues ruling |

## Still running

Box operations; the certification's power; the core binding; the match
harness.
