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
| L2 | A turn that ran out of time is trained as an end_turn decision: 17 of 1,430 side turns (1.2%) in 350 games, where the timer settings make it detectable; undetectable when the reservoir caps the recorded time (the most common setting) | retrain: corpus v3 marks the detectable ones, and their positions carry the TIMEOUT label, which names no action (user decision 2026-09-30; `feature/corpus-v3` 1ccbb40); the undetectable rest (about 0.2-0.3% of end_turn labels) stays end_turn |

## Training against play

| id | finding | disposition |
|---|---|---|
| I1 | The core's reach and zone-of-control context were mapped to hexes through the view's hex-set order, which differs from the core's after a terrain change (Aethermaw from turn 4): in play the simulator refused 64 of 65 moves the mask offered and raised on one; pre-encoding raised on all 578 Aethermaw games | fixed, `fix/core-hex-order` e9eb76d, with a test that crosses the terrain change |
| I2 | `unit_vocab_retrain_box.sh` asserts 190 vocabulary entries; the vocabulary holds 237 (47 variation aliases on their base rows): the box would stop before pre-encoding | fixed, `fix/audit-small` 8dfab62 (the rows are checked) |
| I3 | The recruit-rejected hex flag is set in play and never in the corpus (a replay records where the recruit landed), so its weights are untrained | retrain: the input column is 0 under `observation_parity`; the mask keeps the rejection set (owner: observation builder) |
| I4 | `load_checkpoint` restores the fog gate and the terrain view but not the relevant-set basis, silently: the value tools rebuild a relevant-set checkpoint on the full board and save it that way (not on the retrain or match path) | fixed, `fix/audit-small` 8dfab62 (the checkpoint's basis wins; the value fine-tune encodes in it); `value_pretrain` and `value_head_fit` read the policy's basis now but were not traced further |

## Capacities and silent failures

| id | finding | disposition |
|---|---|---|
| C1 | `core_compare.observation_differences` compares hex-indexed arrays position by position: after Aethermaw's terrain change the applier's hex order differs from the core's, so the certification would flag all 578 Aethermaw games while every model input is identical | fixed, `fix/certify-sweep` 01d5e86 (by hex position, each side through its own geometry) |
| C2 | The Rust unit database falls back to generic stats for an unknown type without a warning (0 unknown types in 2,762 games) | retrain: count and warn (owner: observation builder) |
| C3 | An unknown type name takes the overflow embedding row without a warning on three encode paths (0 in 2,762 games) | retrain: count and warn (owner: observation builder) |
| C4 | A command whose hex holds no unit is skipped without a warning or a count, and its label is still trained (0 cases in 16 games checked; 4 engine-aborted attacks in 294,276, correctly skipped) | queued: count, and warn once per game |
| C5 | An out-of-range advancement choice becomes the first option silently; the extractor's `choose_queue` is filled and never read | queued with C4 |
| C6 | A label whose actor index is out of range adds nothing to the loss, value included, uncounted (the serial path and the holdout probe; the retrain uses checked pre-encoded records) | retrain: the sequence trainer refuses a slot mismatch |
| C7 | Two default filters (competitive 2p, 1,500 commands) do nothing only because the corpus has no `index.jsonl`; with one they would drop the 4,936 mini games at INFO | fixed, `fix/audit-small` 8dfab62 (opt-in) |
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
| O6 | Shroud is treated as fog, so the terrain of never-explored hexes shows (89 games with shroud) | fixed, `fix/no-shroud` 9f3fc13 (user ruling 2026-09-30: every shroud game is quarantined; the scenario builder refuses shroud) |

## The match harness

| id | finding | disposition |
|---|---|---|
| M1 | A move or attack the simulator refuses after the mask offered it is repeated by the argmax player: a move 8 times then the turn is forced to end, a distant attack without bound until the 20-minute kill; nothing recorded | fixed, `fix/match-harness` 8ee2934 (one bounded, counted refusal path; a match game raises on a mask/simulator disagreement) |
| M2 | The unit-vocabulary pre-registration still describes the Knalgan-forced draw | superseded by docs/parity_memory_prereg_20260929.md (uniform draws) |
| M3 | The approved `obs8` self-pin is in no box script | retrain: match 4 of the pre-registration |
| M4 | The reference checkpoint is named by path only | fixed, 8ee2934 (pinned by SHA-256; the model host's copy agrees) |
| M5 | A side's 2,000-action cap is a second, unrecorded horizon | fixed, 8ee2934 (recorded, refused on resume when changed) |
| M6 | `--compile-packed` with shared inference records a value the resume and the fit refuse | fixed, `fix/compile-packed-provenance` 45ac777 (the driver reads the server's setting) |
| M7 | Results lack each side's faction and leader (the pre-registration reads p by faction), the unplayed turns, the code version and the core switch | fixed, 8ee2934; each checkpoint's training epoch is still only in the server log |
| M8 | The retrain script re-derives the reference instead of taking the config's flags | retrain: the new script takes `reference_player.py --flags` |

## Box operations

| id | finding | disposition |
|---|---|---|
| X1 | A second entry of the certification run deleted its `ALL_DONE` and `FAILED` from the model host 10 s after they landed (cause unverified: most likely the container re-ran the onstart after its accepted self-stop); a re-run redoes the whole sweep, which has no done-marker | fixed, `fix/certify-sweep` 01d5e86 (an entry within 30 minutes of a finished run's accepted stop stops again and touches nothing; a marker after the sweep) |
| X2 | The certification sweep uploads about two files per shard, one commit each, every round: 3 x 128 + 30 commits in an hour risks the model host's hourly quota | fixed, 01d5e86 (one folder, one tarball a round) |
| X3 | The certification summary has no denominator: a crashed shard (killed, no summary line) reads as a complete, clean sweep | fixed, 01d5e86 (CLEAN, DIVERGENT or INCOMPLETE against the file list; the run fails unless CLEAN with the oracle passing) |
| X4 | The corpus rebuild's inputs step is skipped on re-entry when the tarball's first member exists, even after an interrupted extraction | retrain script: a marker after the step |
| X5 | The corpus rebuild accepts a `BUILD_DONE` from a build log restored from an earlier entry | retrain script: only the new part of the log is read |
| X6 | The corpus rebuild's default raw tarball is not on the model host | retrain script: `tier-b/corpus_v3/raw_corpus_20260929.tar`, uploaded 2026-09-29 |
| X7 | Smaller: offers show no memory column; `pull_box_records` skips a same-size rewrite; the onstart's first installs have no timeout; `is_absent` matches exception names; failures point at an empty `restore.log` | queued, low |

## The certification's power (done 2026-09-30, by the main session)

Four faults planted one at a time on one corpus game (a defender's hit
points on the applier's side, a village owner in the core's views, a unit
feature in the Python encoder, the defender's weapon choice) are each
flagged; the unplanted run is clean; a tampered engine answer fails the
oracle step with exit 1.

| id | finding | disposition |
|---|---|---|
| P1 | The applier reached the same Rust combat, reach, observation and encoding kernels as the core, so those rules were compared with themselves | fixed, `fix/certify-independent` b0fe050 (the sweep runs the applier with the kernels off, measured at no extra time; diff_core prints which kernels it ran on) |
| P2 | The engine oracle's replay compared only the recorded cases still in today's list, so renamed cases shrank it silently | fixed, b0fe050 (a missing case fails the step) |

## The core binding (done 2026-09-30, by the main session)

`core_of` holds a weak reference and checks identity, so a reused object id
cannot inherit an old binding; every write the simulator makes to its state
goes through the core when it has one; `WesnothSim.fork` forks the core.

| id | finding | disposition |
|---|---|---|
| K1 | A search fork lacked the refusal counters M1 added (the fork is built without `__init__`), so a refusal inside a search fork raised | fixed, `fix/fork-refusal-state` edb6ca9 |
