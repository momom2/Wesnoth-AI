# The audit before the parity-memory retrain (2026-09-29)

The retrain runs only when this audit has nothing open
(docs/parity_memory_prereg_20260929.md "Audit gate"). Ten independent
reviewers, each given a scope and no hypotheses: four on the data path
(the labels; what the network observes against a player; training
against play; capacities and silent failures), and four targeted at the
parts most at risk for the coming boxes, at the user's request (box
operations; whether the certification can fail when the core is wrong;
the binding of Python views to the Rust core; the match harness), and
two on the retrain's own code once it was written (its data path; its
trainer, match path and box script). Their
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
| C4 | A command whose hex holds no unit is skipped without a warning or a count, and its label is still trained (0 cases in 16 games checked; 4 engine-aborted attacks in 294,276, correctly skipped) | fixed, `feature/match-memory` 74750e1 (both appliers warn; the label builder leaves such a decision unpaired, and the sequence pre-encoding counts it) |
| C5 | An out-of-range advancement choice becomes the first option silently; the extractor's `choose_queue` is filled and never read | fixed, 74750e1 (a warning in both appliers) |
| C6 | A label whose actor index is out of range adds nothing to the loss, value included, uncounted (the serial path and the holdout probe; the retrain uses checked pre-encoded records) | retrain: the sequence trainer refuses a slot mismatch |
| C7 | Two default filters (competitive 2p, 1,500 commands) do nothing only because the corpus has no `index.jsonl`; with one they would drop the 4,936 mini games at INFO | fixed, `fix/audit-small` 8dfab62 (opt-in) |
| C8 | A recruit option's max experience is the type's unscaled value (1.43 times the recruited unit's at the 70% modifier 295 of 300 games use) | retrain (owner: observation builder) |
| C9 | The pre-encoded cache's fingerprint covers neither the label builder's version nor the core's phase | retrain: the new pre-encoding records both |

## What the network observes

| id | finding | disposition |
|---|---|---|
| O1 | Delayed shroud updates (`[auto_shroud] active=no`, then `[update_shroud]`) are not modelled: reconstruction clears fog at every move, so the network sees what the player's client had not revealed. 59 of 700 sampled games use it; 3,077 of 25,728 of their decisions hold uncommitted vision and 550 show an enemy the player had not been shown (about 2% and 0.3% of training decisions) | fixed, `feature/delayed-shroud` 37b4820 (the engine's rule in the core, the applier and the extractor; corpus version 4; docs/wesnoth_rules.md "Delayed shroud updates") |
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
| X7 | Smaller: offers show no memory column; `pull_box_records` skips a same-size rewrite; the onstart's first installs have no timeout; `is_absent` matches exception names; failures point at an empty `restore.log` | fixed, 8c89a7d (the RAM column; content hashes; 10-minute bounds; restore messages reach restore.log); the name-based absence check stays: a new absent error type reads as unreachable, which fails the entry loudly |

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

## The retrain's own code (done 2026-09-30, two independent reviewers)

Gate item 4 of the pre-registration. One reviewer read the data path
(delayed shroud updates in both appliers and the extractor, the sequence
pre-encoding, the view and core transfer), the other the sequence trainer,
the probe, memory in the match path and the box script. The data-path
reviewer ran both appliers command by command on 6 corpus games that delay
(2,947 decisions): seen hexes, pending entries, delaying sides and unit
positions matched after every command. The other reviewer reproduced
finding R1.

| id | finding | disposition |
|---|---|---|
| R1 | The pre-encoder's record classes lived in the script, so its spawned workers pickled them as `__mp_main__.GameSequence`, which the trainer cannot import: the box's pass would have crashed at its first window, and re-entry would have kept the records | fixed, `feature/match-memory` 6e0df7b (wesnoth_ai/sequence_records.py; the end-to-end test runs the pre-encoder by path in a subprocess and fails under the old layout) |
| R2 | A failed memory barrier was not kept: a re-entry resumed the pass past it | fixed, 6e0df7b (the checkpoint keeps the verdict; a resume stops at once, tested) |
| R3 | A new stage replaces the staged repository, the raw replays and the corpus with it, while their markers survived; the barrier line was written from a missing input and never recomputed | fixed, 6e0df7b (stage-bound markers in both box scripts; the line only from its input, written whole) |
| R4 | On a new machine the finished matches came back empty and were replayed over the first run's records | fixed, 6e0df7b (fits, timings and game tarballs restored; a match resumes in its directory) |
| R5 | The memory barrier and match 2 compare 64 slots with 0: a memory that is not carried could pass through its extra tokens alone | recorded: the probe reads 64 slots carried against 64 slots reset at every decision (`belief_carried`); the barrier stays as pre-registered |
| R6 | The pre-registered per-phase value AUC was not computed; nothing checked that the pass trains every pre-encoded position; the barrier's standard error treated a game's two sides as independent | fixed, 6e0df7b (the same-turn AUC by turn bucket; exit 4 on a short pass; the standard error across games) |
| R7 | An event handler that leaves undo disabled makes its action final, which commits a delaying side's vision: in our games the Plan Unit Advance modification's first move of each side turn and its menu events. 6 of 400 sampled games both use it and delay; their seen hexes differed at 27 of 2,859 decisions | fixed, f5def12 (the record carries the modification and its menu events; both appliers commit there; wheel phase 25) |
| R8 | An attack that a disconnect aborted before its first draw skipped the commit the engine's handler makes first | fixed, f5def12 |

## The pre-launch audit, round 1 (2026-09-30, six independent reviewers)

User request, 2026-09-30: audit, fix, audit again until an audit comes up
clean. Six reviewers, each given a scope and no hypotheses: the model and
its memory (A), the sequence trainer (B), the training data (C), the match
path (D), box operations (E), the observation's fidelity to the engine (F).
All of it lands on `fix/audit-round1`; the wheel is phase 26, the corpus
version 5, `OBSERVATION_EPOCH` 11.

| id | finding | disposition |
|---|---|---|
| C1 | critical: the replay applier never carried a side's Random choice or the era, so every posterior was one-hot on the true faction (3,760 of 5,528 sampled positions face a Random opponent) | fixed: the state carries both, and the lobby's random faction mode (a Random side under "No Mirror" cannot draw the other side's faction; 45 of 109 sampled replays play it), extracted at version 5 |
| C2 | critical: Hornshark Island's placed units (Young Ogre, Sergeant, Ruffian) are fieldable by no faction, so its games read as posterior errors, 1.5% of positions against a 0.1% barrier | fixed: a type no faction of the era fields carries no evidence; a posterior that leaves out the true faction is now counted with the inconsistent ones |
| C3, E1 | critical: the corpus barrier asserted outcomes == candidates, which quarantined and failed games break | fixed: one check for both box scripts (`build_imitation_dataset.py --check`), tested |
| E2 | major: the certification's oracle step kept only its second comparison's exit code | fixed |
| F1 | major: the certification cannot see the sighting record, the seen types or the parity columns, which exist only in Rust | fixed for the state: `tools/sighting_oracle.py` follows the Python applier and `diff_core --sightings` compares every side's record after every command (10 of 10 sampled games agree over 3,018 comparisons; planted faults diverge on 8 and 10 of 10); the parity encoding's columns stay covered by tests, written into the gate |
| C4, E9 | major: sequence records and a pass could outlive a re-stage | fixed: a run belongs to the stage that began it (`box_bind_run_stage`; `RESUME_OTHER_STAGE=1` continues it), and the sequences are rebuilt for a new stage |
| A1 | minor: a unit that leaves the board where a side cannot see it (a neutral side's kill in fog) vanished from the side's record, a cue no player has | fixed: it stays until the side's end of turn (wheel phase 26) |
| A2, B7 | minor: the memory barrier pairs over games, the pre-registration said game-sides | pre-registration amended: a game's two sides share its luck |
| B1 | minor: obs8's holdout cross-entropy could read games the pre-encoding dropped | fixed: it reads the sequence manifest's games; the barrier line says DECISIONS_DIFFER when the counts part |
| B2 | minor: obs8's side ran in fp32, the arm's in bf16 | fixed: the same autocast |
| B3, E4 | minor: the pass's done check read a restored log | fixed: this entry's bytes and its exit code; a cut pass logs SEQUENCE_TRAIN_CUT |
| B4 | minor: non-finite steps were neither caught nor counted | fixed: a non-finite step is skipped and counted, three in a row stop the pass (exit 5); every probe reading counts non-finite values and the memory barrier fails on any |
| B5 | minor: the probe's chunks shrank as game-sides ended (hours, not 0.4 h) | fixed: a finished game-side hands its row to the next |
| B6, C7, D1 | minor: `OBSERVATION_EPOCH` read 10 against the documents' 11 | fixed: 11 |
| B9 | nit: a resume could repeat a save and a probe, and logged an inflated rate | fixed |
| B11 | nit: no test pinned what each step's memory reads | fixed: tested at windows of two (start, carried with its gradient, detached at a window's start) |
| B12 | nit: the last-seen baseline never forgot a dead enemy | fixed: a sighting whose hex the side sees empty is forgotten |
| A3, B10 | nit: the logged memory gradient covered more than the write | fixed: the write's |
| C5 | minor: a timer that refills to its cap every turn hides a timeout, whose end_turn is then a player's label (about 0.2% of end_turn labels) | recorded: counted in the manifest (`end_turn_timeout_undetectable`). Rejected: recovering them from the action bonus, because the most common setting (turn bonus equal to the reservoir) leaves nothing to recover |
| C6 | minor: the sequence manifest was written in place | fixed: written whole |
| C8 | nits: overflow type lookups uncounted; "value states" listed among the manifest's counts; Dunefolk on the overflow faction row | the lookups counted; the list corrected; the overflow row kept, documented as the posterior's |
| D2 | nit: game records did not carry the memory sizes | fixed |
| D3 | nit: a failed fit was never redone | fixed |
| D4 | nit: a refused recruit stepped the memory, which the corpus never does (the engine records nothing for it) | fixed: the memory goes back; the design text corrected |
| D5 | nit: a compiled per-process player could lose its memory to the next call | fixed: refused |
| D | residual: a failed game is replayed on its seed, so a deterministic failure keeps a match short | recorded: such a match now reads MATCH_FAILED and fails the run's finish; the match can be replayed from the uploaded checkpoint |
| E5 | minor: a re-entry ran the script as later uploaded, and a stage path could be overwritten | fixed: the box runs the stage's own copy; an existing stage path is refused unless `--replace` |
| E6, E7 | minor: the rental preflight ignored the dead-man's switch, the disk, the memory and the raw corpus | fixed: the script's `BOX_MAX_H` plus 1.5 h, its `box-needs`, its `RAW_TAR` |
| E8 | minor: ALL_DONE could lose its time to a slow upload.log | fixed: a third of the reserve stays for it |
| E10 | minor: the switch could mark a finished run FAILED, and gave up after a refused stop | fixed: firing after a finish it only stops the instance, and it retries a refused stop every 10 minutes |
| E11 | minor: restarts got fresh deadlines; an onstart without HF ran nothing | fixed: the deadline is the stage's first entry's; the onstart runs the disk's copies when HF cannot answer, and appends to its log |
| E12 | minor: a match that failed outright still finished the run as done | fixed |
| E13 | minor: worker counts had no memory bound | fixed: `box_workers` |
| E nits | the finish reason's stray newline; MATCH_CUT_MIN under a match's own worst case; match tarballs fetched again; an empty holdout cross-entropy blocked its retry; the runbook's switch factor | fixed; the token's scope (a write token for the one repository) is the user's decision |
| F2 | minor: a delaying side's move blocked by an unseen enemy committed nothing, because the record cut the route before the blocker | fixed: the record keeps the route's next hex and both appliers look for the blocker there |
| F3 | nit: the parity hex time of day was not what the interface shows | fixed: the interface's rule (docs/wesnoth_rules.md) |
| F4 | nit: two texts misstated the discovery rule and cited a missing catalog entry | fixed; the entry written |

## The pre-launch audit, round 2 (2026-09-30, four independent reviewers)

Four reviewers read a frozen checkout of round 1's fixes (896385c): the
fixes themselves, the training data, box operations, and the
observation's fidelity to the engine. The model, the trainer's core and
the match path, where round 1 found nothing critical or major, were
covered through the review of their fixes. The fixes land on
`fix/audit-round2`; the wheel is phase 27.

| id | finding | disposition |
|---|---|---|
| 2E1 | critical on some hosts: `box_workers` (round 1's E13) died under `set -u` on a host without cgroup-v2 `memory.max`, so both runs would stop at their corpus step | fixed: it reads the v1 limit too and takes the cgroup's headroom; tested on every layout |
| 2F1 | major: a mover walking out of a watcher's view was recorded on the last hex it was visible on, where the display shows it walk into the next one (2,571 of 4,343 sampled sighting tokens) | fixed in the core and the oracle (phase 27): the hex after the last visible one; the catalog entry rewritten from the animation code |
| 2F2 | minor: after a fight that refogs the defender's side, the record kept the attacker's hit points from before the fight (16 of 2,766 sampled fights) | fixed: the side records what the fight showed it before its refog, in the core and the oracle |
| 2F3 | minor: the block of a cut route ignored the checkup's `stopped_early` | fixed: a route that ran out of this turn's moves is not a block |
| 2F4 | minor: Hornshark Island's placed units tell a knowing player the faction; the posterior ignores them | recorded: the network sees the placed units as tokens; one map, until a recruit or the leader is seen |
| 2F5 | minor: vision through teleport (Silver Mage) is not modelled | recorded in the catalog's "Not modelled" list, with its frequency (none in 95 sampled fog games) |
| 2F6 | minor: the certification compares two implementations of one reading of the engine for the sighting record, the block, the hex time of day and the Random prior | written into the catalog, the design and the gate |
| 2E2, 2E3 | minor: a match that crashed read as cut; the finish counted earlier entries' match lines | fixed: each match's verdict is decided in this entry, from its last exit code and its fit |
| 2E4, 2E11 | minor: the certification marked its sweep done with shards short of a verdict, and logged every shard's exit as 0 | fixed |
| 2E5 | minor: the hours an instance spends stopped counted against the switch | fixed: the switch charges the stage's running time |
| 2E6 | minor: one failed fetch at a fresh rental left the instance billing | fixed: three attempts, then the onstart stops the instance itself |
| 2E7 | minor: the restart path gave up after a refused stop | fixed: it retries |
| 2E8 | minor: the preflight checked neither the GPU nor the cores | fixed: `gpu_ram_gb` and `cores` in `box-needs`; other instances on the account are warned of |
| 2E9 | minor: a step that survives its KILL held the entry until the switch | fixed: abandoned `BOX_UNKILLABLE_S` later |
| 2E10 | minor: the probe was silent for its whole length, and the pre-registered probe count and time were wrong | fixed: a line per memory size; the cost table corrected |
| 2E12 | nit: restored match tarballs went up again | fixed: the unpacked directory is recorded as landed |
| 2D3 | minor: a stale `finish_done` could let the switch stop an instance mid-upload | fixed: the entry clears it |
| 2C1 | minor: a Rust panic in a pool worker hung the pre-encoding and the corpus build | fixed: counted as that game's error |
| 2D5 | nit: a posterior leaving out the truth was not counted when the truth was not a candidate | fixed |
| 2D6 | nit: a memory reset for a non-finite value was silent | fixed: counted and logged |
| 2D7 | nit: no test exercised the sighting comparison | fixed: a synthetic game, clean, and diverging under a planted fault |
| 2D8, 2F7 | nit: the Python applier's other SideInfo rebuilds dropped the Random choice | fixed: they keep every field |
| 2D4 | nit: the time-of-day test pinned half its rule | fixed |
| 2D9, 2D10 | nits: diff_core's totals mixed its comparisons in; the catalog quote left out the shroud branch | fixed |
| 2C2, 2C3 | nits: a turn ended with under a second left reads as a timeout; a player without automatic moves makes the goto moves the engine would | recorded in the docstrings: a replay cannot tell them apart |
| 2C4 | nit: stored records kept the true enemy faction, and the design said pairs store their value state | fixed: the stored encoding leaves the faction out; the design corrected |

## The pre-launch audit, round 3 (2026-10-01, three independent reviewers)

Three reviewers read a frozen checkout of round 2's fixes (966600e): the
fixes themselves, box operations, and the sighting record's fidelity to
the engine. Two of them found the same defect independently. By the
user's word, the audit stops after this round. The fixes land on
`fix/audit-round3`; the wheel is phase 28.

| id | finding | disposition |
|---|---|---|
| 3F1 | major (one reviewer; minor for the other): round 2's fix 2F2 noted a defended fight after the attacker advanced, so the defending side's record held the advanced type at full hit points, which the player never saw (485 of 85,091 sampled sighting tokens; 0.17% of sampled decisions); the oracle made the same mistake, so the certification passed it | fixed: the fight refogs the defender's side before either unit advances, in the core and the applier (`attack_unit_and_advance`, attack.cpp:1556-1567, quoted in the catalog); the oracle reads the attacker as the fight left it from the applier |
| 3F2 | minor: the oracle took a live unit under the dead defender's id (a plague corpse can take it) for the defender, and skipped the note | fixed by 3F1: the oracle takes the refog from the applier; a test certifies both fights and diverges under round 2's oracle |
| 3F3 | minor: on Hornshark Island a placed Woodsman that advances to a Poacher or Trapper, which only the Knalgan Alliance recruits, excluded the true faction (786 of 89,466 Hornshark decisions) | fixed: the players' units the scenario placed, and what they advance to, are not seen types (`faction_posterior.scenario_unit_ids`, set once the scenario is set up, in the core and the oracle) |
| 3F4 | minor: nothing checked the sighting stream against the record | fixed: `diff_core --sightings` compares the stream at every encoded decision with the one the oracle's record gives |
| 3F5 | nit: an aborted attack noted nothing after committing a delaying side's vision | fixed |
| 3F6 | nit: a teleport step out of view was recorded at its fogged destination; the display shows a teleport's arrival only where it is seen (`teleport_unit_between`, udisplay.cpp:74-113) | fixed, in the core and the oracle; the catalog entry extended |
| 3F7 | nit: the corpus exercises the gone rule once (Micro Isar, one game) | recorded; the synthetic tests cover it |
| 3T1 | nits: the route tests landed one hex into fog, where a leak of the landing hex passes; the core half of 2F3 and a later change to a directory marked landed had no test | fixed: routes two hexes into fog; the core runs the ran-out route beside the applier; the upload test adds a game |
| 3E1 | minor: an entry that met another stage's run cleared that run's `ALL_DONE` and `FAILED` on HF before refusing, and wrote its own | fixed: `box_init` refuses first and sends nothing (`REFUSED` on the disk), as it does when it cannot read whose run it is |
| 3E2 | minor: a GPU stuck after a device fault would hang each remaining match to its cut | fixed: a two-minute GPU check before each match finishes the entry (`GPU_UNRESPONSIVE`); a step that outlives its KILL finishes it too |
| 3E3 | nit: round 2's premise for 2E9 was wrong: GNU `timeout` ends with a KILL to its whole group, itself included, so no stuck child holds the entry | recorded in the runbook; the GPU check covers the risk that remains |
| 3E4 | minor: `box_workers` gave the workers all the headroom, leaving none to the step's parent process, and counted page cache as used | fixed: one share for the parent; `anon` (v1 `total_rss`) as used |
| 3E5 | minor: the uploader's `--clear` had one attempt, and `--landed` ran outside the upload lock | fixed: a clear retries as a restore does; `box_mark_landed` takes the lock |
| 3E6 | minor: the onstart's own stop tried once, in one form | fixed: ten rounds, the Bearer and the query forms |
| 3E7 | minor: `DONE` could reach HF before the final probe, and the barrier read the last probe row whatever its position | fixed: `DONE` waits for the probe file; the final probe row says so, and the barrier line reads `NO_FINAL_PROBE` otherwise |
| 3E8 | minor: a short match's fit stayed on HF while its second attempt ran, and a match whose games could not be counted had no verdict | fixed: the fit and timing are cleared on HF too; `MATCH_FAILED` |
| 3E9 | minor: the preflight warned of other instances on the account, checked no GPU model and let a VM host through | fixed: refused unless `--allow-other-instances`; `gpu=` in `box-needs` (the retrain's: 4090); VM hosts refused |
| 3E10 | nit: the certification's first upload came 30 minutes in | fixed: a round right after the box facts |
| 3E11 | nits: two recovery cases (a start within 30 minutes of a finish; the switch's allowance used up) and per-machine logs were undocumented | recorded in the runbook |
| 3E12 | nit: a full disk | accepted: the cost table derives the 120 GB (3.7-3.8 KB a pre-encoded position, about 21 GB); the step that fills it fails and reports |
| 3E13 | nit: the pre-registration's total left out the probes' row | fixed: 15-20 box-hours, $7.5-12; the switch's comment |
| 3D1 | nit: the sim's PvP defaults rebuilt SideInfo without the Random choice | fixed |
