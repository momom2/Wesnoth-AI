# Project status

The project's current state. CLAUDE.md holds only instructions and points
here. Update this file on `main` when a result lands. When a block is
superseded, move it verbatim to `docs/archive/` and index it there.
Earlier status logs: `docs/archive/claude_md_20261109.md` (CLAUDE.md's
"Current status", 2026-10-08 to 2026-10-21) and the files listed in
`docs/archive/README.md`.

Last updated 2026-11-10.

## In one paragraph

Self-play now beats the reference. The policy-iteration rounds, run on the
desktop's RX 6600 from 2026-10-29, took `own2` through 18 adopted rounds.
At `own2`'s own decode, round 18 beats `own2` by +105.7 ± 3.2 Elo, and
paired on the same seeds it leads round 16 by 3.2 SE. Nothing has been
adopted as the new reference yet: that is the user's call. Since
2026-11-10 03:03 the line runs lean (departures 43-44): each step is
adopted unread, and evaluation falls from about 42% of the GPU to about
11%. A 3,200-game claim and a 3,200-game argmax match every 6th round can
pause it.

## The reference player

`own2` at `raw:t0+eo-1.5` with 64 memory slots, adopted 2026-10-21. It is
pinned in `configs/reference_player.json` (HF
`tier-b/memory_own_round2_20261011/own.pt`, local
`training/checkpoints/own2.pt`). `tools/reference_player.py --flags b`
turns it into run_elo_batch flags. Its lineage: each step is a match of 800
decisive games (6,400 for `own2`) under the observation of its day. Elo
does not chain across observation epochs.

| reference | adopted | over its predecessor |
|---|---|---|
| `own2` | 2026-10-21 | +22.7 ± 4.4 over `own1`: a second round of training on its own games; the memory no longer costs strength |
| `own1` | 2026-10-11 | +88.6 ± 12.7 over `parity3`: trained on its own games, the memory's cost in play cut from 105 to about 22 Elo |
| `parity3` | 2026-10-04 | +82 ± 13 over `parity2` (imitation under the anneal rule) |
| `parity2` | 2026-10-02 | +191 ± 14 over `obs8`: the network observes what a player sees, with a 64-slot memory |
| `obs8` | 2026-09-25 | +73 ± 13 over `terrain` |
| `terrain` | 2026-09-20 | +263 ± 16 over `relset` |
| `relset` | 2026-09-11 | +56 ± 12 over seed2 |
| seed2 | 2026-09-11 | +33 ± 12 over the original imitation seed |

Every reference up to `own2` came from imitation of the human corpus,
followed by fine-tuning on its own recorded games.

## Self-play: the policy-iteration rounds

- **Design:** `docs/training_design_20261019/current_design.md`. The
  pre-registration and its 44 departures are in
  `docs/stage_c_rounds_local_prereg_20261030.md`. The autonomous window's
  card, plan, reviews and log are in `docs/autonomy_20261029.md`.
- **A round:**
  1. π_k plays itself at T=1: 5,000 games to train on, plus 2,500 held out
     (more in rounds 2-6).
  2. One gradient pass computes the outcome-labelled policy gradient over
     the **policy heads only**; the trunk and the memory do not move.
  3. One step is taken on a KL ladder. It is read on the held-out games
     (half A chooses the step, half B confirms it) and adopted if half B
     reads it positive. When the importance weights fail their ESS check,
     a 10,000-game cross-play match decides instead.
  4. Every third round, a claim match plays the round's policy at T=1
     against `own2` at its decode.
- **A lean round** (from round 19, departures 43-44; `--rounds-lean` on
  `feature/lean-rounds`):
  - 5,000 games plus 1,250 held out;
  - the same pass;
  - a step that is the KL search alone, along the average of the round's
    direction and the 8 before it, to the radius. It is adopted unread.
  - Every 6th round, a 3,200-game claim and a 3,200-game argmax match on
    the finals' seeds. The line pauses unless the claim gains 2 combined SE
    over the last claim and the argmax match 1 SE over the last argmax
    reading.
  - About 5.1 h a round.
- **Code:** `scripts/rounds_local.py` and `scripts/chain_local.py`, over
  `tools/pg_corrections.py`, `tools/pg_pass.py`, `tools/pg_round.py` and
  `tools/pg_eval.py`. The library is `wesnoth_ai/pg_*.py`.
- **Wall time on the desktop:**
  - the games: about 2.25 h;
  - the pass: about 2.7 h;
  - the step: about 1.5 h;
  - a claim (12,800 games): about 3.5 h;
  - so about 6.3 h a round, or 9.7 h with a claim.
- **Games played:** stage 1 played 46,000 (stage C and round 1 read them),
  and rounds 2-18 played 150,000. Evaluation matches played about 210,000
  more.
- **Run state:** `C:\Users\momom2\wesnoth_runs\` (`rounds_local/`,
  `chain.log`, `chain_lean.console.log`). The code runs from the detached
  worktree `wesnoth_ai-run` at 5d67c834 (`feature/lean-rounds`).
- **Round 19's first games failed a gate** (2026-11-09 20:41, departure
  42). The readout's I-V check read v against recruit at z −3.66, against
  the gates' 3.45.
  - The signal is diffuse: both shards, every map, not a few games. A
    bootstrap reproduces its SE, and over the line's 228 I-V readings z
    reads RMS 1.03. It fits the gates' 1% false-alarm rate, but the gate
    stands on those games.
  - Round 19 is being replayed on fresh seeds from 29,500,000, under
    departure 42's retry rule, from 2026-11-10 03:03. Every gate must pass
    on the fresh games alone.
  - `wesnoth_runs/tools/retry19_watch.py` writes a STOP before round 19's
    step if the fresh games read v against recruit below z −2.
  - The run ends by 2026-11-13 18:00, with the final into
    `final_match_lean/`.

### Measured

All matches below ran on the desktop with fp16 serving on both sides,
12,800 decisive games each.

**At the reference's decode** (argmax with the −1.5 end-turn offset, both
sides), against `own2`, seed base 900,000:

| policy | Elo |
|---|---|
| round 3 | +31.8 ± 3.1 |
| round 6 | +48.9 ± 3.1 |
| round 9 | +63.1 ± 3.1 |
| round 12 | +82.9 ± 3.2 |
| round 15 | +96.5 ± 3.2 |
| round 16 | +94.1 ± 3.2 (fresh seeds, base 980,000: **+102.7 ± 3.2**) |
| round 18 | **+105.7 ± 3.2** |

**Claims** (the round's policy at T=1 against `own2` at its decode):

| claim | Elo |
|---|---|
| `own2` itself (D0) | −415.2 ± 5.5 |
| round 3 | −296.8 ± 4.3 |
| round 6 | −204.2 ± 3.6 |
| round 9 | −146.5 ± 3.3 |
| round 12 | −100.8 ± 3.2 |
| round 15 | −43.4 ± 3.1 |
| round 18 | −8.7 ± 3.1 |

Every claim landed within 2 SE of the forecast the rounds' own half-B
readings made.

**Both sides at T=1:** round 16 beats `own2` by +429.1 ± 5.7 Elo, round 3
by +134.1 ± 3.3.

### What the numbers do not show

- **At argmax the gains are uneven.**
  - Paired on the same seeds, in score per round:
    - rounds 4-12: about +0.008;
    - rounds 13-15: +0.006;
    - round 16: −0.003 ± 0.005;
    - rounds 17-18: +0.0154 ± 0.0048 together, about +0.008 a round.
  - Off Arcanclave Citadel, round 16 minus round 12 reads only +0.0091 ±
    0.0054. Round 16 wins 560 of its 616 games on that map. Whether that is
    a sound plan or a hole in `own2` is not established.
- **The rounds mostly sharpen the sampled policy toward its own argmax.**
  The gap between the policy at T=1 and at argmax shrank from 415 Elo
  (`own2`) to about 140 (round 15). That headroom is finite.
- **The heads' gradient signal is falling.** Its norm went from 3.97 (round
  2) to about 1.3 (rounds 11-13), while the noise per split stayed flat, so
  each round's 5,000 games buy less (`training/metrics/denoise_20261105.json`).
- **The trunk has never trained in self-play.**
  - Stage C's planted check missed on both seats by about 6 SE. The miss
    lies in the trunk's part of the full-network gradient, and it has not
    been root-caused.
  - P was therefore skipped (departure 31). The heads-only rounds do not
    touch that part.
- **Serving:** fp16 here against bf16 on a box. The fp16 path matches
  fp32's argmax on 200 of 200 bench positions; bf16 has fewer mantissa
  bits. The two kinds of match are not pooled in one fit.

## Open (for the user unless noted)

- **Adopt round 18, or a later round, as the reference.** The evidence is
  a local match of 12,800 decisive games (+105.7 ± 3.2), and the
  checkpoint is `C:\Users\momom2\wesnoth_runs\rounds_local\round_18\pi.pt`
  (SHA-256 c5796941...). Copies of rounds 3, 15, 16 and 18 are in
  `C:\Users\momom2\wesnoth_checkpoints_window\`.
- **Merging `feature/lean-rounds`** (3 commits, the code the line runs)
  and the log commits on `feature/local-gpu-stage1` into `main`. The full
  suite passed at 5d67c834; CI has not run on it.
- **The next training step.** The options, each a single factor:
  - more games per step, against the falling signal-to-noise;
  - training the trunk, after the plant's miss is root-caused;
  - a step that budgets its KL by decision class. Per decision, recruits
    return about 3.4 times what moves do, while moves take 69-90% of the
    step.
- **Pushing `main`.** It holds the window's merge (0.32.0). No GitHub
  credentials are stored on the desktop.
- Unmerged branches from before the window: `feature/int8-linear` (6
  commits) and `feature/async-replies` (4).

## Standing measured facts (from before the rounds)

- **Memory:** within `own2`, 64 slots against 0 reads +13.0 ± 12.3.
  - In `parity3` the 64 slots cost 105 Elo. A stronger end_turn offset
    recovered 65 of it. The rest was the memory continuing its own
    history's patterns, the causal confusion of history-conditioned
    imitation.
  - Training on its own games removed that cost
    (`docs/memory_own_round2_prereg_20261011.md`).
- **The end_turn logit offset −1.5** is worth about +229 Elo at decode
  (`docs/endturn_rule_prereg_20260919.md`).
- **No cheap turn grader passes**: within-position correlation against
  playout truth is 0.22-0.53 (`docs/turn_value_prereg_20260925.md`).
- **Turn-level gaps exist:** 7 of 60 holdout positions have a sampled turn
  better by at least 0.25 (`docs/turn_gap_ref_prereg_20260921.md`).
- **Search:** Gumbel-MCTS-32 over atomic actions lost to argmax by −124 ±
  58 (`docs/raw_argmax_control_20260904.md`). Every search and playout
  runs on the true state, so a search used in play needs a belief model
  (`docs/hidden_information_20260926.md`). Common random numbers across
  branches were measured dead.

## The simulator

- The Rust core (`rust/wesnoth_core`, phase 38) is the state of record. It
  answers every rule: commands, events, combat, reach, vision,
  observation and encoding.
- It was certified against the retired Python applier on 14,376 of 14,376
  replays.
- The engine checks it:
  - the scenario-init oracle, 28 of 28 scenarios;
  - the hidden-units oracle, 54 of 54 positions;
  - live games against the default AI, mirrored from the engine's log on
    the patched 1.18.8 (`tools/wesnoth_build/`).
- `OBSERVATION_EPOCH` is 11 and `CORPUS_VERSION` 5.

## Compute

| | desktop (RX 6600) | a 4090 box |
|---|---|---|
| self-play generation | about 470-570 decisions a second (about 3,450 games an hour in rounds 18-19, about 490 decisions a game) | about 2,900 decisions a second |
| a training step of `own2` | about 150 positions a second (fp16, HIP kernels) | about 1,480 |
| an 800-decisive-game match | about 15 min | about 17 min |

Sources: `docs/local_machine_use_20261029.md` and `docs/box_specs.md`.

- **The desktop** is free to use at any time; the user says when they want
  the GPU.
- **A box** costs about $0.42-0.58 an hour. The Vast balance read $81.18 on
  2026-10-22, and the budget is about $100 (user, 2026-10-09).
