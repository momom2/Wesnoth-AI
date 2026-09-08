# Data contamination review (2026-09-08)

An adversarial review of every training and evaluation path (five
finders, two refuters per finding, Opus) asked one question: does any
number we quote rest on data the measured model had seen? This
document records what was found, what the code now does, and which
numbers keep a caveat. Lists of the affected games are in
`training/metrics/value_head/contamination/lists.json`.

## Findings

### 1. The seed's lineage trained on 108 of the 369 imitation holdout games

The imitation corpus (`replays_dataset_imitation/`, 17,104 games
before this review) takes its holdout by a hash of the source path.
The seed's 5M ancestor was trained on the older `replays_dataset/`
with the legacy last-300 split of `scripts/vast_onstart.sh` SL_MODE,
which knows nothing of that manifest. 114 imitation-holdout games have
a byte-identical twin (same command stream) in `replays_dataset/`, 108
of them in that pass's training rows.

Consequence: every holdout cross-entropy of the seed and of anything
initialised from it is measured on games its lineage partly trained
on. The 2026-09-05 relevant-set arms and the 2026-09-08 value-head
arms compare checkpoints of the same lineage on the same holdout, so
the comparisons stand; the absolute holdout numbers carry the caveat.
Strength is measured by matches, which the corpus does not enter.

Fixed for the future: SL_MODE trains on the imitation corpus with its
manifest split (`--imitation-config configs/imitation.json`); the
imitation holdout's command hashes are recorded
(`imitation_holdout_command_hashes.json`, 369 entries) for any tool
that reads another corpus.

### 2. The seed's value head saw about 364 of the 369 holdout games' outcomes

The A3 value head came from `tools/value_head_fit.py`,
`value_pretrain.py` and `value_finetune.py`, which split
`value_corpus_index.jsonl` themselves (shuffle by seed, first N held
out). That split puts about 364 of the manifest holdout games in the
training rows.

Consequence for the per-phase value study
(docs/value_head_study_20260907.md): the seed's head had seen the
outcomes of nearly every game it was scored on. The two retrained
arms (`value_head_plus_1`, `value_head_plus_material`) trained on the
manifest split, but start from that head. The clean subset below
bounds the effect.

Fixed: the three value tools and `tools/build_human_anchor.py` split
through `tools/replay_dataset.manifest_holdout_split`, which takes
the manifest's flag when a manifest exists (the old split stays for
corpora without one).

### 3. Global feature 5 was god-view under fog

The encoder's `their_villages` feature counted every village the
enemy owns. Wesnoth never shows a player an enemy side's village
count under fog or shroud (docs/wesnoth_rules.md, "Enemy side
statistics under fog or shroud"; `src/team.cpp:704-716`). The value
head could read the enemy's hidden economy, which a human cannot.

Fixed behind a checkpoint flag: `fog_hides_enemy_villages` counts the
enemy villages inside the mover's vision disc
(`wesnoth_ai/visibility.enemy_villages_visible_to`). The flag rides
the checkpoint, the eval server hello, the actor PLAY tuple and the
pre-encoder fingerprint, so the seed keeps the encoding it was trained
with and a new run opts in with `--fog-hides-enemy-villages`.

Measured (`training/metrics/value_head/arms_20260908/phase_seed_gated.md`):
the seed's head fed the gated count reads the same as with the true
count, and the village lead on its own is a weak predictor. Same-turn
AUC by turn bucket:

| turns | head, true count | head, gated count | village lead, true | village lead, seen |
|---|---|---|---|---|
| 1-5 | 0.647 +- 0.018 | 0.645 +- 0.018 | 0.580 +- 0.011 | 0.513 +- 0.013 |
| 6-10 | 0.767 +- 0.018 | 0.766 +- 0.018 | 0.672 +- 0.015 | 0.589 +- 0.017 |
| 11-15 | 0.818 +- 0.024 | 0.816 +- 0.024 | 0.745 +- 0.023 | 0.667 +- 0.025 |
| 16-20 | 0.838 +- 0.034 | 0.838 +- 0.034 | 0.794 +- 0.034 | 0.699 +- 0.037 |
| 21-30 | 0.874 +- 0.051 | 0.881 +- 0.050 | 0.679 +- 0.074 | 0.674 +- 0.073 |
| 31+ | 0.932 +- 0.056 | 0.932 +- 0.056 | 0.881 +- 0.087 | 0.888 +- 0.061 |

The leak carried nothing the head's ranking used. It stays gated
because a self-play player must not read it: a fresh network gates
by default (`TransformerPolicy`, the supervised trainer), a loaded
checkpoint keeps its own setting, and every pre-encoded cache (the
value corpus experiences, the human anchor, the policy anchor) is
built with the consumer's gate and refused under the other one.

### 4. Copies of one match under two names, one straddling the split

79 clusters (164 games) of the corpus hold the same match saved at
two turns or uploaded twice. One cluster had a copy in the holdout
and a copy in the training split (Hornshark Island, 52779/52789).

Fixed: `tools/replay_dataset.match_key` (first 200 commands, map,
starting units) identifies a match; `tools/build_imitation_dataset.py`
keeps one copy per match, the longest, holdout when any copy was; the
same pass over the existing corpus (`tools/dedup_corpus.py`) moved 85
copies to `replays_dataset_imitation_duplicates/`. The corpus is now
17,019 games, 369 holdout; the deduplicated tarball is HF
`tier-b/replays_dataset_imitation_dedup_20260908.tar.gz` and the
staging scripts point at it. The pre-encoded records
(`tier-b/replays_dataset_imitation_encoded_seedvocab_20260908.tar`)
still hold the 85 copies; the trainer reads only manifest games, so
they are inert.

### 5. Self-play midgame starts and the human anchor sampled holdout games

`tools/midgame_starts.py` cut its starting positions from
`replays_dataset/`, which holds the 114 twins; `tools/build_human_anchor.py`
sampled anchor states from the whole index. Positions from a holdout
game in the self-play stream do not carry its outcome, but they carry
its play.

Fixed: both respect the manifest split, and the midgame sampler skips
any game whose command hash is in the imitation holdout list,
whichever corpus it reads.

### 6. Minor

- `tools/player_ratings.py` fits ratings on every game's outcome,
  holdout included; the holdout is 2.2% of the fit. Not changed:
  the ratings are a sampling weight, not a target.
- MCTS opponent-turn nodes are encoded from the opponent's view of
  the state. That is how a searched player sees the opponent's
  options, not a leak; noted as a design fact.

## Per-phase value numbers on the clean subset

The clean subset is the 254 holdout games with no twin in
`replays_dataset/` and no copy in the training split. Same-turn AUC
of the head against the game's outcome, by turn bucket
(`training/metrics/value_head/arms_20260908/*_clean_subset.md`):

| turns | games | seed | plus 1 | control (2026-09-05) | material |
|---|---|---|---|---|---|
| 1-5 | 254 | 0.624 +- 0.023 | 0.633 +- 0.024 | 0.615 +- 0.026 | 0.571 +- 0.018 |
| 6-10 | 222 | 0.749 +- 0.024 | 0.758 +- 0.024 | 0.737 +- 0.025 | 0.683 +- 0.023 |
| 11-15 | 114 | 0.781 +- 0.035 | 0.769 +- 0.035 | 0.762 +- 0.035 | 0.737 +- 0.035 |
| 16-20 | 45 | 0.814 +- 0.048 | 0.833 +- 0.047 | 0.822 +- 0.049 | 0.830 +- 0.043 |
| 21-30 | 18 | 0.910 +- 0.039 | 0.960 +- 0.018 | 0.922 +- 0.041 | 0.865 +- 0.051 |
| 31+ | 7 | 0.987 +- 0.013 | 1.000 +- 0.000 | 0.961 +- 0.039 | 0.928 +- 0.034 |

On the full holdout the seed reads 0.647 in turns 1-5 and 0.767 in
turns 6-10; on the clean subset 0.624 and 0.749. The ordering against
material is the same on both: the head is ahead in the early and
middle game and level with material from turn 16. The
`value_head_plus_material` arm's numbers are added when it finishes.

## Did the extra iteration improve the head? (paired, per phase)

`tools/analysis/value_head_compare.py` pairs the two records by game:
per turn bucket, each game's same-turn score (the share of its turns
where the winner's value is above the loser's) under the seed and
under `value_head_plus_1`, and the per-game difference is the sample.
Brier is likewise a per-game mean. Records:
`training/metrics/value_head/arms_20260908/compare_seed_vs_plus_1.md`
and `compare_seed_vs_control.md`.

| turns | games | same-turn AUC seed | plus 1 | difference, 95% CI | p (paired t) | games better / worse / tied | Brier difference, 95% CI |
|---|---|---|---|---|---|---|---|
| 1-5 | 369 | 0.647 | 0.662 | +0.014 [-0.010, +0.038] | 0.24 | 91 / 86 / 192 | -0.018 [-0.023, -0.012] |
| 6-10 | 337 | 0.767 | 0.766 | -0.000 [-0.023, +0.023] | 0.99 | 55 / 52 / 230 | -0.026 [-0.033, -0.018] |
| 11-15 | 195 | 0.818 | 0.818 | +0.001 [-0.026, +0.027] | 0.97 | 20 / 18 / 157 | -0.025 [-0.035, -0.016] |
| 16-20 | 83 | 0.838 | 0.848 | +0.010 [-0.029, +0.048] | 0.62 | 10 / 8 / 65 | -0.021 [-0.034, -0.008] |
| 21-30 | 28 | 0.874 | 0.867 | -0.007 [-0.082, +0.069] | 0.86 | 4 / 5 / 19 | -0.019 [-0.037, -0.000] |
| 31+ | 8 | 0.932 | 0.943 | +0.011 [-0.016, +0.038] | 0.35 | 1 / 0 / 7 | -0.002 [-0.047, +0.041] |
| all | 369 | 0.729 | 0.736 | +0.008 [-0.009, +0.024] | 0.37 | 109 / 97 / 163 | -0.021 [-0.025, -0.016] |

The ranking did not measurably improve in any phase: every interval
of the same-turn difference covers zero, and most games score the
same under both heads (the winner already leads in most turns). The
calibration did: Brier falls in every bucket up to turn 30 with the
interval clear of zero, and the pooled AUC over all early-game states
rises by +0.009 to +0.042 (game-level bootstrap). The 2026-09-05
control arm reads the same way against the seed: same-turn
differences within noise in every bucket, Brier down in every bucket.
An iteration of the recipe sharpens the head's probabilities; it does
not change which side it thinks is ahead.

## Rules going forward

- The manifest is the only split. Any tool that trains on a corpus
  splits through `manifest_holdout_split`; a corpus without a
  manifest is not a training corpus.
- A number quoted on the imitation holdout names the checkpoint's
  lineage; a lineage that predates this review (the seed and its
  descendants) carries the finding 1 and 2 caveat.
- Matches are identified by `match_key`, not by file name, before
  anything is split.
- A new network never reads the hidden village count: the gate is on
  by default, and a cache built under the other gate is refused.
- The seed's lineage cannot be cleaned; a seed retrained from scratch
  on this corpus, with the manifest split and the gate on, is the
  clean reference and a comparison point (user decision 2026-09-08,
  timing open).
