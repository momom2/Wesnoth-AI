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
Measurement pending: the same per-phase evaluation with the count
gated, to bound how much of the seed's early-game AUC the leak
carries.

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

## Rules going forward

- The manifest is the only split. Any tool that trains on a corpus
  splits through `manifest_holdout_split`; a corpus without a
  manifest is not a training corpus.
- A number quoted on the imitation holdout names the checkpoint's
  lineage; a lineage that predates this review (the seed and its
  descendants) carries the finding 1 and 2 caveat.
- Matches are identified by `match_key`, not by file name, before
  anything is split.
