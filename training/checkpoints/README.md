# Checkpoints

## Quickstart

    python tools/reference_player.py --ensure   # fetch the reference player's checkpoint if missing
    python tools/reference_player.py --path     # its local path
    python tools/reference_player.py            # its record: label, checkpoint, decode, provenance

`--ensure` downloads from the Hugging Face repository
`momom2/wesnoth-model-checkpoints`, which is private: log in first with
`huggingface-cli login`, on an account with access to it.

## Which checkpoint is current

`configs/reference_player.json` names the reference player, one
checkpoint under one decode: since 2026-09-25 it is `obs8` at
`raw:t0+eo-1.5` (HF `tier-b/observation_retrain_20260924/arm_epoch0.pt`,
local `training/checkpoints/obs8.pt`). Every strength claim is a match
against it (CLAUDE.md, "Eval procedure"). The earlier reference players
and the rest of the lineage are listed in docs/checkpoint_naming.md.

## Names

docs/checkpoint_naming.md gives the scheme: a name encodes the
checkpoint's training history as a path through the lineage tree
(`2516k-b-294k-l4-430k`), and the named imitation checkpoints (`seed2`,
`relset`, `terrain`, `obs8`) are listed there with their HF paths. No
name says "final", "best" or "latest": such names age badly (on
2026-07-28 a `_final.pt` dated 2026-07-13 was taken for the newest
checkpoint). A checkpoint's `decision_step` counts its self-play
decisions and its `arch` gives its size:

    python -c "import torch,sys; c=torch.load(sys.argv[1],map_location='cpu',weights_only=False); print(c.get('decision_step'), c.get('arch'))" training/checkpoints/X.pt

`tier_a_campaign.pt` is the default campaign filename of
`scripts/vast_onstart.sh` (`HF_SEED_FILE`) and
`scripts/hf_upload_loop.py` (`CAMPAIGN_FILE`); renaming a local copy
does not change what those scripts read and write.

## What is tracked, and where the rest lives

Git tracks five names from before the Hugging Face era (`.gitignore`
lists them): `supervised.pt`, `supervised_epoch9.pt`, `sim_selfplay.pt`
and its `sim_selfplay_archive_*.pt` snapshots, 3-layer networks of
2026-05 and 2026-06, and `tier_a_5m.pt`, the 5M network that seeded
the 2026-07 tier-a runs. None of them is current. Every later checkpoint lives
on Hugging Face in `momom2/wesnoth-model-checkpoints` (each imitation run
under `tier-b/<run>/`, its checkpoint after the first pass
`arm_epoch0.pt`) and is copied here under a short name (`obs8.pt`,
`terrain.pt`, `relset.pt`), which git ignores.
