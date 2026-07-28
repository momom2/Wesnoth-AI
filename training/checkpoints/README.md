# Checkpoint naming — read this before picking one

**The LATEST checkpoint is the one with the newest DATE in its name, not
the one called "final".** On 2026-07-28 a `tier_a_campaign_final.pt` dated
2026-07-13 (decision_step 1.87M) was mistaken for the newest and used for
an eval; the actual newest was `tier_a_campaign_20260719.pt`
(decision_step 2.75M, ~880k steps further trained). That file has been
renamed to `tier_a_campaign_20260713.pt`. **Do not reintroduce "final",
"best", or "latest" in a checkpoint filename** — they age badly and lie.

## Rules

- **Name campaign checkpoints `<lineage>_<YYYYMMDD>.pt`.** The date is the
  save date. Ties inside a day get a `_hhmm` or a suffix.
- **Verify before trusting a name.** `decision_step` inside the file is the
  authoritative measure of how much training a checkpoint has had:

      python -c "import torch,sys; c=torch.load(sys.argv[1],map_location='cpu',weights_only=False); print(c.get('decision_step'), c.get('arch'))" training/checkpoints/X.pt

  `ls -lt` (mtime) is a good cross-check.

## Reserved filename — do NOT rename

`tier_a_campaign.pt` is **load-bearing infrastructure**, not just an old
local file. It is hard-coded as the live campaign filename in:

- `scripts/vast_onstart.sh` (`CAMPAIGN=`, and the HF seed download), and
- `scripts/hf_upload_loop.py` (uploads it, plus `.holdout`, to the Hub).

Renaming it locally breaks the box seed/upload loop. Leave it alone; the
local copy may be stale relative to the Hub.

## Current lineage (2026-07-28)

| file | decision_step | note |
|---|---|---|
| `tier_a_campaign_20260719.pt` | 2,747,117 | **newest campaign checkpoint** |
| `selfplay_local_20260718.pt`  | 2,299,999 | ladder-comparable to the above |
| `tier_a_campaign_5h_20260715.pt` | 2,290,529 | |
| `tier_a_campaign_20260713.pt` | 1,866,523 | was misnamed `_final` |
| `tier_a_campaign.pt` | (07-03 local copy) | RESERVED pipeline name |

Measured strength (see `docs/eval_20260728.md`): the 07-19 checkpoint beats
the 07-13 one 8-0-0 in the sim ladder, but is statistically indistinguishable
from the 07-18 one (3-1-4) — i.e. the last training leg shows no gain.
