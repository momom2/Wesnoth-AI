# Quarantine (2026-09-03)

Everything built ABOVE the self-play training loop -- the search
variants, target links, auxiliary losses, anchors, memories,
grounding, trust regions, replay tricks -- is quarantined, not
deleted. INVENTORY.md lists each item with its evidence.

## Quickstart

The tree as the inventory read it is commit `8ddf052` (on `main`'s
history; the local tag `pre-restart-20260903` points to it and is not
on the remote). Read any file there, including one removed since:

    git show 8ddf052:tools/selfplay_worker.py
    git show 8ddf052:tools/sim_self_play.py | less
    git ls-tree -r --name-only 8ddf052 tools/

## Rules

- The code stays where it is (moving it would break the tree the
  inventory has to read), preserved at commit `8ddf052`.
- Nothing in this pile is imported by the new minimal loop
  (docs/archive/az_minimal_spec.md); the new entry point depends only on the
  simulator, encoder, model, and the core trainer step.
- INVENTORY.md (written by a subagent, reviewed by the user) lists
  every item with its evidence, so the pile can be sorted:
  keep / salvage-later / refuted / never-tested.
- Anything re-admitted comes back one item at a time, with a
  pre-registered prediction, approved by the user.
