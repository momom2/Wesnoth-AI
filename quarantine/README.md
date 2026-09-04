# Quarantine (2026-09-03)

Everything built ABOVE the self-play training loop -- the search
variants, target links, auxiliary losses, anchors, memories,
grounding, trust regions, replay tricks -- is quarantined, not
deleted. Rules:

- The code stays where it is (moving it would break the tree the
  inventory has to read), preserved at git tag `pre-restart-20260903`.
- Nothing in this pile is imported by the new minimal loop
  (docs/archive/az_minimal_spec.md); the new entry point depends only on the
  simulator, encoder, model, and the core trainer step.
- INVENTORY.md (written by a subagent, reviewed by the user) lists
  every item with its evidence, so the pile can be sorted:
  keep / salvage-later / refuted / never-tested.
- Anything re-admitted comes back one item at a time, with a
  pre-registered prediction, approved by the user.
