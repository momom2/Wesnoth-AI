# The new inputs and the end of the attack from a distance (2026-11-25)

Readings for `docs/input_facts_prereg_20261125.md`. Nothing here trains
or changes a network.

- `reference_distance_attacks.py`: plays games of one checkpoint against
  itself at a given decode and counts the attacks it orders on an enemy
  its unit is not next to. Run on `main` at 0.35.6, before that attack
  left the legal actions.
- `reference_distance_attacks_16_games.json`: its output for the
  reference (own2 at temperature 0 with the −1.5 end-turn offset, 64
  memory slots, seeds 20261202 to 20261217, on the CPU): 0 of 1,660
  attacks, in 7,598 decisions. The simple loop's round-513 network
  ordered 8 of 2,637 (`../input_audit_20261124/summary_24_games.txt`).
- `new_weights.py`: the size of the weights that read the new columns
  (`unit_feat_proj2`), by block of columns, beside the old projection's,
  and how far they moved per round in units of the learning rate. The
  registration's measurement 3.

To read the new weights of the run's checkpoints:

    python -I -X utf8 training/metrics/input_facts_20261125/new_weights.py --lr 1e-5 <round_0544/pi.pt> <round_0560/pi.pt> ...
