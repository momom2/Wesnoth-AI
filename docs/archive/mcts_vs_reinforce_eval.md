# MCTS vs REINFORCE eval

> **CAVEAT:** the checkpoint predates the C51 value head (pre-2026-05-10), so MCTS Q-values are biased toward a random-init head; 10/10 games ended in draws (likely both sides sitting still until `--max-actions` -- passive policy behavior is a known supervised_epoch3 limitation, see BACKLOG `Out-of-scope`). Re-run after the next training cycle produces an aggressive C51-aware checkpoint for a meaningful head-to-head number.

- **Checkpoint:** `training\checkpoints\supervised_epoch3.pt (pre-C51, partial load)`
- **MCTS simulations / decision:** 16
- **Games played:** 10 (target 10)
- **Avg turns/game:** 81.0
- **Avg wallclock/game:** 18.5s

## Overall

- **MCTS wins:** 0 / 10
- **REINFORCE wins:** 0 / 10
- **Draws / timeouts:** 10 / 10
- **MCTS win-rate:** 0.000  (95% Wilson [0.000, 0.278])

## Side breakdown (controls for side bias)

- **MCTS as side 1:** 0 / 5 = 0.000  [0.000, 0.434]
- **MCTS as side 2:** 0 / 5 = 0.000  [0.000, 0.434]

## Per-game log

| # | scenario | MCTS side | winner | turns | wallclock |
|--:|---|--:|--:|--:|--:|
| 1 | The_Walls_of_Pyrennis - Knalgan Alliance (Rogue) vs Knalgan Alliance (Trapper) | 1 | draw/to | 81 | 13.2s |
| 2 | Den_of_Onis - Loyalists (Longbowman) vs Knalgan Alliance (Dwarvish Stalwart) | 2 | draw/to | 81 | 28.9s |
| 3 | Hamlets - Knalgan Alliance (Dwarvish Stalwart) vs Drakes (Drake Flare) | 1 | draw/to | 81 | 22.0s |
| 4 | Clearing_Gushes - Knalgan Alliance (Dwarvish Thunderguard) vs Undead (Bone Shooter) | 2 | draw/to | 81 | 13.2s |
| 5 | Clearing_Gushes - Knalgan Alliance (Trapper) vs Loyalists (Javelineer) | 1 | draw/to | 81 | 16.0s |
| 6 | Ruined_Passage - Drakes (Drake Thrasher) vs Knalgan Alliance (Rogue) | 2 | draw/to | 81 | 21.4s |
| 7 | Sullas_Ruins - Drakes (Drake Warrior) vs Knalgan Alliance (Dwarvish Thunderguard) | 1 | draw/to | 81 | 14.4s |
| 8 | Swamp_of_Dread - Northerners (Troll) vs Knalgan Alliance (Rogue) | 2 | draw/to | 81 | 15.0s |
| 9 | Clearing_Gushes - Knalgan Alliance (Dwarvish Stalwart) vs Knalgan Alliance (Dwarvish Steelclad) | 1 | draw/to | 81 | 12.4s |
| 10 | Thousand_Stings_Garrison - Northerners (Troll Rocklobber) vs Knalgan Alliance (Rogue) | 2 | draw/to | 81 | 28.8s |