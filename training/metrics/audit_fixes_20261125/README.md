# The audit's fixes: readings at and after the switch (2026-11-25)

Readings for `docs/audit_fixes_prereg_20261125.md`. The run switched to
the four changes at round 609; the switch's own files are in
`C:\Users\momom2\wesnoth_runs\audit_fixes_smoke`, and its scripts in
`C:\Users\momom2\wesnoth_runs\tools\audit_fixes_switch`.

| File | What it reads |
|---|---|
| `advance_choices.py` | The advancements a set of games held as choices (`["advance", k]` commands of a record), and the share that took a type other than the first. Standard library only. |
| `pointer_weights.py` | The weapon pointer's weights at a round: the keys' root mean square, and the queries' move from the switch's checkpoint (measurement 3). |
| `round_games.py` | The shape of a round's games from their summaries: outcomes, turns, turns ended for a side, turns left unplayed. |

| Output | Of |
|---|---|
| `advance_choices_smoke.json` | The smoke run's 48 games on the new code, before the switch. |
| `advance_choices_switch_match.json` | The switch match's 800 games (round 608's network on the new code). |
| `advance_choices_round_0609.json` | Round 609's 128 games, the first of the run under the new rules. |
| `pointer_weights_round_0609.json` | The pointer after its first step. |
| `round_games_605_to_609.json` | Four rounds before the switch and the first after it. |

All were run beside the training at Idle priority, on at most 2 threads.
