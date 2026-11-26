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
| `round_checks.py` | A span of rounds' crash barriers (measurement 2): the exact check, the gradient's and the update's norms whole and by parameter group, the value head's cross-entropy, the pass's time. |
| `match_pairs.py` | Two matches set seed by seed: the same setup, the same player, the same outcome, and the correlation of the two outcomes on a seed. |

| Output | Of |
|---|---|
| `advance_choices_smoke.json` | The smoke run's 48 games on the new code, before the switch. |
| `advance_choices_switch_match.json` | The switch match's 800 games (round 608's network on the new code). |
| `advance_choices_round_0609.json` | Round 609's 128 games, the first of the run under the new rules. |
| `pointer_weights_round_0609.json` | The pointer after its first step. |
| `round_games_605_to_609.json` | Four rounds before the switch and the first after it. |
| `round_checks_593_to_624.json` | Rounds 593 to 608, round 609 and rounds 610 to 624, side by side. |
| `pointer_weights_round_0624.json` | The pointer at the first match from the switch. |
| `advance_choices_round_0624.json` | Round 624's 128 games. |
| `advance_choices_match_0624.json` | The 800 games of the match after round 624. |
| `match_pairs_0608_0624.json` | The loop's match at round 608 (old code) against the match at round 624: the same count of wins, 574, by coincidence. |
| `match_pairs_switch_0624.json` | The switch match (round 608's network on the new code) against the match at round 624. |
| `round_checks_609_to_640.json` | Rounds 593 to 608, 610 to 624 and 625 to 640, side by side. |
| `pointer_weights_round_0640.json` | The pointer at rounds 624 and 640. |
| `advance_choices_round_0640.json` | Round 640's 128 games. |
| `advance_choices_match_0640.json` | The 800 games of the match after round 640. |
| `match_pairs_0624_0640.json` | The matches at rounds 624 and 640 (574 and 618 wins), seed by seed. |

All were run beside the training at Idle priority, on at most 2 threads.
