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
| `match_line.py` | The matches from the switch as a mean and as a line through the rounds, each against the switch match, with standard errors (matches taken as independent). |

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
| `round_checks_609_to_656.json` | Rounds 593 to 608, 610 to 624, 625 to 640 and 641 to 656, side by side. |
| `pointer_weights_round_0656.json` | The pointer at rounds 640 and 656. |
| `advance_choices_round_0656.json` | Round 656's 128 games. |
| `advance_choices_match_0656.json` | The 800 games of the match after round 656. |
| `match_pairs_0640_0656.json` | The matches at rounds 640 and 656 (618 and 614 wins), seed by seed. |
| `match_pairs_0624_0656.json` | The matches at rounds 624 and 656 (574 and 614 wins), seed by seed. |
| `round_checks_609_to_672.json` | Rounds 593 to 608 and the four spans of sixteen rounds from 610 to 672, side by side. |
| `pointer_weights_round_0672.json` | The pointer at rounds 656 and 672. |
| `advance_choices_round_0672.json` | Round 672's 128 games. |
| `advance_choices_match_0672.json` | The 800 games of the match after round 672. |
| `match_pairs_0656_0672.json` | The matches at rounds 656 and 672 (614 and 608 wins), seed by seed. |
| `round_checks_609_to_688.json` | Rounds 593 to 608, 657 to 672 and 673 to 688, side by side. |
| `pointer_weights_round_0688.json` | The pointer at rounds 672 and 688. |
| `advance_choices_round_0688.json` | Round 688's 128 games. |
| `advance_choices_match_0688.json` | The 800 games of the match after round 688. |
| `match_pairs_0672_0688.json` | The matches at rounds 672 and 688 (608 and 625 wins), seed by seed. |
| `match_line_to_0688.json` | The five matches of rounds 624 to 688: their mean, and the line through them with and without the switch match. |
| `round_checks_609_to_704.json` | Rounds 593 to 608, 673 to 688 and 689 to 704, side by side. |
| `pointer_weights_round_0704.json` | The pointer at rounds 688 and 704. |
| `advance_choices_round_0704.json` | Round 704's 128 games. |
| `advance_choices_match_0704.json` | The 800 games of the match after round 704. |
| `match_pairs_0688_0704.json` | The matches at rounds 688 and 704 (625 and 623 wins), seed by seed. |
| `match_line_to_0704.json` | The six matches of rounds 624 to 704: their mean, and the line through them with and without the switch match. |
| `round_checks_609_to_720.json` | Rounds 593 to 608, 689 to 704 and 705 to 720, side by side. |
| `pointer_weights_round_0720.json` | The pointer at rounds 704 and 720. |
| `advance_choices_round_0720.json` | Round 720's 128 games. |
| `advance_choices_match_0720.json` | The 800 games of the match after round 720. |
| `match_pairs_0704_0720.json` | The matches at rounds 704 and 720 (623 and 616 wins), seed by seed. |
| `match_line_to_0720.json` | The seven matches of rounds 624 to 720: their mean, and the line through them with and without the switch match. |
| `match_line_0640_to_0720.json` | The six matches of rounds 640 to 720 alone: a flat line. |
| `round_checks_609_to_736.json` | Rounds 593 to 608, 705 to 720 and 721 to 736, side by side. |
| `pointer_weights_round_0736.json` | The pointer at rounds 720 and 736. |
| `advance_choices_round_0736.json` | Round 736's 128 games. |
| `advance_choices_match_0736.json` | The 800 games of the match after round 736. |
| `match_pairs_0720_0736.json` | The matches at rounds 720 and 736 (616 and 583 wins), seed by seed. |
| `match_line_to_0736.json` | The eight matches of the registered reading: their mean, and the line through them with and without the switch match. |
| `match_line_0640_to_0736.json` | The seven matches of rounds 640 to 736 alone. |
| `input_audit_round_0736_summary.txt` | Measurement 5: the input audit's readings (`training/metrics/input_audit_20261124/summarise.py`) on 24 games of round 736's network, seeds 20261202 to 20261225, played from the branch `fix/inprocess-advance-mask` at b3f9bb12. |
| `input_audit_round_0736_games.jsonl` | Those 24 games' results. |

The value estimate's reading on rounds 609 to 704 (`where_609_640.json`,
`where_641_672.json`, `where_673_704.json`) is with the earlier windows,
in `training/metrics/value_reliability_20261120/`.

All were run beside the training at Idle priority, on at most 2 threads.
