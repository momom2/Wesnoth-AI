# Match L: the capped recruit rule Bd in uncoined play (pre-registration, 2026-10-21)

Branch `exp/recruit-rule-l`. Box script `scripts/recruit_rule_l_box.sh`. The
design is `docs/training_design_20261019/current_design.md` (after round 10):
sections 1 to 4, rows "R0" and "Box A: L" of section 4, and
`round10_synthesis.md`. This is a measurement: no network weight changes,
and the rule is not adopted into any player.

Tags as in the design: **[M]** measured (record named), **[D]** derived
(inputs named), **[U]** unknown before this run.

## The question

Batch 1's 60,000 coin games of `own1` (`training/metrics/near_tie_coins_20261020/`)
show that at recruit near-ties between two different units, the runner-up
carries a learnable advantage. Capped at each side's first two switches a
game, a rule fitted to those labels has an exact value in the coin world,
where every other band decision of the game is coined at P = 1/4. L plays
the capped rule on one seat of `own1` against `own1` with no coin, and asks
whether that first-order value holds in uncoined play. With the rule's own
interactions measured near zero, a miss reads the shift from coined to
uncoined play.

## R0: the frozen rule and its seal (measured)

`tools/recruit_rule_fit.py` at commit 01049c2, run on batch 1's records
(the game tarballs and luck-walk rows of matches b00 to b06; their SHA-256
in the rule file's provenance). Readout:
`training/metrics/recruit_rule_l_20261021/recruit_rule_bd_13f8f00e4a7c_readout.{json,txt}`.

**The rows.** 514,281 band decisions whose two candidates recruit different
unit types, in 60,000 decisive games (4.29 per side-game). Each row's enemy
faction is read from its record's setup; the round-10 synthesis lacked it on
6% of rows. Labels T = s (D − P)/(P(1 − P)) (z − β·S), with β from
`configs/luck_beta_20261019.json` (the same β as `readout_b.json`).

**The rule Bd** (`round10_synthesis_checks.py`). The predicted advantage of
a1 over a0 for the mover is

    f = (w[fac|u1] − w[fac|u0]) + (m[fac|u1@enemy] − m[fac|u0@enemy]) + c (n0 − n1),

with n0, n1 the mover's earlier recruits of each type and a key the fit never
saw worth 0. It is a ridge fit with ridge λ on the unit and matchup columns
and 1 on c. The rule switches to a1 iff f > 0. It has 301 columns: 43 units,
257 matchups and the count.

**The ridge** minimises the design's progressive loss over the synthesis's
grid [M]:

| λ | 2,000 | 5,000 | 10,000 | 20,000 |
|---|---|---|---|---|
| progressive loss | −624.0 | **−743.6** | −700.1 | −588.6 |

**The frozen rule** is fitted on all seven matches at λ = 5,000:
`training/metrics/recruit_rule_l_20261021/recruit_rule_bd_13f8f00e4a7c.json`. Its SHA-256 is
`13f8f00e4a7ce0ce5a598adc628c5a2f9d5a08e8ead903e0e5e5eac70b3f4d44`, taken
over its content: units, matchups, count, ridges, band 0.35, cap 2 and the
formula's text. The decode refuses a file whose content does not hash to
its hash.

**The seal: +24.6 ± 3.0 Elo** [M]. This is the capped rule's exact
coin-world value. Each match's rule is fitted on the other six (leave one
match out, the condition of a fresh match). The value sums the labels of
each side-game's first two rule rows (f > 0). The second row is weighted 3
when the first one's coin played a1, and 1/3 when not. This is
`round10_provenance_r_pdis.py`'s identity, the performance-difference terms
of ranks 0 and 1, exact since those states do not depend on later ranks.
Units: Elo per side-game, 800/ln 10 per unit of mean result; SEs clustered
by game. Its parts:
- first order (unweighted) +23.9; interaction at rank 1 +0.6 ± 1.8;
- 1.34 switches per side-game; the frozen rule makes the same 1.34 in batch
  1's states;
- for comparison, the uncapped rule's first-order F is +29.2 ± 3.2 with a
  second-order term of −13.3 ± 4.7 over all ranks. That is why L plays the
  capped rule: its forecast has no second-order unknown.

**The differential check** [M]. The same code, with the enemy faction kept
only where the synthesis had it (0.940 of the rows), reads the synthesis's
numbers: progressive loss −586.7, −699.3, −658.4 and −555.3 at the four
ridges, and capped Bd +24.1 ± 3.0 at 5,000 (`round10_synthesis_checks.out`).
Adding the 6% of enemy factions moves the seal from +24.1 to +24.6.

**By mover faction** [M]. The faction's part of the seal (the parts sum to
it), then the faction's own value and switch rate per side-game it moved
(19,793 to 20,225 each):

| faction | part | own value | switches per side-game |
|---|---|---|---|
| Drakes | +3.8 ± 1.3 | +23.2 ± 7.9 | 1.53 |
| Knalgan Alliance | +2.0 ± 1.3 | +11.7 ± 7.6 | 1.41 |
| Loyalists | +1.7 ± 1.0 | +10.2 ± 6.0 | 0.98 |
| Northerners | +5.8 ± 1.2 | +34.9 ± 7.4 | 1.35 |
| Rebels | +2.6 ± 1.2 | +15.5 ± 7.0 | 1.25 |
| Undead | +8.7 ± 1.3 | +51.7 ± 7.9 | 1.52 |

**The turn-1 state shift** [M] (`round10_provenance_critique_checks.py`
section 2, on Bd's rows). Each side's first rule switch on its turn 1 comes
at 0.375 per side-game, with 1.9 earlier band decisions on average. These
labels are reweighted by 1[no earlier deviation by either side]/0.75^K to
the states of uncoined play before them. They read +11.89 ± 1.29 in the
coin world and +12.02 ± 1.81 at uncoined prior states. The shift is
**+0.13 ± 1.31**. The continuation after each switch stays coined in the
labels.

## What the build adds

- **The decode hook** (`tools/recruit_rule.py`, `tools/raw_player.py`).
  - The trigger: a decision of the raw player at argmax whose two best joint
    priors (after the end_turn offset) are recruits of two different types,
    with gap G under the rule's band.
  - The choice: the rule's player plays a1 iff f > 0, at its first two such
    switches of the side in the game, and a0 otherwise.
  - The inputs, read off the live game: the mover's and the enemy's
    factions from the state's sides, and the earlier recruits from the
    simulator's command history.
  - The log, per decision, in the record: side, decision index, command
    index, G, the inputs, f, whether it switched, the switches before it,
    a0 and a1.
  - A switch the engine refuses stays counted against the cap, as R0's rows
    count it. Batch 1 had 1 refused band decision in 9.3 million.
  - With the flag absent, play and records are the decode's before the hook,
    bit for bit (`tests/test_coin_games.py`, whose old-decode game also
    fails if the rule step is ever called).
- **The seat coin.** One coin a game, P = 1/2, is the SHA-256 of the seed
  under its own domain. It puts the rule on player A (D = 1) or B (D = 0)
  and is logged in the record's coins. `--recruit-rule PATH` through
  `run_elo_batch` and `elo_eval_game` tags both sides `+rr13f8f00e4a7c`, and
  is refused with near-tie coins or the attack coin: the match is uncoined.
- **Decode against R0, tested.** Five of batch 1's recorded coin games
  (`tests/fixtures/recruit_rule_batch1.game.jsonl.gz`) cover all six
  factions and 48 recruit rows, 13 of which played a1. They are replayed on
  the core. At every recruit band decision R0 reads off the record, the
  player's rule step is given the live state and the command history so
  far. Its logged inputs and command index equal R0's, and so does its f,
  as R0's design matrix times the frozen rule's weights computes it, to
  the last bit (`tests/test_recruit_rule.py`).
- **L's readout** (`tools/recruit_rule_readout.py`): extract per match,
  then a pooled readout, with the checks below.

## Match L

`own1` against itself, labels `own1a` and `own1b`, both at the reference
decode (`raw:t0+eo-1.5`, 64 memory slots), on the pool configuration
(`run_elo_batch`, 120 games in flight, six a worker process, shared
inference). The rule plays on the seat each game's coin chooses. The run
plays 37,800 decisive games in four matches of 10,000, 10,000, 10,000 and
7,800, from seed 700000.

| Readout | Prediction | Source |
|---|---|---|
| Hooked seat's Elo, the logit of its mean score (elo_collect's fit on the games relabelled by seat beside it) | **+24.6, window [+17.6, +31.5]** | [D] the seal ± 2√(3.0² + 1.79²); match SE 347.4/√37,800 = 1.79 (`round10_synthesis_budget.py` §1) |
| Null line | 3.57 | [D] 2 × 1.79; the seal sits 6.0 SE above it |
| Transfer ratio a null bounds | below 0.29 | [D] (3.57 + 2 × 1.79)/24.6 |
| P(lower bound > 0) at transfer ratios 0.3 and 0.5 | 0.98 and 1.00 | [D] the budget's formula at the seal |
| Hooked seat's Elo by its faction | report; coin-world own values above | [D] SE 347.4/√6,300 = 4.4 per faction at 37,800 games |
| Switches and rule decisions per hooked side-game, by faction | report; coin-world switches above | [U] in uncoined states |

The checks are those of the design's I-cap, I-key, I-match and I-side. Any
failure stops the run at the match that shows it:
- every match is read whole: each result has its record, and both tags and
  the result name the rule's hash and seat;
- each record's seat coin is the one its seed gives, recomputed by separate
  code;
- every rule decision is the hooked side's. Its inputs and f are recomputed
  by R0's code from the record (the setup's factions, the commands' recruits,
  the design matrix) and equal the logged ones, so the enemy key is the
  setup's;
- the switches recomputed from those f and the cap equal the logged ones.
  No side-game switches more than twice. Every switch played a1 and every
  other rule decision a0;
- the rule switches at least once on every faction;
- the seat coin is within 3 SE of 1/2, and seat against side A's side has
  χ² ≤ 9. Over the 37,800 planned seeds, the coin puts the rule on player A
  in 18,845 games (χ² 0.32), the seat-by-side χ² is 0.21, and the hooked
  seat moves first in 0.501 of games [D, computed from the seeds]. Capped
  games' replacements add a few seeds.

## What each outcome means, and the decision

From the design's row "Box A: L". The rule's own interactions at the cap
read +0.6 ± 1.8, and its turn-1 state shift +0.13 ± 1.31 (both R0).

- **Inside [+17.6, +31.5]:** first-order labels price uncoined play at a
  step whose interaction is measured near zero. S1 is proposed.
- **Below the window, lower bound above 0** (reading in (3.57, 17.6)): a
  partial link. With both measured near zero, the shortfall is the coined
  continuation. S1 is proposed; S2's capped screen reads the same link on
  `own2`.
- **Null** (reading ≤ 3.57; transfer ratio below 0.29): if I-cap and the
  label path are clean, coined continuations do not price uncoined ones. No
  round is bought. The user chooses among:
  - one coined side per game (2.0× games per label);
  - p 0.1 (2.08×);
  - Branch S, the reset, or Branch L.
- **Above +31.5:** reported.

The per-faction readings are reports. A faction whose reading falls far
from its coin-world value names where the link fails, but no decision rests
on it.

## Cost and wall

[D] From batch 1's box (RTX 4090, EPYC 7702, 31 usable cores, $0.577 an
hour; `training/metrics/near_tie_coins_20261020/`):

| Step | Time | Source |
|---|---|---|
| Entry to the first match (staging, wheel, tests) | 5.6 min | stages.txt, walls.txt: entry 21:42:39, GPU check 21:48:12 |
| 37,800 games | 99 to 101 min | 22,500 to 22,900 games an hour, start-up and last wave included: match a 6,400 games in 1,007 s, b05 10,000 in 1,601 s |
| Four extractions | at most 2 min | the coin rows took 25 to 28 s per 10,000 games |
| Four readouts | at most 3 min | readout_b took 45 s on 60,000 games' rows |

In all 1.83 to 1.85 hours, **$1.05 to $1.07**. The design's $0.98 counts
only the games, at the pool's $2.598e-5 a game. On a host 1.41× slower
(`round6_provenance_prices.py`) it is about 2.6 hours. `BOX_MAX_H` is 3.5,
1.9× the estimate. The rental's fund check then needs 5 hours at the
offer's price.

## Seeds

L plays 700000 to 737799 (four matches from 700000, 710000, 720000 and
730000), a capped game of slot i replaced by seed + k × 1,000,000. On
2026-10-21, no script, document or `training/metrics` record on `main` names
a seed from 700000 to 739999, nor one of those plus a multiple of a million.
The seeds taken are 300000 to 312799 (`own2`'s base match), 500000 to
506399 and 600000 to 659999 (the near-tie coins), and their replacements.
`N_L` is refused above 40,000.

## Launch

From a commit with a green CI run:

    python tools/stage_code.py --out /tmp/stage_20261021_rule_l.tar.gz \
        --script scripts/recruit_rule_l_box.sh --upload tier-b/staging/stage_20261021_rule_l.tar.gz
    python scripts/rent_box.py search --min-hours 5
    python scripts/rent_box.py create OFFER_ID --onstart recruit_rule_l_box.sh \
        --stage tier-b/staging/stage_20261021_rule_l.tar.gz --hours 1.9 --disk 60

Records go to `tier-b/recruit_rule_l_20261021`: `readout_l.json` and its
log, the matches' rows and walls; the games as `games_lNN.tar.gz`.

## Measured (2026-10-21)

Box 57053425 (RTX 4090, EPYC 9654, $0.551 an hour), stage
`tier-b/staging/stage_20261021_rule_l.tar.gz` (970ffa6), entry 12:44 UTC,
done 14:17 UTC, about $0.9. Records in
`training/metrics/recruit_rule_l_20261021/` (the games stay on HF).
37,820 games, 37,800 decisive, 20 capped, every slot played (four matches,
1,111 to 1,425 s each).

Checks, all clean: 158,621 rule decisions and 49,161 switches on the hooked
seat (1.30 per hooked side-game, against R0's 1.34), never more than two a
side-game; at every one the logged inputs, the rule's value and the switch
equal the frozen rule's, none played otherwise, none off the seat, none
refused. The seat coin put the rule on side A in 18,851 of 37,820 games
(χ² 0.37).

**Reading: `own1` + capped Bd beats `own1` by +15.3 +- 1.8 Elo** (mean z
+0.0441 +- 0.0051; elo_collect and the logit of the mean score agree). The
lower 2-SE bound is +11.7. The reading is **below the sealed window
[+17.6, +31.5] and above the null line 3.57**: the transfer ratio to the
seal (+24.6) is 0.62.

By the rule above, a partial link: the rule's own interactions and its
turn-1 state shift were measured near zero, so the shortfall is the coined
continuation (labels read with every other band decision coined at
P = 1/4; the match plays both sides uncoined). S1 is the next stage,
proposed to the user with its rulings; S2's capped screen reads the same
link on `own2`.

The readout's Elo by hooked faction mixes the rule's effect with each
faction's own strength against a uniformly drawn opponent; it does not read
the rule by faction.
