# Pre-registration: the dice's share of a match's result at argmax (2026-10-19)

Written before any box for it, on branch `exp/match-luck` (the tools
`tools/match_luck.py` and `tools/match_luck_readout.py`, the box script
`scripts/match_luck_box.sh`). The self-play design
(docs/selfplay_pg_design_20261015.md) waits on its number.

## Question

How much of the variance of a game's result, between players at argmax,
does an exact luck correction of the fights remove? At argmax both players
are deterministic, so once the map, the factions and the sides are drawn,
the dice and the recruits' traits are the only chance in a game.

The design needs the number twice:
- **Behaviour D** (argmax play with sparse sampled deviations) rests partly
  on "the game's other noise is only the dice and the other deviations, so
  the luck correction removes a larger share", and R0 predicts that under D
  the correction "cuts the variance by more, the size given by
  `exp/match-luck`'s measurement".
- **The luck pass** over records (its Q~ the cost-weighted material margin,
  a leader kill scored at the game's result, its scale cross-fitted) is in
  the build list. Whether it earns its place there depends on this share.

What is on record:
- 500 human ladder games (2026-08-17, docs/archive/credit_assignment_design_20260817.md
  "Q3 answered"): out of sample, HP luck explained 0.031 of the result's
  variance and cost-weighted kill luck 0.050; fights that could advance a
  unit were left out (16.9%).
- 36,209 playouts at temperature 0.5 from human midgame positions
  (docs/turn_value_prereg_20260925.md): HP and kill luck explained 0.072,
  fitted in sample.
- Neither is argmax play, where the players' own randomness is gone.

## The estimator

At each attack of either side in a recorded game, the Rust core enumerates
the fight's exact outcomes (`GameCore.attack_outcomes`, its advancement
branches merged back so the probabilities hold for the first-type
advancement the matches play). Each column Q gives the luck

    L = Q(the realized outcome) - sum over outcomes o of p(o) Q(o),

which has mean zero given the position and the attack. A game's luck S is
the sum of its L. The corrected result is G = z - beta . S, with z = +1 or
-1 for side A's win or loss.

The columns are those of `tools/match_luck.py`, read on the two fighters
from side A's view:
- `hp`: the HP margin;
- `material`: cost x hit points / maximum hit points;
- `material_nonterminal` and `terminal`: the board's material when the
  fight does not end the game, and +-1 when it kills a side's last leader;
- `value_50`, `value_100`, `value_200`: tanh of the board's material over
  that many gold, +-1 at a terminal outcome.

beta is fitted by least squares on the decisive games with two-fold
cross-fitting: each half of the games is corrected with the beta fitted on
the other half, so no game's own result enters the beta that corrects it.
The split is random, with seed 0.

### Where it departs from AIVAT, and why

AIVAT (Burch, Schmid, Moravcik, Morrill and Bowling, AAAI 2018; read in
its arXiv version 1612.06915v2) adds, for every chance event and every
decision of a player whose strategy is known, a term "expected value
before the choice minus the value after the observed one". It takes the
expectation across the imaginary observations of private information. Its
value function is fixed in advance; in HUNL they used self-play values of
a solved abstraction. There, the chance terms alone (MIVAT) cut the
standard deviation by 18%. With one agent's strategy known, AIVAT cut it
by 68%.

- **Only the chance terms (MIVAT's).** At argmax both strategies are
  deterministic, so AIVAT's action terms are identically zero. The dice
  are public, so there is no private chance to average over. Under
  behaviour D the deviations are drawn from a known policy, and their
  action terms are R0's to measure, as the design says.
- **Only the fights.** Three kinds of chance carry L = 0 and are counted:
  - the recruits' trait rolls (the material columns cannot see them: a
    trait changes a unit's maximum, not its share of it);
  - the advancement choice (none in these matches: the first type);
  - the setup draw (map, factions, leaders).
  AIVAT corrects the seat assignment as a chance event. Here sides
  alternate by game index, so the seat is stratified, not drawn.
  L = 0 keeps the estimator unbiased and removes less.
- **The scale is fitted, not fixed.** The columns are fixed functions of
  the state after a fight. Their weights beta are fitted, as MIVAT learned
  its value function from data (White and Bowling 2009). Cross-fitting
  makes the weights correcting a game independent of that game. AIVAT's
  Lemma 1 holds for any value function fixed before the game is read, so
  the estimator stays unbiased over all games. A weight fitted on the
  game it corrects would bias the mean and understate the variance.
- **A local value.** Q reads the two fighters and the rest of the board's
  material at the moment of the fight; it is not the game's value. The
  share it removes is a lower bound on what a better value would.
- **The PURE reading conditions on decisive games.** A capped game is no
  result, so the mean-zero guarantee holds over all games, not over the
  decisive ones when the dice decide which games reach the cap. The
  readout reports the decisive-game bias (`bias_t`) and the reading over
  all games with caps as draws, where the guarantee holds.

The realized outcome is read by applying the recorded command to a fork of
the rebuilt position, so its dice are the game's own. The tests check the
expectation against the simulator's dice on 1,500 seeds per case: a plain
fight, an advancement and a plague kill (tests/test_match_luck.py).

## Data

23 matches recorded whole, 2026-10-08 to 2026-10-12, every one played at
argmax on both sides: raw temperature 0 with one end_turn offset or
another, or the look-ahead player, which requires raw temperature 0. Every
game is on the 21-map Ladder pool with fog, both factions drawn uniformly.
The tarballs are on HF, pinned by SHA-256 in the box script (read from HF
2026-10-19, 123 MB):

| HF folder (tier-b/) | matches | decisive games |
|---|---|---|
| own_confirm_20261011 | own64_selfplay (A/A), own64_vs_p3, own64_vs_own0, own0_vs_p3slots0 | 4 x 800 |
| memory_own_round2_20261011 | gen_own1_selfplay (A/A), r1_own2_64_vs_own1_64, r2_own2_0_vs_own1_0, r3_own2_64_vs_own2_0 | 4 x 800 |
| memory_own_trajectories_20261011 | m1_own64, m2_own0 | 2 x 800 |
| decode_knobs_20261012 | confirm_kv15_vs_own1; kv00 (A/A, the null arm); kv10; kv15 | 800; 400; 1,200; 1,200 |
| lookahead_exposure_gate_20261012 | la_exposure_vs_own1 | 800 |
| lookahead_material_gate_20261009 | la_material_vs_parity3 | 800 of 991 |
| parity3_baselines_20261008 | selfpin (A/A), slots64_vs_slots0, slots64_vs_slots16 | 800 of 1,020; 800 of 810; 800 of 1,021 |
| memory_in_play_parity3_20261009 | a1_eo25, a2_reset, a3_eo35 | 3 x 800 |
| rollout_regrade_20261012 | own1_eo25_vs_own1 | 800 |

That is 18,800 decisive games by those runs' records, about 19,500 with the
capped ones, and 2,800 decisive in the four A/A matches. The walk counts
every game it reads; these counts are the records', not re-checked here.

## Estimand and predictions

**Primary:** the ratio Var(G) / Var(z) over the decisive games of the
pooled matches, with one pooled cross-fitted fit, for the control variate
`material_terminal` (the columns `material_nonterminal` and `terminal`:
the design's Q~). One minus it is the share the correction removes. The
readout gives a 95% bootstrap interval over games, with the cross-fitted G
held fixed.

Predictions (written before any walk):

| quantity | prediction |
|---|---|
| primary ratio, `material_terminal` | 0.85 (0.75 to 0.93) |
| `joint` (all seven columns) | 0.83 (0.73 to 0.92) |
| `hp` | 0.92 (0.85 to 0.97) |
| `value_100` | 0.86 (0.76 to 0.94) |
| the A/A matches' ratios against the others' (the pooled fit, by match) | within 0.05 |
| games of 1-15 turns against 26-35 turns | lower (more removed) |
| the decisive-game bias of the primary, pooled | within 2 standard errors of 0 |
| over all games, caps as draws, primary ratio | within 0.02 of the decisive reading |
| a won game's worth in gold (the primary's two weights' ratio) | 100 to 400 |
| each match's corrected Elo against its PURE Elo | within one PURE standard error |

## Barriers (a run that fails one is void, and nothing is read)

- Every column's luck, pooled over all games (capped ones included), has
  a mean within 4 standard errors of 0. A column outside that means the
  enumeration does not describe the dice the games were played with.
- Realized outcomes outside the enumerated support are under 0.1% of the
  attacks computed.
- The attacks computed (enumerated) are at least 90% of all attacks.
  Predicted: at least 99%. The counter weapon recorded is the core's own
  choice in every simulator game, and the DP's caps bind only on large
  berserk fights.
- Games whose record does not rebuild (rows with an error) are under 2%.
  Predicted: none.
- At least 18 of the 23 matches are walked whole and pooled. A match left
  out (unfetched, its walk failed, or not at argmax) is named in the
  report.

## Reading rule

The primary ratio decides. When its interval straddles a boundary, the
reading with the smaller effect holds.

- **Large: ratio at most 0.85.** At argmax the dice's first-order share is
  a sixth or more of the variance. In the design:
  - the luck pass stays in the build list and in R0's variance table;
  - D's prediction becomes "cuts the variance by up to 1 - ratio", the
    share before the deviations add their own noise.
  A luck-corrected Elo, as a companion to the PURE reading of every match,
  is proposed to the user: at ratio r a match reaches the precision of 800
  games in 800 r. It is a new estimand, so it is not a strength claim
  until the user rules.
- **Small: ratio between 0.85 and 0.95.** The luck pass stays in R0 (one
  record walk), and D's prediction is set to at most 1 - ratio. D's case
  then rests on its other argument: only the windows that hold a
  deviation train.
- **Negligible: ratio at least 0.95.** The luck correction leaves the
  design's build list and R0's variance table, and its decision record
  says why. At argmax the result's variance comes from the setup and from
  the dice's consequences that no linear correction reaches. Only a
  critic or more games reduce it.
- **The control variate:** if `joint` or a value column reads a ratio
  lower than the primary's by 0.02 or more, the design's Q~ becomes that
  set of columns.
- **The decisive-game bias:** if the primary's bias exceeds 3 standard
  errors pooled, the dice decide which games reach the cap. Any
  luck-corrected Elo then reads caps as draws, never decisive games alone.

Reported, not judged: every column per match, the pooled fit by match, by
game length and by map, and the counts of attacks not computed by reason.

## Box

One rental, CPU only (the walk replays games on the Rust core; any GPU):
16 to 32 effective cores of a Ryzen or EPYC (not a Xeon E5 v4, 3 to 4
times slower: docs/box_specs.md), at least 8 GB of memory, 20 GB of
disk. Searched 2026-10-19: an RTX 3060 host with a Ryzen 9 3950X, 16
cores, at $0.088 an hour, and RTX 4090 hosts with an EPYC, 32 cores, from
$0.42 an hour.

The steps:

| step | estimate |
|---|---|
| bring-up and the Rust build | about 10 minutes |
| the tests | 2 minutes |
| the downloads (123 MB) | 2 minutes |
| the walk | 10 to 25 minutes, one pool per match |
| the readout | 1 minute |
| the final upload | a few minutes |

The walk takes 0.2 to 0.8 s a game after a 10 s warm-up per worker,
measured on the laptop on 2026-10-15 and not measured again: about 1.1 to
4.3 CPU-hours for the whole set. The run takes 30 to 50 minutes:
$0.05-0.08 on the first class and $0.21-0.35 on the second. `BOX_MAX_H`
1.5 caps it, with the final round on top.

Each match is fetched against its pin and walked in a step of its own
(cut at 20 minutes, or after 10 minutes without a new row). A match that
fails is left out and the others go on; the readout reads only the matches
walked whole.

## Records

- On HF, `tier-b/match_luck_20261019/`:
  - `rows.jsonl`: one row per game, written as each game is walked. It
    holds the result's fields, z, each column's S and sum of squares, and
    the attacks' counts by reason.
  - `readout.txt` and `readout.json`.
  - `data.txt`: the tarballs fetched.
  - `matches_failed.txt`.
  - the step logs, `box.txt` and `status.txt`.
- Pulled into `training/metrics/match_luck_20261019/` once `ALL_DONE`
  lands; `rows.jsonl` stays on HF if it is over the pull's size cap.
- The readout runs again on the laptop from `rows.jsonl` with numpy alone
  (`python tools/match_luck.py readout rows.jsonl`). Readings beyond the
  ones above cost no box, and are not pre-registered.
- The result goes in a "Measured" section of this file, on `main`.

## Decision record

- Rejected: correcting the setup draw (a baseline per map, faction pair and
  side) in this measurement. It is not the dice the design asks about, and
  R0's critic baseline absorbs it.
- Rejected: one walk over every match in a single step. A crash or a hang
  in one match ended the run before the readout. A match cut partway holds
  its first games in name order, one side's first.
