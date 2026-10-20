# Near-tie coins: the attack known answer and batch 1 (pre-registration, 2026-10-20)

Branch `exp/near-tie-coins`. Box script `scripts/coin_batch_box.sh`. The
design is `docs/training_design_20261019/current_design.md` (round 8):
sections 3.1 (generation and coins), 3.2 (the label), 3.7 (invariants,
plants and controls) and stages S0(a), S0(c) and S1 of section 4. This run
fits no head and changes no network weight (user ruling 2026-10-15).

Tags as in the design: **[M]** measured (record named), **[D]** derived
(inputs named), **[U]** unknown before this run.

## What the build adds

- **Near-tie coins** (`tools/coin_decode.py`, `tools/raw_player.py`). At a
  decision of the raw player at argmax, the joint priors after the end_turn
  offset give the argmax a0, the runner-up a1 and their gap
  G = log p(a0) − log p(a1). If G < w, a coin D, 1 with probability P, plays
  a1 when it lands 1. The coin is SHA-256 of (game seed, mover's side, the
  mover's decision index) under its own domain, so it reads no prior and no
  dice. Every band decision goes into the game record: side, index, command
  index, G, P, D, a0 and a1. With w = 0 (the default) play and records are
  the decode's before the coins, bit for bit (`tests/test_coin_games.py`).
  A coin player's procedure tag is `+coin<w>p<p>`, so its games never pool
  with a reference match.
- **The attack coin.** One coin per game, P = 1/2, keyed by the seed, puts
  the attack offset (every attack's logit raised by x, the archived knob of
  tag `archive/feature-decode-knobs`) on player A (D = 1) or B (D = 0). The
  record and the result file name the side. Tag `+aoc<x>` on both sides.
- **The labels** (`tools/coin_labels.py`, numpy only):
  T_i = s_i ω_i (z − β·S), ω_i = (D_i − P_i)/(P_i(1 − P_i)), with z side A's
  result (capped games dropped and counted), s_i the mover's sign against
  side A, and S the seven luck sums of `tools/match_luck.py`'s walk. β is
  the least-squares fit of z on the seven sums over the 18,800 decisive
  games of `training/metrics/match_luck_20261019/rows.jsonl`
  (`tools/fit_luck_beta.py`, `configs/luck_beta_20261019.json`; in-sample
  Var(z − β·S)/Var(z) = 0.847). The proof that E[T_i | H_i] = Q(H_i, a1) −
  Q(H_i, a0) under the coin policy for any fixed β is in the module's
  docstring; `tests/test_coin_labels.py` checks it on synthetic games with
  known effects.
- **The checks.** I-coin (coins against P by chi-square, overall, per
  0.05-nat bin and in capped games; slopes of D − P on G and on the decision
  index); I-log (the record's command at each band decision is the coin's
  choice); the band census. By code separate from the label code
  (`tools/coin_plant.py`): the coin plant, z′ = z + Σ_j s_j(D_j − P_j)x_j with
  x_j = δ r_j 1[pair type of j = the commonest type], r_j a ±1 sign keyed
  by (game, side, index); and the dice control, the slope of z as the label
  code reads it on side A's hit-point luck sum.

## Stage A: the attack known answer (S0(a); S0(b) in round 7)

`own1` against itself (labels `own1a`, `own1b`, both at the reference
decode with 64 slots, tagged `raw:t0+eo-1.5+aoc1`) on the pool
configuration (`run_elo_batch`, 120 games in flight, six a worker process,
shared inference), the attack offset +1 on the player the attack coin
chooses; 6,400 decisive games from seed 500000.

| Readout | Prediction | Source |
|---|---|---|
| Hooked player's Elo (decisive games, 400 log10 of its score odds) | window [−111, −41] | [D] `analysis/round8_provenance_knob_checks.out`: the reduced-form slope −0.2178 ± 0.0487 z per nat of attack offset on unselected games, −76 ± 17 Elo, with this match's SE 4.34 at 6,400 games (`round7_synthesis_power.out` §1) |
| Attack coin's mean label | window [−0.640, −0.236] z | [D] same records: 2Δ, the coin moving the hook between the players (−0.436 ± 0.097 reduced form, −0.449 ± 0.092 decision-count fit; the window is the union) |
| Label SE | 0.023 z | [D] 2 √(0.847)/√6,400, ω = ±2 and the in-sample residual share above |
| elo_collect, A against B | 0, SE 4.34 (a report) | [D] each player hooked in half the games |
| Dice control | window [+0.00261, +0.00572] | [D] mean +0.00416 ± 0.00009, between-pair sd ≤ 0.00049 (`round8_invent_chance_ledger.out`), SE 0.00032 √(1600/n), at 3 sd for n = 6,400 (formula of `round8_synthesis_power.py` §2) |

What each outcome means:
- Both windows and the dice control pass: the per-seat hook reaches the
  seat its coin names, and the label path reads a known effect with the
  right sign through the result, the coin and the luck join. Stage B runs.
- A sign error in the shared path (the side recorded, z's orientation)
  reads about +77 Elo and +0.449 z. A hook that ignores the coin and stays
  on player A reads about 0 for the hooked player and about −77 for
  elo_collect's A against B.
- Hooked Elo in its window, label out: the label path (ω, s, β, the join)
  is broken. Label in, Elo out: the effect itself differs from the knob
  search's games; the label code is then not shown wrong, but the known
  answer is not confirmed either.
- The dice control out: z is mis-joined or flipped in the label path.
- Any of these holds stage B (`GATE_B_ON_A=1`, the design's rule for
  S0(a): fix before any coin game).

## Stage B: batch 1, the smoke first (S0(c) and S1's batch 1)

`own1` self-play as in stage A, without the attack offset, coins on both
sides at w = 0.35 and P = 0.25 [C, design 3.1]; 100,000 games from seed
600000 in matches of their own: 2,000 (the smoke), then 10,000 each. Every
match is walked, its coin rows and plant rows written and the readout of
the matches done rewritten (`readout_b.json`).

**The smoke is a gate.** Its own readout must pass every check at 3 SE
(the design's plant lines); otherwise the run stops after it and names the
failed check. Each check's false-alarm probability on correct code is
about 0.003 [D: two-sided normal tail at 3]; with nine checks, about 0.02
that one fires.

| Readout | Smoke (2,000 games) | Batch 1 (100,000) | Source |
|---|---|---|---|
| Band decisions per side-game | 76.1 | 76.1 | [D] 0.339 of 224.6 decisions per side-game (design 3.1, carried from rounds 5–6); the smoke re-measures it |
| Coin plant, slope per mover side | 1, SE 0.0387 | 1, SE 0.0055 | [D] SE √(3/G) for G games: per game the plant's variance is 3m²δ⁴ against mδ² of x², m the side's planted decisions (diagonal 1.33 m, same-side pairs 2m(m−1), cross-side pairs m·m′); the design's numbers (`round8_simplify_power.py` §4) |
| Plant with a wrong mover sign / a coin read one decision off | −1 / 0 | −1 / 0 | [D] the same expansion; the random signs r make the second 0 whatever the pair types' sequence (shown in tests) |
| Dice control | [+0.00245, +0.00587] | [+0.00268, +0.00565] | [D] as stage A; a flipped result reads about −0.00416 |
| I-coin (overall, bins, capped, slopes on G and index) | χ² ≤ 9 or within 3 SE | same | [D] the coins are drawn at P independently of G and the index |
| I-log: commands that are not the coin's choice | 0 | 0 | by construction; refused recruits and cut-short approaches are counted apart (unresolved) [U] |
| Capped games | 0 of 1,600 in own1 self-play | same | [M] design 2 (round 5–7 record); dropped and counted |
| Pooled mean label, SE | SE 0.0038 | SE 0.00054 | [D] σ² 4.48 per label (design 2: residual 0.84 × 1/(P(1 − P))), uncorrelated within a game (`round8_provenance_label_variance.py`), 152.2 labels a game |
| Mean label by 0.05-nat bin, by pair type, by mover | [U] | [U] | no record measures the near-tie advantage (design 6.1); a cell's SE is the pooled SE over √(its share), the shares measured by the smoke |

The label means carry no prediction: nothing on disk measures them. What
each outcome would mean:
- Pooled mean above 0 by 2 SE: on average the runner-up is the better
  choice at band decisions, a systematic error of the decode within the
  band. Below 0 by 2 SE: the argmax is better on average. Near 0: the band
  holds no first-order advantage on average; it may still hold a spread
  across positions, which only S1's head fit reads (design 3.5).
- A pair type whose mean is away from 0 by 2 SE names where a switch rule
  would act. The design states one sign in advance, as a report and not a
  gate: (non-attack argmax, attack runner-up) is expected negative, at most
  −0.0003 per switch at 2 SE (point −0.0096) [D, design 2,
  `round8_provenance_critique_checks.py` A].
- The bins read the mean's shape in G. S1's head gives each kind pair an
  antisymmetric constant and a margin slope (design 3.3), which puts a
  same-kind pair's mean at 0 as G → 0; a same-kind bin near G = 0 away from
  0 would contradict that form.
- A failed plant on the batch with a passing smoke: the label path breaks
  on data the smoke did not hold (a later match's join or rows); the
  labels are not used until it is found. A dice control that drifts across
  matches: a share of results joined to the wrong game.

The design's smoke also lists I-φ, the feature spectrum and the logging
overhead of the server's feature gather. This build logs no features, so
none of them is read here. The price with the coins' logging is read from
`matches.txt` (each match's wall and games) against the pool's steady
$1.912e-5 a game [M, `round7_provenance_price_tail.out`].

## Cost and wall

[D] Stage A 6,400 games and stage B 100,000 at the pool's steady
21,966 games an hour [M, `round7_provenance_price_tail.out`]: 17.5 minutes
and 4.55 hours; twelve matches' start-up and last wave, 36 s each in that
record, 7 minutes; the walks at the 2026-10-19 walk's rate (800 games in
9–11 s on 31 workers [M, `training/metrics/match_luck_20261019/walls.txt`])
22 minutes; the coin and plant rows, three passes over the records at about
1 ms a game (0.7 ms read on the laptop for a record of a decode-knob match
without coins [M, this branch]), 5 minutes; the readouts about 5; bring-up,
wheel and tests 20–25 minutes (the fp8 run's). In all about 6.0 hours,
$2.5 at $0.42 an hour, of which the games are $2.03 (106,400 × $1.912e-5).
On a host 1.41× slower [M, `round6_provenance_prices.py`] about 8.0 hours,
$3.4. `BOX_MAX_H` 9.5.

## Seeds

Stage A plays 500000 to 506399, stage B 600000 to 699999, a capped game of
slot i replaced by seed + k × 1,000,000. On 2026-10-20 no script, document
or `training/metrics` record names a seed base above 250000 (the
decode-knob confirmation, 250000 to 250799), and the lead's separate `own2`
match plays 300000 to 312799.

## Amendment before any result (2026-10-20, written at launch)

The run plays batch 1 at N_B = 60,000 games (`--env N_B=60000`), with
BOX_MAX_H 4.5 (`--env BOX_MAX_H=4.5`), on instance 56955777 (RTX 4090, EPYC
7702, $0.577 an hour), from stage `tier-b/staging/stage_20261020_coins.tar.gz`
built from 2eab835. The reason is the window's box budget: the user allowed
$5 in all, and the own2-base match (`docs/own2_base_prereg_20261020.md`) runs
alongside. Stage A and the smoke are unchanged. Batch 1's readings at 60,000
games, by the formulas above [D]:
- coin plant per mover side: SE √(3/60,000) = 0.0071;
- pooled mean label: SE 0.00054 × √(100,000/60,000) = 0.00070;
- dice control window: [+0.00267, +0.00565] (`tools/coin_plant.dice_window(60000)`).
If the box's hour cap ends batch 1 early, the readings are those of the
matches completed, each match of 10,000 games read whole, and the count is
reported.
