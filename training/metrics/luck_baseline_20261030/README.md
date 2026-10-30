# The luck dealt so far, as a per-decision baseline (2026-10-30)

**Question.** The label R = σ_s(z − b − βᵀS) subtracts the game's whole dice
luck S (design §1 step 2). Stage 1 found that the value head's reading at
each decision, −α·v̂_t, cuts the label noise E[R²] from 0.880 to 0.643, and
so enters by the design's rule. That rule is a cut of at least 2 SE and at
least 5% of the luck-only noise
(docs/selfplay_t1_local_prereg_20261028.md, "Label noise"). The best
per-decision baseline, though, is

  E[R | h_t] = σ_s(E[z | h_t] − b − βᵀS_<t),

where S_<t is the luck dealt before decision t: the luck of every attack
whose command comes before the decision's. The value head reads
E[z | h_t] from a state that already carries S_<t, while R has had it
subtracted. So beside −α·v̂_t, a term in x_t = σ_s·S_<t should take out the
part of v̂_t that is luck.

Both terms are functions of the history before a_t. So the corrected label
keeps the gradient unbiased (design §1.1), and I-V checks this below.

**Answer: yes.**

- With both terms, E[R²] a decision is **0.596**, against 0.643 for the
  value head alone. That is a further cut of 0.047 ± 0.002 (z 24), 5.4% of
  the luck-only noise, which passes the rule.
- One coefficient along the labels' own β does as well as seven free ones,
  and its coefficients are stable across the two folds.

## Measured

`partial_luck.py`, on all 30,000 games of the local stage 1
(`C:\Users\momom2\wesnoth_runs\selfplay_t1_local`, 11,348,812 decisions), in
242 s on 10 CPU processes. The setup:

- The coefficients are fitted on one fold of games and read on the other
  (two folds by game index parity), as `tools/selfplay_readout.py`'s
  `label_noise` fits α and κ.
- E[R²] and its SE are a ratio over game totals.
- x_t is σ_s times the cumulative luck of the attacks whose command index
  is below decision t's `cmd`, by `tools/match_luck.attack_luck` on the
  core before each attack. It never includes decision t's own attack.

The table is in `partial_luck.json`:

| Label | E[R²] a decision | Cut from luck alone | Share | Coefficients (fold 0, fold 1) |
|---|---|---|---|---|
| luck alone | 0.8796 ± 0.0054 | | | |
| − α·v̂_t | 0.6427 ± 0.0053 | 0.2369 ± 0.0030 | 26.9% | α 0.747, 0.754 |
| − γᵀx_t (7 columns) | 0.8609 ± 0.0045 | 0.0186 ± 0.0013 | 2.1% | |
| − α·v̂_t − γᵀx_t (8) | 0.5954 ± 0.0042 | 0.2842 ± 0.0037 | 32.3% | α 0.829, 0.831; γ unstable |
| **− α·v̂_t − γ·βᵀx_t (2)** | **0.5957 ± 0.0042** | **0.2839 ± 0.0037** | **32.3%** | **α 0.833, 0.834; γ −0.871, −0.900** |

Over the value head alone, the luck term cuts 0.0473 ± 0.0019, 7.4% of the
value-corrected noise.

**The two checks:**
- The first two rows reproduce stage 1's readout (`readout.json`
  "label_noise": luck 0.8796 ± 0.0054, v 0.6428 ± 0.0053, α 0.744 and
  0.757). The small differences come from the folds: index parity here,
  the readout's own hash there.
- **I-V** (design §5) is each correction's per-game sum against
  1[a ∈ X] − p(X). Its mean must be zero within the gates' z of 3.45.

| Correction | end_turn z | attack z | recruit z |
|---|---|---|---|
| v̂_t | −1.28 | 1.21 | 0.21 |
| βᵀx_t | −1.26 | −0.65 | 1.01 |

The z's for v̂_t are the readout's own (`readout.json` "i_v" "v": −1.28,
1.21, 0.21). The luck term passes, so it is read before its event.

**Why the one-coefficient form.**
- Along β, γ reads about −0.89 in both folds. The theory says −1: a
  correction of +βᵀx_t gives back the luck that R subtracted and that v̂_t
  already sees.
- The seven free columns gain only 0.0003 of E[R²], and their coefficients
  change sign between folds (the luck columns are collinear: the value_50,
  value_100 and value_200 terms).

## What it changes

The gradient's noise per game scales with E[R²] (design §1.1, "Noise").
Measured against the per-side R that the pass applies today, both
corrections cut it to 0.596/0.880 = 0.68 of that. The value head alone cuts
it to 0.73.

At equal noise, a round then needs about 0.68 of today's games: 13,500 in
place of 20,000.

**Neither correction is built.** `tools/pg_pass.py` weights every decision
of a game-side by the same R (`w = R[(game, side)]`), and `tools/pg_labels.py`
fits no α. The design's corrections enter by the rule of §4, and stage 1 found
that v̂ enters. So stage C's labels should carry these corrections before it
runs.

Building them is work on `feature/pg-fp16`'s line:
- a fit of α and γ on the held-out games, frozen as b and β are;
- the luck so far computed in the pass's replay (tools/pg_positions.py),
  about 60 attack enumerations a game;
- per-decision weights in the pass.

    <cpu python> training/metrics/luck_baseline_20261030/partial_luck.py \
        --run C:/Users/momom2/wesnoth_runs/selfplay_t1_local --jobs 10 --out partial_luck.json

## Further history features: rejected by the rule (2026-10-30)

`value_features.py` tests three more baseline features, on the same 30,000
games, cross-fitted. Each is a function of the history before a_t:
- `v_opp`: minus the opponent's latest reading;
- `v_prev`: the side's own previous reading;
- `v_mean`: the mean of the side's readings so far.

Results are in `value_features.json`:

| Label | E[R²] a decision | Over v̂ + luck | Share of luck-only |
|---|---|---|---|
| v̂ + luck | 0.5958 | | |
| + v_opp | 0.5818 | 0.0140 ± 0.0002 | 1.59% |
| + v_prev | 0.5950 | 0.0008 ± 0.0000 | 0.09% |
| + v_mean | 0.5957 | 0.0001 ± 0.0001 | 0.01% |
| + all three | 0.5813 | 0.0144 ± 0.0002 | 1.64% |

The opponent's reading is a real gain (z 70), but at 1.6% of the luck-only
noise it is below the rule's 5% (design §4: "5% of about $70 of rounds is
$3.5"). The side's own earlier readings add nothing to v̂_t. **None
enters.** The rule prices a correction's upkeep; at 1.6% it does not pay
for itself.
