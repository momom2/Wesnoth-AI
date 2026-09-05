# Turn proposer and grader: the cheapest pipeline to a confirmed large-gap turn (design, 2026-09-05)

Inputs: docs/turn_gap_prereg_20260904.md (run 1 and the confirmation),
`training/metrics/turn_gap/{run1,confirm1}.json` re-read with plain
python (scripts in this scratchpad: `analyze_turn_gap.py`,
`analyze_turn_gap2.py`), docs/box_specs.md, docs/plan_20260904.md
sections 3-5, docs/selfplay_redesign_20260904.md, BACKLOG.md rulings of
2026-09-05, and the code cited by file:line below. Nothing here was
run on a box or through torch.

## 0. Summary and ranking

Cost per confirmed large-gap example (out-of-sample gap >= 0.25 at
P = 160, the prereg's estimand), ranked. "Validated" means replayed on
the recorded outcomes; "predicted" means derived from recorded costs
plus an unmeasured hit rate.

| rank | pipeline | screen $ | confirm $ | confirmed | $ per confirmed | status |
|---|---|---|---|---|---|---|
| - | as run: flat 40 playouts x 5 candidates; flat 160 x 5 on the 12 nominal | 0.90 | 0.84 | 3 | 0.58 | measured |
| 1 | A: sequential screen (rounds of 10, drop at upper 2-SE bound < 0.25); confirm base + selected alternative only, flat 160 | 0.58 | 0.29 | 3 | 0.29 | validated on the recorded outcomes (section 2.5) |
| 2 | A with a sequential confirmation capped at 80 per candidate | 0.58 | 0.11 | 2 (gaps 0.96, 0.38) | 0.35 | validated; loses the 0.28 example, keeps the ones worth distilling |
| 3 | A + shared inference server for the playout workers (1.48x measured for eval games through the same game loop, `_play_one_eval_game`; wiring turn_gap's workers to `tools/eval_inference_server.py` is new code) | 0.39 | 0.20 | 3 | 0.20 | the multiplier is measured, its transfer to playouts is not |
| 4 | 3 + Rust worker path (enumeration 6.9 -> 2.5 ms, masks 2.2 -> 0.76, encode 1.33 -> 0.4; about 1.3x on the worker's Python) | 0.30 | 0.15 | 3 | 0.15 | component numbers measured, end to end not |
| 5 | 4 + the "continue" edit proposer (1-3 deterministic candidates per position) instead of 4 samples | 0.24 | 0.10 | 2-3 | 0.11-0.17 | predicted from the blunder pattern of section 2.1 |
| 6 | 4 + K = 8 samples with sequential dropping | 0.42 | 0.20 | 4-6 | 0.10-0.15 | predicted |
| 7 | 4 + a forward-only pre-grader with residual SD <= 0.2 (grade the top 1-3 of ~30 edits) | 0.12-0.24 | 0.15-0.20 | 5-8 | 0.04-0.08 | conditional on section 6's measurement; the prior says the current head fails it |
| 8 | 7 + relevant-set tokens (2-3x on playout cost, needs the retrain) | | | | 0.02-0.04 | plan 1.4 |

Reading: $0.29 needs no new measurement, only two stopping rules and
a candidate filter in `tools/turn_gap.py`. $0.10-0.17 is engineering
already in flight (plan 1.3-1.5) plus one proposer measurement. $0.05
needs the pre-grader row, or two proposer rows plus the token cut; the
pre-grader is the one measurement that costs almost nothing (section
6).

Two findings from the files that the prereg's reading missed
(section 2.7): 12 of the 48 sampled alternatives were different turns
in the confirmation run than in run 1 (bf16 sampling is not
reproducible from a seed), so at positions 20, 49 and 57 the
confirmation graded a turn nobody selected; and position 20 carries a
matched alternative at +0.40 over the base on 160 fresh playouts, a
probable fourth large gap. The three confirmed positions are intact.
A pipeline must record command lists and confirm by replay.

The structural fact that sizes the whole direction (section 5): at
any efficiency in this table a rollout-graded turn searcher costs
$0.006-0.013 per side-turn, so an 800-game gate as a player costs
$50-300. Rule 2 (no distillation before a match win) cannot be
satisfied by this pipeline as a player. Its product is (a) the gap
distribution, already measured, and (b) confirmed (base, alternative)
pairs that validate a cheap grader. The player that gets matched is
the cheap one: edits + a boundary value net + one reply, about 600
decisions and $1.5e-4 per side-turn, an 800-game match for about $4
today and about $1 after the engineering multipliers. A ruling is needed on whether individually
confirmed examples may be distilled without a teacher match (section
5).

## 1. Recorded cost constants (run 1 and confirm 1)

| quantity | value | source |
|---|---|---|
| one playout | $7.5e-5 (13,300 per dollar); 11.4 process-seconds; mean 14.6 turns (median 13), 16.9 in the confirmation | `summary.wall_secs`, `provenance.jobs`, per-playout `turns` minus `turn_number` |
| per game-turn per process, 14 workers on a 17.5-core quota | 0.78 s (0.68 in confirm 1) | same |
| per decision inside a playout | about 38 ms per process (14.6 turns x 2 sides x ~10 decisions), $2.5e-7 at $6.5e-6 per process-second; the single-process benchmark is 17.6 ms: the difference is 14 processes contending for one GPU at batch 1 and 17.5 cores | docs/box_specs.md "Pipeline baseline" |
| one candidate turn (10-15 decisions) | 0.4-0.6 s process, about $3e-6: 25x cheaper than one playout, 1,000x cheaper than a 40-playout grade | same |
| one value read (encode + forward, batch 1) | 7.7 ms, about $5e-8 | box_specs baseline |
| one argmax opponent reply (~10 decisions) | about $2.5e-6 | same |
| outcome SD per candidate at P = 40 | 0.88 when the mean is within 0.3 of 0; 0.63 at 0.6-0.9; 0.19 above 0.9 | run1 outcomes |
| playouts per candidate to confirm a gap at 3 SE (SD 0.9) | gap 0.25: 233; 0.3: 162; 0.5: 58; 0.75: 26; 1.0: 15 | (3 x 1.27 / gap)^2 |
| playouts decided within t turns of the boundary | t = 5: 0.17; 10: 0.38; 15: 0.59; 20: 0.75; 30: 0.90 | run1 `turns`, `capped` |
| `fork_guard` per decision | free unless `SIM_FORK_GUARD=1` | tools/mcts.py:114-153 |

Proposal cost is never the constraint. Every design question below
is "hit rate per graded candidate" against "playouts per grade".

## 2. What the 60 recorded positions already answer (no box)

### 2.1 The three confirmed positions are two blunders and one find

Confirm 1's 160 fresh playouts per candidate (SE about 0.07 per
candidate; decisions per turn in brackets; `*` marks a sampled
alternative whose turn in confirm 1 differs from run 1's, section
2.7):

| position | base | alt0 | alt1 | alt2 | alt3 | alternatives >= base + 0.25 | kind |
|---|---|---|---|---|---|---|---|
| 4 (Hellhole, turn 7) | -0.28 [11] | +0.44 [16] | +0.29 [20] | +0.68 [15] | +0.64 [21] | 4 of 4 | base blunder: every sampled turn beats it by 0.6-1.0 |
| 59 (Hamlets, turn 9) | -0.74 [3] | -0.76 [6] | -0.35 [8] | -0.36 [6] | -0.72 [6]* | 2 of 4 | base blunder in a lost position: two sampled turns cut the loss by 0.4 |
| 18 (Basilisk, turn 5) | +0.24 [11] | +0.53 [10] | +0.09 [8] | +0.28 [16]* | -0.25 [8] | 1 of 4 | a find: one specific turn, same length |
| 20 (borderline) | +0.06 [9] | -0.16 [14] | +0.26 [16]* | +0.46 [13] | -0.09 [2] | 1 of 4 (alt2, +0.40, not the one run 1 selected) | blunder-like; run 1's selected turn (alt1, 10 decisions, +0.60 on 40) was never re-graded |
| 29 (borderline, 0.22 oos) | -0.71 [12] | -0.49 [11] | -0.72 [12] | -0.76 [15]* | -0.85 [13]* | 0 | a find, marginal |

Over run 1, positions by the number of alternatives at >= base + 0.25
on 40 playouts: 0 alternatives in 48 positions, 1 in 7, 2 in 3, 3 in
1, 4 in 1. Of the 7 "one alternative" positions (2, 3, 18, 20, 42,
49, 57), 1 confirmed (18) and 1 borderline (20); of the 5 "two or
more" (4, 9, 14, 29, 59), 2 confirmed (4, 59) and 1 borderline (29).
Small counts, but the direction is that a base turn beaten by several
different samples is the better screening signal, and it is cheaper to
detect: the pooled statistic mean(alternatives) - base uses 160
playouts against 40 and ranks 4 first at z = 5.5 and 59 sixth at z =
1.7 among the 60, while 18 is absent from its top 12
(`analyze_turn_gap2.py`).

The base's decision count on the confirmed and borderline positions
is 11, 3, 11, 9, 12 against 15-21, 6-8, 10, 13-16, 11 for the winning
alternatives. In 3 of 5 the winning turn does more than the argmax
turn. Over all 240 alternatives, T = 1 samples average 2.8 more
decisions than the base and 0.063 less value; the correlation of the
decision difference with the value difference is 0.05, but
alternatives with 4 or more extra decisions reach >= +0.25 in 12-15%
of cases against 4-9% for the rest. Weak and confounded; enough to
put a "do not end the turn yet" edit in the first proposer arm, not
enough to conclude the seed under-acts.

### 2.2 Combat luck in the turn or a different plan

Every candidate of a position shares one combat salt
(tools/turn_gap.py:148-149, :331-343), so each post-turn state is one
realization. Position 4 cannot be luck: four alternatives with four
different action sequences, hence different fights, all beat the base
by 0.6-1.0. Positions 18 and 59 cannot be separated from luck with the
JSON (one realization per alternative; 59 has two independent good
alternatives, which lowers the odds). The control costs about $0.01
per example: re-derive the alternative under a second turn salt and
grade it with 20 playouts against the base under the same salt
(section 4.4). The in-turn actions are not recorded (candidate dicts
at tools/turn_gap.py:269-276 hold seed, decision count, state key,
terminal flag), so "were the extra decisions attacks or moves" is
also unanswerable; recording the command list per candidate is a
few hundred bytes and should be added.

### 2.3 Whether the value head ranks the confirmed alternatives

Not answerable from the files: no value read is recorded. It is
answerable for about nothing on a box: the base turns and most
sampled alternatives reproduce from their recorded seeds and salt
(`alternative_seed` :156-158, `_candidate_turn` :264-277; section 2.7
on the ones that do not), so 300 post-turn states cost 300 side-turns
of argmax/sampled play (about 3 process-minutes), and each value read
is one forward (`boundary_value`, tools/turn_search.py:117-124).
Section 6 pre-registers it.

Prior for the answer. The seed's head is the A3 human-corpus fit on a
frozen trunk (docs/archive/credit_assignment_design_20260817.md:
215-222: holdout outcome AUC 0.824 at the level; the A4 counterfactual
probe separated real coordinate substitutions from placebo 6.9:1 at a
median delta of 0.079, two C51 atoms). Against that: its outcome AUC
on self-play by turn decade is 0.40 at turns 1-10 and 0.80 at 11-20
(docs/archive/teacher_arms_findings_20260829.md:295); its temporal
turn-scale movement scored AUC 0.527 (docs/archive/gbc_spec.md:47-58);
the benchmark positions sit at turns 2-25, most at or below 14. Within
a position the level bias cancels, so the ranking question is open,
but the expectation is a residual SD of 0.3-0.4 in outcome units,
which section 3.3 shows is not enough.

### 2.4 Shorter playouts as a grader

Outcome counted only if decided within c turns of the boundary, else
0 (a value head at the cap would replace the 0; how much it recovers
is the same unknown as 2.3):

| cap | cost factor | mean abs change of the position gap | within-position Spearman vs the full grade | out-of-sample gap at 4 / 18 / 59 |
|---|---|---|---|---|
| 5 | 0.32 | 0.18 | 0.08 | -0.01 / 0.01 / 0.21 |
| 10 | 0.58 | 0.14 | 0.56 | 0.38 / 0.06 / 0.29 |
| 15 | 0.76 | 0.10 | 0.69 | 0.76 / 0.26 / 0.48 |
| 20 | 0.88 | 0.06 | 0.77 | 0.82 / 0.28 / 0.41 |
| 30 (as run) | 1.00 | 0 | 1 | 0.96 / 0.28 / 0.38 |

Playouts are already short (mean 14.6 turns at temperature 0.5, 10%
capped at 30), so a cap buys at most 1.3-1.7x and a cap under 10 loses
the ranking. Cap 15 with the 0 rule is a reasonable default for
screening (position 18 survives at 0.26); it is not a large lever.

### 2.5 Sequential grading, replayed on the recorded outcomes

Screen (all 60 positions, outcomes taken in recorded order; the base
always keeps playing; rounds of n0 playouts per surviving candidate;
an alternative is dropped when gap_hat + z SE_diff < 0.25; the
position stops as a nominal hit when the best surviving alternative
has gap_hat >= 0.25 and gap_hat - z SE_diff > 0; budget 40 per
candidate):

| n0 | z | playouts | nominal hits | the 3 confirmed among them |
|---|---|---|---|---|
| flat 40 (as run) | - | 12,000 | 12 | yes |
| 10 | 2.0 | 7,790 | 12 (the same 12) | yes |
| 10 | 1.5 | 6,180 | 16 (4 extra false positives) | yes |
| 8 | 1.5 | 5,448 | 17 | yes |

Confirmation on (base, run 1's selected alternative) only, confirm 1's
160 fresh playouts in rounds of 20; confirm when gap_hat >= 0.25 and
gap_hat - 2 SE >= 0.10, reject when gap_hat + 2 SE < 0.25:

| position | verdict | playouts per candidate at stop |
|---|---|---|
| 4 | confirm (+1.15) | 20 |
| 59 | confirm (+0.57) | 40 |
| 3, 49 | reject | 20 |
| 9 | reject | 40 |
| 2 | reject | 80 |
| 14 | reject | 100 |
| 18, 20, 29, 42, 57 | undecided at 160 (gap_hat 0.11-0.28) | 160 |

Total 2,240 playouts ($0.17) against the run's 9,600 ($0.84): the
run graded all five candidates to get a secondary reading the
pipeline does not need. At 20, 49 and 57 the confirmation's
"selected alternative" was a different turn from run 1's (2.7), so
those three verdicts are about turns nobody selected; 4, 18 and 59
are intact. A gap of 0.28 (position 18) needs about 250
playouts per candidate to put its lower 2-SE bound above 0.10; gaps
at or above 0.5 confirm in 20-40. The confirmation bar should be set
by what a distillation target is worth, and the cheap examples are
the large ones.

Successive halving on run 1's outcomes (keep the top half of the
alternatives after 10 playouts, the top one after 20), judged
against the 160 fresh playouts over the alternatives whose turn is
the same in both runs (11 positions with at least two such): the
finalist is the 160-playout best, or within 0.06 of it, in 8 of 11;
the three misses (3, 9, 14) are positions where no alternative beats
the base by 0.25, so the ranking among near-equals is noise either
way. The flat 40-playout selection of run 1 picked the same
finalists. Selection on 20 playouts is as good as on 40 for gaps
>= 0.3.

The noise-corrected within-position SD of true candidate values is
about 0.15 (60 positions, 40-playout means) to 0.20 (the 12,
160-playout means): most sampled turns are close to the base, and the
large gaps are the tail.

### 2.6 What the files rule out

- Common random numbers across candidates' playouts: the salt carries
  the candidate index (tools/turn_gap.py:152-153), so playouts are
  independent by construction; the recorded correlation at equal
  playout index is +0.006. The panel measured CRN dead for atomic
  deviations (docs/selfplay_redesign_20260904.md, fact 5); whole-turn
  post-states differ more, so no measurement is proposed.
- The capped fraction (10-11%) and the temperature-0.5 playouts are
  not a validity problem; both runs satisfied the prereg's crash
  barriers.

### 2.7 Sampled turns did not all reproduce between the two runs

Comparing `n_decisions` and `sample_seed` per candidate across
run1.json and confirm1.json (the only cross-run comparable fields;
`post_state_key` is process-local, tools/turn_gap.py:271-273): all 12
base turns reproduced; 36 of 48 sampled alternatives reproduced; 12
did not (18 alt2, 20 alt1, 29 alt2 and alt3, 42 alt0 and alt3, 49 alt1
and alt3, 57 alt0, alt2 and alt3, 59 alt3), with decision counts
differing by 1 to 6. Same seed, same state, same salt: the sampling
draw `rng.choice(len(priors), p=p)` (tools/raw_player.py:41) flipped
on priors that differ at bf16 precision between runs (the confirmation
ran 12 workers instead of 14 on the same box; bf16 logits carry about
three significant digits, so a last-bit difference upstream moves p
by about 1e-2 and a draw over ~350 actions flips with about that
probability per decision, 10-30% per 10-16-decision turn). The
argmax turns are robust to it, as docs/box_specs.md found for the
eval harness.

Consequences. (1) At positions 20, 49 and 57 the selected alternative
was a different turn in the confirmation, so the prereg's
out-of-sample gap was not measured there; the three confirmed
positions (4, 18, 59) are unaffected, as are the confirmation's
headline counts. Position 20 carries a matched alternative (alt2) at
+0.40 over the base on 160 fresh playouts, a fourth large gap that
the pre-registered rule could not count. (2) A pipeline must record
every candidate's command list and confirm by replaying it
(`materialize`, tools/turn_search.py:311-330), never by re-sampling
from a seed. (3) Sampled proposals are not reproducible under bf16
inference; deterministic edits of the argmax turn are, up to argmax
near-ties. (4) The section-6 reconstruction will lose about a quarter
of the sampled alternatives; candidates are paired with a run's
playouts only when their decision count matches that run's record.

## 3. Proposers

### 3.1 Sampling (a)

- T = 1, best of 4: measured. Per-sample value -0.063 against the
  base; 3 of 60 positions confirmed (about 1% of graded candidates);
  mean out-of-sample gain of the best of four +0.05 per turn.
- Lower temperature: at T = 0.5 each sample plays as well as the base
  over a game (22-18, docs/box_specs.md "Raw player temperature"), so
  the mean of the best of four rises. The large-gap hit rate is a
  different quantity: blunders (4, 59) are found by any different
  turn; finds (18) need diversity, which T = 0.5 has less of.
  Unmeasured; one screening run ($0.58 at pipeline A) answers it.
- Top-k of each factored decision, diversity constraints: the joint
  prior over every legal action is already enumerated per decision
  (tools/raw_player.py:59-75 through
  `enumerate_legal_actions_with_priors`, wesnoth_ai/action_sampler.py:
  517-533), with `actor_idx`, `type_idx`, `target_idx`, `weapon_idx`
  on each `LegalActionPrior` (:491-507). Constraints (a different
  first actor, a different target, a different class) are filters on
  that list at no extra model cost. They belong in the edit proposer
  below rather than in a sampler.
- K = 8 instead of 4: the screen with sequential dropping costs about
  1.4x (bad samples leave at 10 playouts), and positions where one
  sample in four to ten is good become findable (P(at least one) 0.57
  against 0.34 at a 10% per-sample rate). Predicted hit rate 1.5-2x.
  Cheapest proposer change to test; it needs no new code.

### 3.2 Local edits of the argmax turn (b)

Enumerated from the base's decision spine. `record_spine`
(tools/turn_search.py:222-240) already records, per decision, the
pre-action fork, the chosen action and the full legal list with
priors; `materialize` (:311-330) replays a command list on a fork
under a salt, skips clean bounces and appends `end_turn`. An edit is
"decision j takes action a' instead of the argmax; argmax from j + 1
on". Candidate a' per decision:

| class | a' | count per base turn |
|---|---|---|
| continue | at the base's `end_turn`: the best non-end action, then argmax; 1, 2, 3 forced continuations | 3 |
| second choice | the top-2 (top-3) action by joint prior at decision j | 1-2 per decision, ~10-20 |
| other class | the best action of each class the argmax did not pick (attack / move / recruit / end_turn) | up to 3 per decision, mostly duplicates of the above |
| attack target | the same attacker's best other target; the same target from the best other hex; the other weapon | 1-3 per attack |
| move | the same unit's second-best destination | 1 per move |
| recruit | the second-best type on the same hex; the same type on the second-best hex | 1-2 per recruit |
| exact-combat attack (panel's surviving lever) | the top-2 attacks by expected material swing from `tools/combat_outcomes.py` (0.114 ms per call) where they differ from the prior's | 1-2 per attack |

About 30 edits per position before deduplication by post-turn
`state_key` (tools/turn_gap.py:344-350); expect 15-25 distinct. Cost:
30 edits x ~10 decisions = 300 decisions, about $1e-4 per position.
Grading them all is the problem: 25 x 26 sequential playouts = 650
playouts = $0.05 per position, $3 for 60. Without a pre-grader the
edit set must be small, so the first arm is the continue class alone
(3 candidates; pipeline row 5): it targets the blunder pattern of
2.1 and is deterministic, hence reproducible and re-derivable under
a new salt. The second-choice and target classes wait for the
pre-grader verdict or the engineering 3-4x.

Where the edits do worse than samples: a find like position 18 may
need two coordinated changes; one-deviation edits with argmax
continuation cannot express it unless the argmax continuation
happens to complete the plan. The panel's TCS probe measured
one-coordinate substitutions at a median accepted delta of 0.07-0.11
(tools/turn_search.py:1-12), i.e. mostly small gaps.

### 3.3 Value-head-guided proposals (c): how good the pre-grader must be

Setting: N candidates, m graded by playouts, one true large-gap
candidate g above the rest, pre-grader noise SD sigma_s per candidate
in outcome units. P(the target is in the top m of N) with N = 30,
m = 3, computed from the binomial count of nulls that outrank it:

| gap g | sigma_s 0.10 | 0.15 | 0.20 | 0.25 | 0.35 | 0.50 |
|---|---|---|---|---|---|---|
| 0.3 | 0.99 | 0.60 | 0.19 | 0.05 | 0.01 | 0.00 |
| 0.5 | 1.00 | 1.00 | 0.90 | 0.60 | 0.15 | 0.02 |
| 1.0 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 0.60 |

So a pre-grader that enriches 10x for gaps >= 0.5 needs sigma_s <=
0.2, i.e. the accuracy of a 10-20 playout grade (SD 0.9/sqrt(P) =
0.20-0.28), at a cost far below 10 playouts; a forward is 1/1,500 of
a playout, so any forward-based grader passes the cost side. For
0.3 gaps it needs sigma_s <= 0.15. The true within-position spread is
only 0.15-0.20 (2.5), so sigma_s = 0.2 corresponds to a correlation
of about 0.7 with the truth.

How to measure sigma_s on the recorded candidates: reconstruct the 60
candidates of the 12 positions (160-playout truth, noise SD 0.07;
up to 60 candidates after the matching of 2.7);
regress the grader's score on the true mean within position (one
intercept per position, one slope); sigma_s is the residual SD with
the truth's 0.07 subtracted in quadrature. Fifty to sixty points pin it to
about +-20%. Repeat on the matched candidates of all 60 positions
against the 40-playout means (noise 0.10-0.14) as a check. Three
graders from the same reconstruction: V(post-turn), V after one
argmax reply by the opponent (plan section 5's read point; needs an
argmax variant of `project_value`, tools/turn_search.py:167-211, which
samples from the priors), and post-turn material
(sum of `Unit.cost` x hp / max_hp, wesnoth_ai/classes.py:158, exact
and learning-free). Pass at sigma_s <= 0.2; kill at >= 0.3. The prior
(2.3) puts the current head at 0.3-0.4.

If the head fails, the pre-grader is the value net the plan already
schedules (own trunk, human turn-boundary states, docs/plan_20260904.md
section 5), and the confirmed pairs from this pipeline are its
validation set: its sigma_s on held-out pairs is the number that
licenses the pre-graded pipeline.

### 3.4 Panel ideas that survive the rulings (d)

- The exact combat-outcome enumerator as an edit generator (3.2, last
  row) and as the in-turn grader of realized combat: the expected
  post-turn material of an edited turn instead of one realization.
  No search-loop dependence, config-first.
- Certify-or-abstain (RCTC's shape): the base stands unless an
  alternative is confirmed. That is the confirmation stage.
- Human midgame splices as positions: the benchmark set already is
  one.
- Rejected by the rulings or the data: CRN pairing (measured dead),
  per-site R = 800 certification (2/delta^2 at delta 0.03; here
  delta >= 0.25 makes R = 15-60 affordable, which is the whole
  reason the turn frame works), anything conditioned on the MCTS
  loop, temperature as an object of study.

## 4. Graders

1. Sequential screen (2.5): rounds of 10, drop at upper 2-SE bound <
   0.25, stop at a nominal hit; 35% fewer playouts at identical hits.
   Implementation: replace the per-candidate loop in `_run_playouts`
   (tools/turn_gap.py:280-301) by a round-robin over surviving
   candidates; the salt scheme (:152-153) is unchanged, so a
   sequential run reproduces a flat run's outcomes prefix by prefix.
2. Confirmation on two candidates, sequential in rounds of 20 with a
   cap (80 for a distillation pipeline, 160-250 if 0.25-0.3 gaps are
   wanted). A `--candidates` filter on the confirmation run.
3. Pooled screen statistic alongside the best-of-K: mean of all
   alternatives minus base, z-scored. Free; detects blunders earlier
   (position 4 at z 5.5 on 40 playouts, about 2.7 at 10).
4. Cap 15 with the 0 rule for the screen (2.4); full cap for the
   confirmation. About 1.3x on the screen.
5. Playout temperature 0.5 on both sides, as ruled (decisive, no
   stalls; 22-18 against argmax).
6. Common random numbers across candidates: rejected (2.6).
7. A value head at the cap, or as the whole grader: conditional on
   3.3.
8. Fewer tokens per leaf, shared inference, Rust step and encode: the
   generic path, not this pipeline's; multipliers in section 0.

## 5. Selection bias, unbiased gain per distilled example, and the match problem

Bias sources and the fix for each:

- Screen selection (best of K on noisy means): the prereg's split-half
  (`split_gap`, tools/turn_gap.py:434-443) and the fresh-salt
  confirmation (`playout_offset`, :89-90, :287-288). A flat-size
  confirmation is unbiased for the selected alternative's gap.
- Optional stopping in a sequential confirmation: the point estimate
  at a data-dependent stop is biased upward by up to about one SE at
  the stop (0.10-0.15 at 20-40 playouts), which is not negligible
  against a 0.25 bar. Cheapest fixes: (i) decide on odd-indexed
  playouts, estimate on even-indexed ones: unbiased, and on the 12
  positions it cost 2,560 playouts against 2,240 (the decision runs on
  half the data, so a few more positions stay undecided); (ii) audit:
  a random quarter of confirmed examples gets 40 fresh playouts per
  candidate ($0.006 per audited example), which gives an unbiased
  aggregate gain over the distilled set at SE about 0.2/sqrt(n_audit).
  Use (ii) for the pipeline's headline (the aggregate is what predicts
  the teacher) and (i) only when per-example estimates are needed.
- In-turn realization (2.2): re-derive the alternative under a second
  turn salt (for an edit, the same edit rule; for a sample, the same
  seed until divergence, `materialize`'s bounce handling covers the
  rest) and grade it with 20 playouts against the base under the same
  salt: about $0.003-0.006 per example. An example that does not hold
  at >= 0.15 on the second realization is not a plan and is not
  distilled.

The match problem. Per searched side-turn, pipeline A spends about
130 screening playouts plus a fifth of a 190-playout confirmation,
about 170 playouts, $0.013; a game has about 31 side-turns for the
teacher's side (docs/box_specs.md, raw:t0.5 vs raw:t0 median 31
turns); an 800-game match costs about $320, $80 after a 4x from
engineering, $50 with the pre-grader. The credit is $61. So the
rollout-graded searcher cannot be gated as a player, and rule 2 as
written blocks distilling from it. Two readings, for the user to
rule on:

- R1 (recommended): this pipeline is an instrument. It produced the
  gap distribution and it produces confirmed pairs; the pairs
  validate a cheap grader (3.3, the plan's own-trunk value net after
  one reply). The teacher that is matched and distilled is edits +
  cheap grader + one reply, about 600 decisions per side-turn
  ($1.5e-4 at today's 38 ms per decision), an 800-game match for
  about $4 today and about $1 after the engineering multipliers. The
  rollout pipeline's job is then to
  make that grader's sigma_s measurable on positions with known gaps,
  a few hundred pairs at $0.1-0.3 each.
- R2: distill individually confirmed examples (each at >= 3 SE and
  re-realized) and gate the student by the 800-game match. The
  budget buys 200-600 examples at $0.1-0.3, about 1,200 at $0.05;
  each carries 10-20 atomic decisions of which a handful differ from
  the argmax, so 2-10k corrective atomic labels against a net fitted
  on 2.8M decisions (the panel's XOD arithmetic wanted 60-80k). If the
  corrections concentrate on "do not end the turn yet", the student
  learns a tempo shift, which the panel already routes to a gated
  config scalar. R2 is a weak bet at this credit; it is listed
  because it is the literal reading of the question.

## 6. Pre-registration draft: forward-only pre-graders on the recorded candidates

One factor: does a forward-only grader rank candidate turns against
the playout truth well enough to pre-grade (sigma_s <= 0.2)? Three
read-outs of one reconstruction, each with its own number.

Procedure. On a box already rented for phase-1 work (the job is a
few process-minutes; alone it is $0.17 of bring-up plus $0.02):
reconstruct the 300 candidates of run 1 from `configs/bench_states.json`
and their recorded seeds and salt (tools/turn_gap.py:148-158,
:264-277; `load_states`, tools/bench_pipeline.py:155-170), one process
per position as the tool runs. Per candidate record: the command list,
V(post-turn) from the mover's side (`boundary_value`,
tools/turn_search.py:117-124), V after one argmax reply by the
opponent (an argmax variant of `project_value`, :167-211; the reply is
played under the position's turn salt), post-turn material for both
sides (cost x hp/max_hp), the mover's material after the reply, and
the exact expected material swing of the turn's attacks
(`tools/combat_outcomes.py`, where the DP is defined; else the
realized value). Write the 300 rows next to run1.json. Nothing is
played out.

A reconstructed candidate is paired with run 1's 40 playouts when its
decision count equals run 1's record, with confirm 1's 160 when it
equals confirm 1's (both when both), and dropped otherwise (2.7:
about a quarter of the sampled alternatives; all bases match).

Estimands, per grader: sigma_s = residual SD of the grader's score
against the 160-playout mean within position (12 positions, up to 60
candidates, per-position intercept, one global slope, the truth's
0.07 subtracted in quadrature), with its standard error; the same
against the 40-playout means on the matched candidates of all 60
positions (attenuated; a check); the fraction of the three confirmed alternatives ranked above
their base; the AUC over the 240 alternatives of the grader's
within-position score for "alternative beats base by >= 0.25 on
playouts".

Predictions (before the run): V(post-turn) sigma_s 0.35 (0.25-0.45),
confirmed-above-base 2 of 3; V after one reply 0.30 (0.22-0.40), 2 of
3; material 0.30 (0.2-0.4), 3 of 3 on the blunders and 1 of 1 on the
find only if it was a material find; AUC 0.6-0.7 for all three.

Reading. Pass: any grader at sigma_s <= 0.2 with its 2-SE upper bound
below 0.3 and all three confirmed alternatives above the base: build
the pre-graded pipeline (row 7) and measure it as the next factor.
Inconclusive: 0.2-0.3. Kill: every grader at >= 0.3 or missing a
confirmed alternative: no forward-only pre-grader exists at this
head; the pipeline proceeds without one (rows 1-6), and the boundary
value net becomes the phase-2 prerequisite it already is in the plan,
with these 300 rows as its first held-out set.

Rider, same box, separate number ($0.01): re-realize the confirmed
alternatives of 4, 18, 59 under a second turn salt (the same sample
seed replayed under the new salt follows the recorded turn until the
first combat outcome differs; the reconstruction step above records
the command list first, so `materialize` can replay it from there)
and grade them with 20 playouts against the base under the same
salt.
Prediction: 4 holds at >= 0.5; 18 and 59 hold at >= 0.15 in 2 of 2.
Kill for an example: below 0.

Cost line: reconstruction 300 turns + 300 replies, about 6,000-9,000
decisions, under 10 process-minutes; rider 120 playouts; total under
$0.03 on a rented box, $0.20 if the box exists only for this.

The measurement after this one, whichever way it reads: the continue
edit proposer on the same 60 positions with pipeline A's grading
(60 x up to 4 candidates x ~26 playouts, about $0.5 plus $0.1 of
confirmation). Prediction: 4 and 59 confirmed from 1-3 deterministic
candidates each, 18 not; per-candidate hit rate 2x the sampler's.
Kill: 0 confirmed. Second-choice and target edits, and K = 8 sampling,
follow only behind a pre-grader or the engineering multipliers,
because their candidate sets are 5-8x larger.

## 7. Code needed (all in tools/turn_gap.py and a small helper; no simulator change)

- Round-robin sequential playouts across surviving candidates with
  the drop/stop rules as config (`--screen-rounds`, `--drop-z`,
  `--stop-z`, `--confirm-cap`); the flat mode stays as the default so
  the two runs on file remain reproducible.
- `--candidates` filter for confirmation runs; the odd/even estimate
  and the pooled-alternatives z in `summarize`.
- Per-candidate command list in the record (2.7), and confirmation
  runs that replay recorded commands through `materialize` instead of
  re-sampling from the seed; `--record-graders` for the section-6
  read-outs (value, one-reply value, material, expected swing).
- `--proposer {sample,continue,edit}` on `record_spine`/`materialize`,
  with the continue class first; edits deduplicated by `state_key`.
- The re-realization control as a `--rerealize` pass on a result file.

About 250-350 lines with tests, none of it on the training path.

## 8. Rejected, one line each

- rejected: CRN across candidates' playouts, because atomic CRN was
  measured dead and whole-turn post-states diverge more.
- rejected: playout caps under 10 turns as the grader, because the
  ranking correlation falls to 0.08-0.56 for a 1.7-3x saving.
- rejected: grading all five candidates in the confirmation, because
  the pipeline uses only the selected one (3x the confirmation cost).
- rejected: playouts at temperature 0, because argmax self-play stalls
  (17 of 40 capped).
- rejected: the rollout-graded searcher as a match player, because an
  800-game gate costs $50-300 at any efficiency in reach.
- rejected: fitting a value function from confirmed pairs alone,
  because the credit buys 10^2-10^3 pairs; the human turn-boundary
  corpus trains it and the pairs validate it.
- rejected: T = 0.5 sampling as the first proposer arm, because it
  costs a full screening run to learn a hit rate, while the continue
  edit tests the pattern the data already shows for 1-3 candidates.
