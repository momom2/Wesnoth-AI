# Seed: devising the training algorithm and regimen (2026-10-19)

This file is what the design group starts from. Every round's files sit beside it in this folder.

## 1. The mandate (from the user, who decides)

Devise the training algorithm and regimen that takes the current player to beyond the strongest humans at Battle for Wesnoth (two-player multiplayer, the ladder pool). The user wants an innovative approach, an upgrade over the literature, not the beaten path.

Rulings that bind every proposal:
- **Learned only.** Deep learning on the project's own transformer and the exact simulator. No hand-written strategy, evaluation heuristics, scripted tactics or symbolic game knowledge beyond the rules. No language model is fine-tuned; methods from other fields are borrowed, not their models.
- **Strength first.** Readability and steering are not goals for now.
- **Self-play is part of the end state.**
- **The network picks actions, never whole turns.** A turn's later actions are conditioned on the dice of its earlier ones (combat outcomes), so a plan is adapted on the fly. Any comparison of choices must compare single decisions, each followed by play that re-decides after every roll. Comparing fixed action sequences, or turns already played with their own dice, is invalid.
- **No fire-and-hope.** Past months ran many runs that ended "no signal"; almost every improvement since August came from bug fixes. Every stage of a proposal must predict, before it runs, the number its readout should show, so a miss points at the stage that failed. Every training code path must pass a planted-signal check first: a training run on a modified game where the right answer is known, recovering a signal of known size at the predicted rate.
- **Correct by design.** The math and its invariants come before code; tests derive from the spec.
- **The acting player uses only its own observation** (fog of war). Components used only in training (a critic, a teacher, a grader) may see the full state.
- **The network keeps a memory** across decisions within a game; its size is open.
- **Budget.** About $85 of rented GPU time now (an RTX 4090 at about $0.42 an hour). On the order of $1,000 is conceivable for one run whose case is strong. The user approves every box. No training runs before the user agrees to the plan: this group designs, it does not run.
- **The yardstick.** Every strength claim is a match against the fixed reference player `own1` (800 decisive games, one standard error about 12 Elo), never against the current self.

## 2. The game and the assets

- Two players alternate turns on a hex map: 21 maps, 6 factions, drawn uniformly. A turn is a sequence of actions by one side: recruit on castle hexes (gold from villages), move units (movement points, terrain costs, zones of control), attack adjacent enemies (weapon choice; the defender strikes back), end the turn.
- Combat is stochastic: each strike hits with a probability set by the defender's terrain; damage depends on weapon, resistances, a day/night cycle. Units gain experience and level up. Recruits get random traits.
- Fog of war: each side sees only what its units see.
- The game ends when a side has no leader, or at a turn cap. About 20 turns, 9 to 11 actions per side per turn, about 450 decisions a game, one win/loss result.
- **The simulator** (Rust core, bit-exact with the real engine): fork a state in 0.02 ms, apply an action in 0.05 ms, the exact outcome distribution of an attack in 0.06 to 0.11 ms.
- **Generation:** about 17,000 self-play games an hour on one 4090 (120 games in flight, the network batched on a shared inference server); so about 34,000 playouts from mid-game to the end an hour.
- **Training:** about 1,500 positions a second on one 4090 (packed trunk, compiled).
- **Human data:** 14,376 replays, exactly reconstructed. The imitation corpus trains on the winning side's decisions only, about 4 million positions an epoch.
- **The network** (`own1`): a transformer, d_model 384, 8 layers, 15.6M parameters, over tokens for units, hexes and recruit options of the side's observation; a 64-slot memory carried across decisions; heads: factored action logits (which unit or recruit slot acts or end_turn; the action kind; the target hex; the weapon), a distributional value head (51 atoms), a belief head (hidden enemy units).
- **Play:** argmax at each decision, with the end_turn logit lowered by 1.5 (a hand-set offset the user wants replaced by something learned).
- The code: Python (`wesnoth_ai/`, `tools/`), the Rust core (`rust/wesnoth_core/`), the trainer (`tools/sequence_train.py`), the match driver (`tools/run_elo_batch.py`). Reading the code to ground a design is fine. The older plans under `docs/` are history; do not adopt their framing.

## 3. The record (measured results)

Elo from 800-decisive-game matches, one standard error about 12.3. "Argmax" is the reference decode.

How the current player got its strength:
- Every reference so far comes from imitation and fixes to it. Lineage gains: a better encoding of the relevant units +56; the hex's terrain set plus the end_turn offset +263 (the offset alone about +229 at decode); a corrected observation +73; observing only what a player sees plus the 64-slot memory +191; an annealed learning-rate schedule +82; training on its own recorded games toward its memoryless reading +88.6.
- The memory, trained only to predict humans from their histories, cost 105 Elo in play (causal confusion: it continued its own history's patterns, ending turns four times as often). Training on its own trajectories removed most of the cost; a second round saturated (+16.5, not significant).
- Imitation's remaining gain: 0.061 of holdout loss per further epoch at the peak rate.

The end_turn mechanism: the actor head puts end_turn's whole probability on one entry while the probability of acting is split across units. Taken jointly, end_turn's mass competes against each single four-way action and wins too often. A decode rule that compares end_turn's mass with the largest unit's total mass gained +193 +- 14; the offset -1.5 gains +229 +- 15.

Self-play and search:
- Nothing produced by self-play has beaten any reference.
- An AlphaZero-style loop (search over atomic actions, distilling visit counts plus results into the shared trunk): the targets were the prior or a truncation of it, and the value gradient carried 94-99% of every update through the shared trunk.
- The seed checkpoint at argmax beats the same weights sampling at temperature 1 by +412 +- 97. Gumbel-MCTS with 32 simulations over atomic actions, leaves read by the value head, loses to argmax by -124 +- 58.
- Policy-gradient self-play on a current network was never measured; a design was withdrawn on analysis (the tilt toward the played action too weak; the advantage mostly the value head).
- Common random numbers across search branches were measured useless.

Evaluators:
- Ranking 5 candidate turns at 199 human positions against playout truth (corrected within-position correlation): the value head 0.22, a fitted linear grader 0.27, the hit-point margin after the turn 0.40, rollouts read four turns ahead 0.53.
- Learned critics (up to the policy's size, up to tens of thousands of self-play games, at most 24 positions per game sharing one result, so the effective sample was about the game count): on the true state about 2 standard errors better than material but failing the correlation test; on the player's own observation no better than material; more games of the same kind did not help.
- One-step look-ahead with a material evaluator: -8.7 +- 12.3 (it mostly rewarded recruiting); with gold and exposure added: -8.7 +- 12.3.
- Selection among candidate turns by rollouts on the true state: +0.049 +- 0.017 result units (win +1, loss -1) over the base turn; from the mover's own view with hidden units deleted, -0.002 +- 0.018; in belief worlds, true state +0.033 +- 0.016 and knowing the hidden units +0.015 +- 0.015. **Suspect:** the candidates were turns already played with their own dice, the truth subtracted only a linear estimate of that luck, and the playouts continued at temperature 0.5, not argmax. Part of the gain may be selecting the turn whose dice went well. Do not build on it without re-measuring per decision.
- Decode-time logit offsets (attack, recruit, time of day) tuned against matches: +1.7 +- 12.3, flat.

Luck (measured 2026-10-19): over 23 argmax matches (18,800 decisive games), a cross-fitted AIVAT-style correction of every attack's outcome leaves 0.940 (0.928-0.957) of the result's variance with a material-plus-terminal correction and 0.848 (0.838-0.859) with seven columns. This is what those linear corrections capture, a lower bound on the dice's share: at argmax both players are deterministic, so with map, factions and sides fixed, all of a result's variance is chance.

## 4. The lead's current proposal (to be critiqued, revised or replaced)

- **Labels from forks, per decision.** At a decision point, fork the game, apply each option, roll fresh dice, let the policy play on to the end (re-deciding after every roll). The label is the result's difference between options. One playout costs about $0.00001; two options x 8 playouts about $0.0002; 10,000 decision points about $2. The cost that matters is how many labelled decisions the policy needs.
- **First instance: a learned stop rule** replacing the -1.5 offset. At each decision: end now, or take the best non-end action and continue. A small head on the frozen trunk predicts the difference from the observation. Kill: it must predict held-out labels better than the best constant offset. Known answer built in: averaged over positions, it must imply an offset near -1.5.
- **The labels price the future under the current policy** (both sides, all later decisions), so they go stale once the policy changes. Fixes: iterate (approximate policy iteration: regenerate the labels with the new rule inside the playouts, retrain; Lagoudakis & Parr 2003; Scherrer et al. 2015), and keep each step small (conservative policy iteration, Kakade & Langford 2002: a stale label's error is proportional to how far the policy has moved), so each round's prediction holds. Later, bootstrap from a learned value of the next position instead of re-labelling, which needs a value the project has not managed to learn.
- **A grader** trained on paired fork labels (two options from one position cancel the position's own value) to amortize the playouts. AlphaGo's value network approached the accuracy of rollouts with its strong policy at about 15,000x less compute, after training on 30 million positions each from a different self-play game, and mixing it 50/50 with fast rollouts beat either alone. At this project's prices a million single-playout labels cost about $12.
- Each round is scored against `own1` and predicts its own gain from its labels (the performance-difference lemma: the gain equals the sum over the new policy's own decisions of each decision's advantage under the old policy).

## 5. Earlier ideas (material, not a starting frame)

From four independent memos earlier today, after testing against the record:
- Killed: luck-corrected returns as the main lever (the dice's measured share was too small for their thresholds); critics trained on game results; one-step advantages on single actions with a critic; outcome-driven policy gradient (the user's ruling and the arithmetic: about 19,000 games per decision context at a 2-point effect).
- Alive: policy iteration with playouts as the evaluator and a privileged (full-state) training-only teacher distilled into the observation-only policy; a grader trained on within-position paired labels, graded on the 199-position benchmark against 0.22 / 0.27 / 0.40 / 0.53; dense prediction targets (the next observation, the final board) as auxiliary heads with a capped gradient share; imitation over both sides conditioned on a pre-game rating (needs player names in the replays, unverified); a belief sampler of hidden units for decision-time search, oracle first; starting playouts from mid-game positions; mixed-outcome positions as the useful ones (gains sit in about 12% of positions).
- Questions raised: whether batch-dependent GPU arithmetic makes argmax games non-deterministic (a second source of randomness); how many labelled decisions a policy of this size needs to move; strategy fusion when a fogged student copies a full-state teacher; exploitability, which Elo against one reference does not show.

## 6. The user's feedback after round 4 (2026-10-20), binding from round 5

On round 4's claim that every stage has a predicted number, a kill line and a planted-signal check, the user wrote: "That's fire-and-hope. Pulling numbers out of one's arse then pikachu-surprised-face if it doesn't work." And: keep iterating.

What this requires of every proposal, critique and synthesis from round 5 on:
- **Every number has a provenance.** Each number that a decision, a stage order, a budget or a kill line rests on is one of: *measured* (name the record file and how it was read), *derived* (show the derivation from measured inputs and stated theory), or *unknown*. A guessed range ("prior 0.2 to 0.8, not derived", "rho_C 0.55 to 0.75") is not a prediction and may not carry a decision.
- **Unknowns are measured before anything relies on them, or nothing relies on them.** Order the work so that no costly stage depends on an unmeasured number. Prefer procedures whose correctness does not depend on the unknown: improvement adopted only when data certifies it, stages where every outcome teaches something specific.
- **Measure from what exists first.** Records already on disk (whole game records beside match results, the luck walk's `training/metrics/match_luck_20261019/rows.jsonl`, `training/metrics/critic_step1_20261009/`, the replay corpus) can settle many unknowns now. Agents may run light single-process Python analyses on existing records: read-only, numpy and the standard library only (no torch import, no network forward pass, no simulator, no GPU, no training, no pytest), each under about a minute of CPU. Scripts and outputs go in `analysis/` in this folder, and every number reported from them names the script that produced it.
- **A kill line states what a result would mean**, derived from what is known; it is not a guess with a deadline. When a stage's outcome cannot be predicted, say so, and design the stage so that every outcome is informative.
- **Complexity is a cost.** The project's main failure mode has been bugs. Every mechanism must earn its place, and between two designs with the same guarantees the simpler one wins.

## 7. The first box measurements of this design (2026-10-20/21, after round 8)

Run within a $5 window the user allowed; both measurements train nothing. Records on `main` in `training/metrics/near_tie_coins_20261020/` and `training/metrics/own2_base_20261020/` (the large rows, coin, plant and game files stay on HF under `tier-b/` with the same folder names); code on the branch `exp/near-tie-coins` (`tools/coin_decode.py`, `tools/coin_labels.py`, `tools/coin_rows.py`, `tools/coin_plant.py`, `scripts/coin_batch_box.sh`); pre-registrations `docs/near_tie_coins_prereg_20261020.md` and `docs/own2_base_prereg_20261020.md`, each with its "Measured" section.

**S0(b), the base** [M]: `own2` against `own1`, both at 64 slots and the reference decode, 6,400 decisive games: **+22.7 +- 4.4 Elo**, lower bound +14.0, so the base π_0 is `own2`. Agrees with R1's +16.5 +- 12.3. Wall 3,533 s on 20 workers (an RTX 4090 with an EPYC 7702), about $0.66.

**S0(a), the attack known answer** [M]: `own1` against itself, attack logits +1 on the seat a logged coin chose, 6,400 decisive games. The hooked seat: **-41.3 +- 4.4 Elo** (window [-111, -41]); through the label code, **-0.242 +- 0.023 z** for 2Δ (window [-0.640, -0.236]); the match's mean z on the hooked seat -0.118, half the label's 2Δ, as it must be. Both readings pass at the edge of their windows: the effect is about half the -76 +- 17 Elo the decode-knob records predicted. The label orientation and the match path are right. Dice control 0.00444 (in window).

**S0(c), the smoke, and batch 1 of S1** [M]: `own1` self-play at the reference decode with coins on both sides, w 0.35, p 0.25, on the pool (`run_elo_batch`, 120 games in flight). Batch 1 ran at 60,000 decisive games (the design's 100,000 cut to fit the window), 23 capped, 9,317,548 band decisions. Checks, all passed:
- coin plant per mover side 0.994 +- 0.008 and 0.995 +- 0.008 (target 1; the derived SE was 0.0071);
- dice control slope 0.00436 +- 0.00005 (window [0.00267, 0.00565]);
- I-coin: chi-square overall 2.25, bins p 0.64, slope of D - P on G and on the index within 1.3 SE of 0, capped games fine;
- I-log: 9,328,402 band decisions played as the coin chose, 0 mismatched, 2 unresolved;
- band census: **77.6 +- 0.13 band decisions per side-game** (derived 76.1 from `parity3`'s margins; quartiles 47, 67, 97).

Labels [M], T = s ω (z - β·S) with β from the luck walk (b = 0, γ = 0):
- **pooled mean +0.0003 +- 0.0007** (the derived SE was 0.00070); by mover A +0.0012 +- 0.0010, B -0.0006 +- 0.0010;
- by 0.05-nat margin bin, from [0, 0.05) to [0.30, 0.35): -0.0023 +- 0.0017, -0.0001 +- 0.0018, +0.0009 +- 0.0017, -0.0016 +- 0.0018, +0.0005 +- 0.0020, +0.0020 +- 0.0021, +0.0038 +- 0.0019;
- by ordered pair type (a0 > a1): move>move +0.0003 +- 0.0009 (5.6 M labels); **recruit>recruit +0.0070 +- 0.0024 (z 2.9, 0.77 M labels)**; attack>attack -0.0029 +- 0.0029; attack>move -0.0003 +- 0.0031; end_turn>move -0.0017 +- 0.0032; move>attack -0.0031 +- 0.0032; move>end_turn -0.0018 +- 0.0036; the nine smaller types all within 1.7 SE of 0.

What this settles and what it does not:
- The average near-tie decision carries no advantage either way: the runner-up's mean advantage is within +-0.0017 z of 0 at 2 SE, pooled. No typed constant is resolved at a family-wise 5% line over 16 types; recruit>recruit is the one candidate (z 2.9).
- **The spread v is still unmeasured**: a zero mean is compatible with any spread. Reading v needs the head's fits (ruling (e)): the design's spread estimator is built from per-batch head weights.
- Derived numbers held where they could be checked: the pooled SE (0.00070 derived, 0.00070 measured), the census (76.1 against 77.6), the plant SE (0.0071 against 0.0082, about 15% wider).
- Price on this host: 22,500 to 22,900 games an hour (stage A 6,400 in 1,007 s; batch match b05 10,000 in 1,601 s), at $0.577 an hour, $2.56e-5 a game; the whole box about $2.0 for 66,400 games.
