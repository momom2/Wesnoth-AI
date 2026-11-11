# Audit: what stands between "learn from who won" and the training that runs (2026-11-11)

The user asked for it on 2026-11-11, after the match test
(`docs/trunk_step_match_prereg_20261111.md`) showed that the KL step-size
rule had kept the trunk from training: "Run an audit of the project to
list all such modifications which complicate training dynamics."

## How it was done

- **The plain loop** the code is compared with: the policy plays games;
  each decision's log-probability gradient is weighted by its side's
  outcome, +1 or −1; the weights move by a fixed learning rate times the
  sum; repeat.
- **What is listed:** everything that reweights, rescales, restricts,
  averages, sizes, selects, gates or stops, and every place where
  something other than the network decides play.
- **Four readers** went through one area each (subagents, read-only): the
  outcome to the gradient, the gradient to the next policy, the network
  and play side, and the documents. The code is
  `feature/fixed-step-rounds` (f34d8766), which is the lean line's code
  (21862a0a) plus the fixed step.
- **Checked by hand afterwards:** the lines marked ✔ below. The other
  citations are the readers' and were not re-read.
- **Run evidence** is round 22's records
  (`C:\Users\momom2\wesnoth_runs\rounds_local\round_22\`).

"Evidence" below means a reading in games unless it says proxy. Apart
from today's match test, **no game has ever compared training with one of
these mechanisms against training without it.** What exists is variance,
cosine and first-order readings.

## The list

### A. The learning signal: who gets credit

A decision's weight is not ±1. It is
`z − b − β·S − α·v_t − γ·β·x_t` ✔ (`tools/pg_labels.py:185`,
`tools/pg_pass.py:360-362`).

| # | Mechanism | What it does | Active | Evidence |
|---|---|---|---|---|
| A1 | Held-out games | 1,250 of a round's 6,250 games (20%) are never trained on. They fit A2-A5. Before the lean rounds they also sized and judged the step. | yes | none in games |
| A2 | Baseline b | Subtracts the expected result for the map, seat and faction pair. The per-cell term is shrunk to nothing (round 22: k = 6.9e8). | yes | proxy: noise 1 → 0.983 per game-side (stage 1) |
| A3 | Whole-game luck, β·S | Subtracts seven sums of dice luck over the game, with β refitted each round by least squares on the held-out games. | yes | proxy: 0.983 → 0.851 (stage 1); round 22 reads 0.764 ✔ |
| A4 | Value baseline, α·v_t | Subtracts α times the value head's own expectation at the decision. α ≈ 0.83, refitted each round ✔. Nothing trains the value head. | yes | proxy: per decision 0.838 → 0.603 (round 22 ✔) |
| A5 | Luck so far, γ·β·x_t | With γ ≈ −0.78 ✔ it adds back most of the luck dealt before the decision. Not in the design (departure 2). | yes | proxy: → 0.542 (round 22 ✔); at stage 1 it entered at 5.08% against a 5% rule |
| A6 | Entry rule | A4 and A5 switch on only if they cut the noise by 2 SE and 5% and pass a statistical gate ✔ (`tools/pg_corrections.py:157-158`). So the weight's form can change from one round to the next. | yes | at 2,500 held-out games "in an occasional round the luck term drops out" |
| A7 | Capped games | A game with no winner counts 0 and stays in. | armed; none occur | none needed |
| A8 | Lost games | Up to 10 games a chunk may be missing or fail the luck walk, among them any game killed at 20 minutes. | armed; 0 in round 22 | none |

On paper A2-A5 cut the per-decision noise from 1 to about 0.54, which is
worth about 1.85 times the games, less the 20% held out: about 1.5 times
net. That is a variance reading. They are unbiased only if fitted on
games the round does not train on, which is what A1 is for.

### B. The gradient pass

| # | Mechanism | What it does | Active | Evidence |
|---|---|---|---|---|
| B1 | Heads-only pass | Only the four heads' final maps (7 tensors) get a gradient. The trunk, the encoder and the memory get none (`tools/pg_pass.py:267-268`). | rounds 2-22; off in the fixed mode | against: the match test (+88.2 with the rest, +34.6 without) |
| B2 | 16-decision windows | The gradient through the memory is cut every 16 decisions (`tools/pg_pass.py:410`). | no effect on a heads-only pass; **live in a full pass** | open: 8 games read a cosine of 0.62 with the uncut gradient; 128 games did not replicate it |
| B3 | fp16 backward | Each window's loss is scaled by a power of two; a window that overflows is run again at half the scale (21 of 4,926 in round 22). | yes | proxy: cosine 0.999995 with fp32, on 8 positions |
| B4 | Replaced actions | When the simulator plays something other than the ordered action (a forced end of turn, a dropped attack, a moved recruit), the ordered action is still trained at full weight. | armed; how often is not counted in `pass.json` | none |
| B5 | Checks that abort | Foreign games, record fingerprints, the replay identity, the precision floor. They stop the pass and change no number. | yes; none fired in a round | n/a |

The replay identity's limit is "a twentieth of the smallest radius the
step search reads" ✔ (`tools/pg_pass.py:184-187`). Its justification
names the rule that has just been removed.

### C. The direction

| # | Mechanism | What it does | Active | Evidence |
|---|---|---|---|---|
| C1 | Per-weight normalisation ("reduced AdamW") | Each weight's gradient is divided by the root of its mean squared per-window gradient. So a weight moves by its signal-to-noise ratio, not by its gradient, and the result depends on which decisions share a window. | every round, and in the fixed mode | proxy, at equal KL on the full network: plain gradient 0.0128, this 0.0124, muon 0.0147, each ± 0.0037. No measured benefit. |
| C2 | The arm, chosen once | Stage C compared five directions at one equal KL and kept heads-only (+0.110 against +0.012). The choice was never read again. | rounds 2-22 | **against:** at equal KL the full step was 8 times too small, so this comparison was the KL rule's. The match test contradicts it. |
| C3 | Which weights are "heads" | Four final maps. `target_k_proj` and the weapon head's first layer are left out. | yes, and it defines the fixed mode's two groups | none |
| C4 | Direction averaged over 8 rounds | The heads' direction is averaged with the 8 rounds before, each scaled to unit norm. | from round 11 (2 rounds), 8 from round 19; kept for the heads in the fixed mode | proxy: 2 rounds +0.018 ± 0.009; 8 rounds a cosine only (+0.117 ± 0.078) |
| C5 | Fixed lengths (new today) | Each group's direction is scaled to unit norm, then to a hand-set length (0.5 and 0.019). The step never shrinks as the gradient does. | the fixed mode | the match test, for the rest's length at `own2`; nothing at round 22's policy |

### D. Step size and adoption (the family the user removed)

| # | Mechanism | What it did | Active | Evidence |
|---|---|---|---|---|
| D1 | KL search | Sized the step so the policy moved a set amount. | rounds 1-22; off in the fixed mode | against: the match test |
| D2 | The range's edge | Capped the step at 0.5 nat a game-side. The cap exists so that the evaluator's importance weights can read a step, not for training. | rounds 2-22; off in the fixed mode | it bound the step in 12 of rounds 2-18 ✔; the 2x rung "never fell in range" |
| D3 | The radius carried over | The next round's radius is this round's adopted KL, and halves on a revert. In lean rounds it could only fall. | rounds 1-22; off in the fixed mode | none |
| D4 | Halves, revert, weight check, cross-play | Half A chose the step, half B confirmed it, a heavy-tailed step went to a 10,000-game match. | rounds 1-18 | the revert never fired in 17 rounds; the weight check fired twice and both matches adopted; evaluation took about 42% of a round |
| D5 | Adopted unread | No reading of a step before it is kept. | rounds 19-22 and the fixed mode | rounds 19-22 have no reading from games |

### E. Sizes set by hand under the old rule

- **5,000 training games a round** (departure 25). It was derived from
  the heads-only direction's critical batch of 281 games, for a
  KL-normalised step. The full-network direction's reads 44, so a round
  is probably far larger than the trunk needs (reasoned, not measured).
- **1,250 held-out games** (departures 34, 36, 43), cut three times with
  no strength reading. The cut to 2,500 doubled the noise that falsely
  failed round 12's gate.
- **The KL subsample** (250 game-sides in lean rounds): unused in the
  fixed mode.

### F. Gates and stops (they change no update, but they halt training)

| # | Gate | What happened |
|---|---|---|
| F1 | The readout's 18 tests on each round's games (1% false alarms by construction) | Fired twice. Round 12: traced to the test's own SE, which was then changed. Round 19: no fault found, and the games were replayed. |
| F2 | The corrections' own gate (A6) | Never failed a round. |
| F3 | The claims' transfer stop | Fired at round 3, 2.2 SE under forecast ✔. The line went on past it (departure 33) and reached +105.7. |
| F4 | The pause on three confirmed readings | Never fired. |
| F5 | The lean stops (a claim by 2 SE, an argmax match by 1 SE, every 6th round) | Never evaluated: the line stopped before round 24. |
| F6 | The planted-signal check and stage P | The planted check failed. P was then skipped (departure 31), so it blocked nothing: "P stays skipped". |
| F7 | `--end-by`, STOP files, display resets, failed-part markers | Operational. |

### G. The play side: where something other than the network decides

| # | Mechanism | What it does | Where |
|---|---|---|---|
| G1 | Two different players | The gradient is computed for a player that samples at temperature 1 with no offset. Strength is measured at argmax with an end-of-turn offset. | training against evaluation |
| G2 | The end-of-turn offset, −1.5 | A constant added to the end-of-turn logit ✔ (`tools/raw_player.py:59-62`). It was fitted by an Elo sweep on `relset`, six references ago, and never refitted ✔ (`configs/reference_player.json:18`). | the reference's decode and every evaluation match |
| G3 | Joint argmax | The argmax runs over the flat list of whole actions, where end of turn is one entry and a unit's mass is split over type, target and weapon. | evaluation only |
| G4 | The attack's hex | The network picks attacker, target and weapon. The simulator picks where to attack from, by lowest path cost ✔ (`tools/wesnoth_sim.py:318`), and plays the approach itself. | all games |
| G5 | The route | The network picks a destination; the planner picks the path. | all games |
| G6 | Advancement | A unit with several advancements always takes the first ✔ (`rust/wesnoth_core/src/core_units.rs:108-109`). The network is not asked. | all games |
| G7 | Forced end of turn | An action the simulator cannot translate, or eight refusals in a row, ends the side's turn. | all games |
| G8 | Caps | 200 turns and 2,000 actions a side end a game with no result. Training counts it 0; evaluation replaces the game and ignores it. | both, differently |
| G9 | Rejected recruit hexes | The legality mask reads them, but the network's input has that column written 0 ✔ (`rust/wesnoth_core/src/core_encode.rs:354-359`). So after a bounce the network decides again on the same input under a different mask. **This contradicts CLAUDE.md's Principle 6** ("mirrored in the encoder's features"). | all games |
| G10 | What the network sees | Only villages, castles, visible units, reachable hexes and their neighbours, and unseen hexes within 6 of an own unit. | all games |
| G11 | The memory's hand-set start | The gate's bias starts at −2 (it keeps about 88% a step) ✔. | the network |
| G12 | The opponent | Itself only, at temperature 1. | training |

At temperature 1 the training player samples exactly from the network's
masked law: no bias, prior, clamp or forced action was found on that
path.

### H. Already out

- The 78 mechanisms quarantined on 2026-09-03
  (`quarantine/INVENTORY.md`: 14 refuted, 27 untested, 6 validated on a
  proxy only). The rounds' code imports none of them.
- Nothing from `wesnoth_ai/trainer.py` is stepped by the rounds. The one
  import from `wesnoth_ai/optimizers.py` is two constants for the unused
  muon arm.
- No value, belief, imitation or entropy loss; no discount; no reward
  shaping; no clipping; no importance weights; no dropout.

## What the audit suggests

Three kinds of item deserve a reading in games, as the KL rule got today.
Each is one factor.

1. **Downstream of the KL rule, so now unsupported:** the arm choice
   (C2), the round size (E), and the replay limit's justification (B5).
2. **Never measured and now live:** the 16-decision window (B2) and the
   per-weight normalisation (C1). Both shape every full-network step, the
   match test's included. C1 is the cheaper to read: a plain-gradient
   candidate can be built from stage C's saved pass.
3. **Something other than the network decides** (G2, G4, G6, G9). These
   touch Principle 3, and G9 touches Principle 6. They are the user's to
   rule on.

The label corrections (A) are different in kind. They are standard
variance reduction and unbiased in principle. Their cost is the
machinery: a fifth of the games, a fit a round, and two gates that have
stopped the line. Whether training without them is as strong per GPU-hour
is not read.

## Record errors found on the way

- The code named the fixed step "departure 49"; the pre-registration
  numbers it 50. Fixed (`feature/fixed-step-rounds` 27cea50a).
- `docs/status.md` said a step was "adopted if half B reads it positive".
  The rule adopted unless half B read below −2 SE. Fixed with this audit.
- `docs/status.md` said every claim landed within 2 SE of its forecast.
  Round 3's read 2.2 SE under and fired the transfer stop. Fixed with
  this audit.
- `docs/status.md` counted 48 departures; there are 50. Fixed with this
  audit.
- Departure 50 did not say that its full pass keeps the 16-decision
  window. Added on `feature/local-gpu-stage1`.
- Stale docstrings: `wesnoth_ai/action_sampler.py:41` and
  `tools/raw_player.py:50-51` describe gates the code does not have. Not
  fixed.

## Not covered

- How often B4 and G7 happen in real rounds (the counts are inside
  `pass.pt`).
- The box path (bf16) and the live bridge.
- `tools/pathfind_sim.py`, `tools/neutral_ai.py` and the formulas inside
  `tools/selfplay_readout.py`'s gates.
- Whether any Ladder map has an acting neutral side.
- The origin of the 200-turn cap.
