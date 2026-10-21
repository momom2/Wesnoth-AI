# Material for the optimizer discussion

Six decisions gate self-play training. Each has its options, evidence and the group's recommendation. Details: `current_design.md`; costs: `analysis/round11_synthesis_plan.out`, at $0.577 an hour. "±" is a reading's uncertainty (one standard error); a difference is clear when it exceeds twice its uncertainty.

## The proposal in brief

`own2` plays itself at temperature 1, both sides, with its memory, and no end_turn offset. After each round of games, every move gets a score:
- +1 if its side won, −1 if it lost;
- less what the map and factions predict;
- less the dice luck the simulator computes exactly for every attack.

The weights move in the direction that makes high-scoring moves more likely: the gradient, summed over the round's games.

How far they move is the KL divergence between the new and old players' move probabilities, in nats (natural-log units): ½ nat per side-game is about 0.002 per decision.

A step is checked on games it did not learn from: the old player's games are re-read as if the new player had made the same moves, each weighted by how much likelier the new player was to make them (importance weighting). This says, without playing, how the new player would score against the old one, for steps up to about ½ nat per side-game.

## 1. The update: how the network learns from a round

**Options:**
- (a) One step per round, sized by KL and checked on held-out games (recommended).
- (b) An ordinary training pass with a learning rate, checked by matches every few rounds.
- (c) Several passes over each round's games (PPO).
- (d) Natural-gradient methods, whose steps are measured by how much play changes (TRPO, ACKTR).

**Evidence:**
- **Rounds are probably mostly noise.** The one measured noise level for learning from results in this game is a 43-number recruit preference. A step's signal equalled its noise only at about 11,600 games [measured, `round11_simplify_noise.out`], and the network has 15.6 million weights.
  - In that regime (a) and (b) move the weights the same way, to first order; (a) has nothing to tune and lets optimizers be compared on one recorded gradient.
- **Step size matters.** On record, the recruit rule gained +18.0 Elo at first order and lost 12.3 ± 4.5 of it to its own side effects [measured, round 10]. A step must be checked on the real outcome, not on a forecast.
- **(c) reuses data**, which saves little (training is two thirds of a game's cost [derived]) and fits the batch's noise; PPO's gains were traced to implementation details ([Engstrom et al. 2020](https://arxiv.org/abs/2005.12729)).
- **(d)** ACKTR reports 2–3× sample efficiency ([Wu et al. 2017](https://arxiv.org/abs/1708.05144), abstract). It needs new code through the memory.

**Recommendation: (a).** (d) is built only if stage C shows richer optimizers winning.

## 2. The optimizer

Under (a), the optimizer is only the rule that turns the round's summed gradient into a direction. Learning rate, schedule, momentum and weight decay disappear.

**Options:** AdamW's per-weight normalization (how `own2` was trained); plain SGD; Muon (orthogonalized layer updates); SOAP (Adam in a basis fitted to each layer); schedule-free; K-FAC (an approximate natural gradient).

**Evidence:**
- **Muon.**
  - In a study of language-model agents its gain (success 0.29 → 0.55, single seed) disappeared in the paper's later version under a control matching update sizes ([Ruan et al. 2026](https://arxiv.org/abs/2607.16169), v1 against v4).
  - Another paper reports Muon collapsing in RL fine-tuning of language models and blames noisy gradients ([Fan et al. 2026](https://arxiv.org/abs/2605.19282)).
  - Fine-tuning an AdamW-trained model with Muon gave no significant advantage, by its authors' reading ([Liu et al. 2025](https://arxiv.org/abs/2502.16982)).
- **SGD.** It falls behind Adam on transformers, whose layers differ widely in curvature ([Zhang et al. 2024](https://arxiv.org/abs/2402.16788), supervised). Normalization pays mainly at large batches ([Zhang et al. 2019](https://arxiv.org/abs/1907.04164)).
- **SOAP.** No reinforcement-learning evidence found.
- **Schedule-free.** There is nothing to schedule here, and its weight averaging conflicts with always playing the current weights.
- **The [LessWrong post](https://www.lesswrong.com/posts/9gqjdRyJxSM6kS7zi/i-trained-gpt-2-with-full-batch-gradient-descent-so-you-don)** (one run per setting, a rate not retuned for full batch, a Muon-family optimizer). Three findings transfer:
  - A set used repeatedly generalized worse (3.52 against 3.25 validation loss), so each game is used once.
  - Exact-gradient steps at a fixed rate oscillated (consecutive gradients at cosine −0.945, the loss rising on a third of steps). Here KL sets the step and held-out games check it.
  - Bigger fresh batches saved steps, not data (18.8× the data for 6.8× fewer steps). Games are the cost here, so rounds stay near the cheapest size.
- **The built bench** (16 imitation passes from scratch, $2.50) measures high-signal supervised training, not this update.

**Recommendation: decide by measurement in stage C**, at no extra games. One recorded gradient becomes a step by AdamW-style normalization, SGD, Muon, the output heads alone, and the heads' temperatures alone; each moves the same KL and all are compared on the same held-out games. AdamW-style is the default; another replaces it only on a clear win. SOAP is built only if AdamW-style clearly beats SGD; K-FAC only if SOAP clearly beats AdamW-style. Keep the 16-arm bench for a future imitation pass.

## 3. Step size

**Options:**
- (a) Each round tries three sizes (½×, 1×, 2× the last), picks the best on one half of the held-out games, and confirms on the other half. The step is undone only if the confirmation shows a clear loss (recommended).
- (b) A fixed learning rate.
- (c) Grow or shrink the step by comparing the last round's forecast with a match (the challenge stance's rule).

**Evidence.** A simulation where a step's gain is its first-order effect less a side effect growing with its square, 12 rounds from 1/16 of the best size [derived, `round11_synthesis_plan.out` §6, `round11_challenge_critique_rule.out`]:

| Rule | Share of the best possible gain kept per round |
|---|---|
| (a) | 62–90% |
| (c) | 39–83% |
| A fixed step | 44% |

(a) wins because the offline check sharpens as the step shrinks; a match does not. Past ½ nat per side-game a match against the previous round reads the largest size ($0.88).

**Recommendation: (a).**

## 4. The human prior and the opponent

**Options:**
- (a) Released: `own2` is the starting point and nothing pulls back toward it (recommended).
- (b) A pull toward `own2` (a KL anchor) from the start.
- (c) A pool of past versions as opponents from the start (AlphaGo).

**Evidence:**
- **An anchor would tax sharpening**, the change temperature 1 most needs: on the September network, argmax beat sampling by +412 ± 97 Elo in 40 games [measured].
- **The evidence for anchors** is from Diplomacy, where play must stay human-compatible.
- **Without an anchor, self-play can cycle.** A pull toward a fixed reference provably converges ([Sokota et al. 2023](https://arxiv.org/abs/2206.05825)).
- **A pool game** costs 3.4× a self-play game and trains one side [derived].
- **Claims against the reference** every third round detect gains that do not transfer.

**Recommendation: (a).** On a transfer miss you choose an anchor or a pool.

## 5. The yardstick

**Options:**
- `own2` at its pinned argmax decode with the −1.5 offset, the reference since 2026-10-21 (recommended).
- `own2` at temperature 1.

**Evidence:**
- **The pinned decode.** It is the strongest player on record. Claims start at minus the sampling debt and show directly whether self-play repays it, which is your hypothesis. The debt also contains the offset's worth.
- **A temperature-1 yardstick** is the starting point itself: it repeats what the rounds already read.

## 6. First measurements and their cost

| Stage | Cost | What it reads |
|---|---|---|
| 1. No training | $1.65 | 30,000 logged games at temperature 1; the sampling debt (12,800 games against the reference); memory on against off. Reads game length and prices, turn-cap rate, score noise and which corrections cut it, that the player samples what it logs, the sharpening slope |
| C. Gradients and candidate steps on stage 1's games, nothing played or kept | $2.73–5.71 | Whether one round's step gains measurably, and per dollar; the optimizer; the step size; the round size; the planted check offline |
| P. The planted check played | $0.65 (+$0.33) | A planted reward recovered at its predicted size; a match testing the sharpening slope in play |
| Each round | $2.78–3.97 at 20,000 games | 18–27 rounds fit in the budget |

The sharpening slope, read free from stage 1's logs, is how much a slightly colder player would gain against the temperature-1 player: your hypothesis at first order.

**Questions for you:**
- **Stage 1.** Approve it?
- **Stage C.** Does it count as a measurement or as training? It computes round 1's step without playing or keeping it. If training is approved afterwards, round 1 uses that step, so nothing is discarded.
- **The budget.** Did the $5 window count ($81.44 left, or $84.10)? What ceiling for rounds?

## What would run first, and what each outcome would mean

Stage 1 runs first, then stage C, for $4.38–7.36 together. Stage 1 also prices the sampling debt in Elo.

**The sharpening slope:**
- **Clearly positive:** colder play beats temperature 1, so training has at least that direction to follow.
- **Near zero or negative:** sharpening alone does not repay the debt; any repayment must come from other changes to the policy.

**Stage C:**
- **Round 1's step gains clearly on held-out games:** rounds are proposed, with the measured Elo per dollar and the rounds the debt would take at that rate.
- **The gain is not clear at 20,000 games, but the measured curve projects a clear gain at a larger round costing at most $18:** rounds at that size.
- **Neither:** learning from results cannot be resolved within this budget. The choice is the $1,000 case, sized from the measured curve, or a stop.

**The optimizer** is whichever wins clearly, otherwise AdamW-style.
