# Round 2 challenge: intent iteration

Stance: move the unit of policy improvement from the single action to a learned **intent**, a discrete code that conditions the network's actions and can be re-chosen after every roll. A discrete latent-variable imitation model learns the intents from the human corpus. Designed coins randomize them inside argmax self-play, the policy's own play to the end of the mover's turn prices them, and a fogged head selects among them. The current design improves the argmax policy one action at a time near its mode; this one improves which coherent mode the policy plays, then grows the set of modes from its own games. The zero-floor critic is kept as an ingredient: both designs need it.

## 1. Why the action is the wrong unit

Four of the current design's quantities depend on s, the advantage spread among the top candidates at a decision:
- **Calibration error**, σ_r/(N s²)^½: 0.06 to 0.12 at 400,000 deviations for s = 0.01 to 0.02, and millions of deviations below s = 0.005 (its open question 3).
- **Fallback F**, which needs σ²·d_eff/v deviations for r² = 0.5: 800,000 at v = 5×10⁻⁴, and 4 million at the critique's bound of 10⁻⁴.
- **Gain per round**, at most 10 switches of about β·s each.
- **Assembly of coordinated changes.** A change needing d coordinated decisions shows a positive one-step advantage under the base continuation only at its last decision, so it takes up to d rounds to assemble (open question 5).

The record puts the gains elsewhere:
- The largest lever, the −1.5 offset (+229 Elo), reshapes every turn, about 20 decisions per side-game.
- 6 of the 7 confirmed better turns took more decisions (`docs/turn_gap_ref_prereg_20260921.md`; two of them carry luck).
- Argmax beats temperature-1 sampling by +412 Elo: incoherent per-decision randomness is what costs strength.

**Spread.** An intent bundles the decisions a plan changes. Suppose d of a turn's decisions differ from the base, each with spread s and mean pairwise correlation ρ̄. Then

    Var[Q(x,z) − Q(x,z_0)] ≈ s²·(d²ρ̄ + d(1 − ρ̄)).

At d = 4 to 6 and ρ̄ = 0.3 to 0.6, s_z/s ≈ 2.8 to 4.9, so a coin on an intent carries 8 to 24 times the information of a coin on an action. Per unit of dilution, if the mean cost also grows with d, the advantage is 2 to 4 times; it is more if human modes are competitive, which S1 measures. The calibration error falls as 1/s_z, the fallback needs 8 to 24 times fewer coins, and one switch carries a coordinated change.

**Horizon.** Policy iteration over intents has about 20 choice points per side, 72 with re-selection after rolls, against about 225 decisions. Temporal abstraction shortens the effective horizon ([OPAL, Ajay et al. 2021](https://arxiv.org/abs/2010.13611)).

**Evidence.**
- *For:* latent plans beat direct imitation of actions in MiniRTS ([Hu et al. 2019](https://arxiv.org/abs/1906.00744)); AlphaStar conditioned on strategy statistics from human games ([Vinyals et al. 2019](https://www.nature.com/articles/s41586-019-1724-z)); persistent exploration beats per-step dithering ([Dabney et al. 2021](https://arxiv.org/abs/2006.01782)).
- *Against:* abstractions help "only in limited contexts" ([Jong, Hester & Stone 2008](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/AAMAS08-jong.pdf)); strong conditional decoders ignore latents ([Bowman et al. 2016](https://arxiv.org/abs/1511.06349)); a latent space on the data's support caps improvement at the vocabulary ([PLAS](https://arxiv.org/abs/2011.07213)), which 2.6 addresses.

## 2. The algorithm

### 2.1 The intent learner

`own1`'s network gains an intent token, a learned embedding of z ∈ {1..K}, with K = 16 in round 1. It enters the trunk like the global token, so the memory, written before acting, records it. Three parts:
- the decoder π_θ(a | h_t, z), which is the factored action heads, initialized from `own1`;
- a prior p_θ(z | h_t);
- a training-only posterior encoder q_φ(z | h_{t0}, a_{t0..end}), a two-layer transformer over the rest of the side-turn.

On the corpus's winning side-turns τ:

    L = E_{z~q_φ} [ −Σ_{t∈τ} log π_θ(a_t | h_t, z) ] + β·max(λ, KL(q_φ ‖ p_θ(·|h_{t0})))

The code is drawn by straight-through Gumbel-softmax, with λ as free bits. One own-trajectory pass with sampled codes follows: `own1`'s recipe against causal confusion, about $2.1 on record. In play, a_t = argmax π_θ(a | h_t, z) with the decode's end_turn offset. Intent points are side-turn starts, and from round 2 also the decision after each fight resolves.

### 2.2 Generation: intent coins

At a side-turn start, a coin keyed by (game seed, side-turn), separate from the dice, draws z uniformly from the prior's top 4 with probability 0.2. Otherwise it takes the selector's choice. That gives about 4 coins per side-game, the current design's count, with the law logged. A game is a factorial experiment over about 8 intent draws and about 160 dice events (110 fights, 46 trait rolls, 2 advancements; `training/metrics/match_luck_20261019/readout.txt`).

The critic is the current design's: privileged, reading both memories and both current codes, trained on Σ_t R_t². An optional K-way head adds the intent draws to the identity, L_j = C(x, z_j) − Σ_z q_j(z) C(x, z), from one forward; omitted, they leave a floor of about 8·s_z² ≈ 0.007 of Var(z).

### 2.3 Pricing an intent: turn closure under the policy's own play

At an intent point with true state x, take each code z among the prior's top 4 plus the base's z_0. Play the mover's turn from a fork with the network at argmax given z, on its own fogged observation, re-deciding after every roll, to the mover's end_turn x_e. The estimate is

    Ŷ(x, z) = C(x_e) − Σ_{k in the turn} L_k(C),    E[Ŷ | x, z] = E[C(x_e) | x, z] =: Q(x, z).

The L_k are martingale differences, so the turn's dice are integrated and never selected. Each candidate is a single choice followed by the policy's own re-deciding play, which is the ruling's form.

**Cost.** 4 codes × 2 samples × about 23 forwards (10 decisions, 13 outcome reads) ≈ 200 forwards, or $1.1×10⁻⁵ per point. At 40 side-turn starts a game, 400,000 targets cost $4.4.

**Noise.** In-turn luck is about 0.022 a turn (2.6 fights × 0.0086), roughly halved by the correction, against a signal of s_z² ≈ 10⁻³: it costs cheap samples, not bias. S3 picks between this estimator and the K-way head by calibration.

### 2.4 Selector, calibration, switch, forecast, audit

**Selector.** A fogged head S_ψ(h, z) regresses Â(x, z) = Q(x, z) − Q(x, z_0) by squared loss, with folds split by game. It converges to E[Â | h], the information-state advantage, with no strategy fusion (the current design's argument, after Baisero & Amato).

**Calibration.** The luck-corrected result is regressed on x̃ = (1[z = k] − q_k)·S_ψ(h, k), plus a nested model of mean cost by code rank and prior log-odds. The slope β_z has standard error σ_r/(N s_z²)^½: 0.041 at N = 400,000, σ_r² = 0.6 and s_z = 0.03, and 0.025 to 0.061 for s_z from 0.05 to 0.02.

**Switch.** π_{k+1} plays argmax_z (α + β_z S_ψ) where that is positive with posterior probability ≥ 0.8, capped at 8 switches per side-game.

**Forecast.** Actions given a code coincide, so the performance-difference lemma runs over intent points only: J(π′) − J(π) = E_{π′}[Σ_j A^π(h_j, z′_j)]. Ĝ_old and Ĝ_plug follow as in the current design.

**Audit.** Fair coins between z′ and z_0 where they differ: standard error (4Mσ_r²/n)^½ = 0.035 at M = 8 and n = 16,000 games ($0.4).

### 2.5 The stop rule

Intents carry the turn's length: a code whose human turns use every unit plays long turns. π_1^raw, at offset 0 for every code, tests how much of the end_turn bias the intent absorbs. The head's factorization remains, so the prediction is partial: the offset-0 penalty shrinks against Match A's.

### 2.6 Beyond human modes

After each round:
- the codebook is refit on π_k's own turns (both sides) plus the human turns, gaining 8 codes and keeping those the selector uses (self-intents), so the improved player's turn types become selectable;
- the selected codes are distilled into p_θ, so the next base is π_k.

From S6, the current design's action-level student runs inside the chosen intent, audited by type: intents carry the large moves and actions refine them.

**Invariants (tests).**
- Each law sums to 1.
- R_t(V) = 0 on an enumerable duel with a two-code coin.
- The centred intent regressor has mean zero.
- Folds split by game.
- No fog-off token and no posterior output reaches the decoder, prior or selector in play.
- Permuting results nulls β_z and the audit.
- KL ≤ log K per side-turn.
- A plant's code reproduces its variant (P-I).

## 3. Regimen

Prices are the current design's; a corpus epoch with the encoder is about $0.40.

| Stage | Cost | Predicted readout | Kill line |
|---|---|---|---|
| **S0 instruments** | $1 | (a) Generation versus eval numerics agree on ≥ 98% of argmax decisions. (b) `own1` and `parity3` decodes on 1,000 logged states give the share d of side-turns where they disagree at least once (expected ≥ 0.5). | (a) as in the current design. |
| **S1 intent learner** | $6 | **P-I** (learner path): 20,000 games where a hidden fair coin per side-turn picks `own1` or `parity3`, and the learner trains on them. The posterior recovers the variant at ≥ 0.95 on disagreeing turns and 0.50 ± 0.03 on the others; KL per side-turn ≈ d·ln 2; the code's decoder matches its variant on ≥ 90% of disagreeing decisions. **P0_z** (coin and audit path): 40,000 games with fair code coins per side-turn against `parity3`; the midpoint estimate reproduces `own1`'s +88.6 ± 12.7 over `parity3` within 2 combined standard errors (one is about 19 Elo). Then the corpus model: (i) KL 0.7 to 2.0 nats per side-turn (a weak prior); (ii) holdout NLL gain per side-turn between half the KL and the KL; (iii) modal-code decode against `own1`, −20 to +20 Elo; (iv) codes drawn from the prior, argmax within, against `own1`, −15 to −80 Elo (action sampling: −412). | P-I or P0_z outside its band: that path is broken. KL < 0.3: posterior collapse. (iii) below −40: `own1` stays the base and codes are alternatives only. (iv) below −150: the modes are not competitive plans; the intent line ends and the current design proceeds unchanged. |
| **S2 critic** | $12 | The current design's S1 unchanged (P1 plants, 50,000 games, ρ_C curve), played by the modal-code player with intent coins, a quarter of it against `own1`. New: on 2,000 games, the corrected turn-closure spread ŝ_z among the top-4 codes and the same critic's one-step spread ŝ among the top-4 joint actions. Predicted ŝ_z/ŝ in [2, 5]. | ŝ_z/ŝ < 1.5: intents carry no more signal than actions; the critic passes to the current design's S2 with nothing lost. ρ_C ≥ 0.80: fallback F_z. |
| **S3 selector** | $8 | **P2_z**: the village tax with a hidden rate, at intent points. The selector's planted component follows the information-state rate, not the hidden bit (leak test), and π_1 on the taxed game gains villages by its forecast. Then 400,000 turn-closure targets; β_z in [0.5, 1.0] with standard error ≤ 0.06; held-out gain over the nested model above 2 standard errors; turn closure against the K-way head chosen by calibration. | P2_z outside its bands: stop. β_z < 0.3: fallback F_z. |
| **S4 round-1 player** | $2 | Ĝ_old, Ĝ_plug and the audit (16,000 games) are written first. Prior for sizing: 4 to 8 switches of about 0.6·s_z calibrated each, **+15 to +85 Elo, central +35**, on top of (iii)'s base. Matches: π_1 and π_1^raw against `own1`, π_1 against `parity3`. | Adopt at ≥ +25 Elo and within 2 combined standard errors of the audit. Gain over `parity3` under half that over `own1`: exploitation. |
| **S5 rounds 2–3** | $9.5 each | Critic warm-refit on 25,000 games; self-intents; re-selection after rolls. **P6** (refit path): offset-0 turns mixed into the refit corpus at a known share; a new code recovers them at P-I's accuracy. Each round's Ĝ_plug and audit precede its match; prior at most the last round's gain. | P6 outside: the refit path is broken. Two rounds below one match standard error: go to S6. |
| **S6 distill, nest** | $8 | **P5** as in the current design; selected codes distilled into the prior; the action-level student inside intents, predicted to add at most round 1's intent gain. | P5 outside: the distillation path is broken. |

The planned path costs about **$56**, and at most about $72 with one rerun.

S1 and S2 test the premise in four parts, at about $18, before any player is built:
- the code carries information (KL);
- the decoder obeys it (P-I);
- the modes are competitive (iv);
- they spread more than actions (ŝ_z/ŝ).

A kill there costs about $6, since S2's critic is the current design's own.

**Fallback F_z.** A linear head on frozen features, fitted to intent coins alone, reaches r² = 0.5 at σ²·d_eff/s_z² = 0.8·500/9×10⁻⁴ ≈ 450,000 coins, about 55,000 games ($1.4); at s_z = 0.02, 125,000 games ($3). F needs 8 to 24 times more. F_z is outcome regression (episodic natural actor-critic), so it waits for the user's ruling.

**The $1,000 case** is argued only after three rounds land within their audits: 30 or more rounds with the vocabulary refit from the player's own turns, a larger critic, a league check. Nothing links `own1` to human ratings.

## 4. What is new, and the nearest work

Searched 2026-10-19. No paper matched the combination; absence is not proven.

1. **A factorial design over learned plans**: known-propensity coins on a discrete intent vocabulary, inside argmax self-play with an exact luck identity, so information per coin scales with s_z². Nearest:
   - micro-randomized trials ([Boruvka et al. 2018](https://arxiv.org/abs/1601.00237));
   - AlphaStar's z, drawn for diversity and trained by pseudo-rewards, not used as an experiment;
   - ez-greedy, which persists but has no estimator.
2. **Intents priced by the policy's own play to the end of the turn**, luck-corrected and re-selectable after every roll. Nearest:
   - options with interruption (Sutton, Precup & Singh 1999);
   - [TAP](https://arxiv.org/abs/2208.10291), a beam search over VQ codes with the decoder as its model;
   - PLAS and OPAL, policies over latents for offline RL.
3. **Self-intents**: the vocabulary refit each round on the improved player's turns. OPAL extracts its primitives once. Not verified beyond my searches.
4. **A planted-latent check of a behaviour model**, whose recovery rate is predicted from the measured disagreement of two known policies mixed per turn.

## 5. The weakest point

**The vocabulary.** Human turns must yield codes that:
- carry information;
- the decoder obeys;
- stay competitive when drawn;
- differ in value more than single actions.

Strong conditional decoders are known to ignore latents, and Wesnoth turns may be too situation-specific for 16 to 32 reusable modes. If the modal-code base is weaker than `own1` (S1 iii), codes become alternatives only, roughly halving the reach.

**Self-intents** are the least predictable step: nothing measured says a refit vocabulary moves past human modes rather than re-describing them.

**The critic** remains a shared risk, softened more by F_z than by F.

**Two rulings for the user:**
- Does an intent held through a turn, re-selectable after every roll, satisfy "never whole turns"? Every comparison is a single choice followed by the policy's own re-deciding play, but the code shapes the turn.
- Is F_z covered by the killed outcome policy gradient?

## 6. Decision record

- Rejected: per-action temperature as exploration, because it measured −412 Elo.
- Rejected: continuous latents, because they have no enumerable coin law.
- Rejected: diversity-driven skill discovery (DIAYN-style) as the first vocabulary, because it gives no competitive base.
- Rejected: intents valued by turns played with their own dice, because of the ruling.
- Rejected: a learned world model (Stochastic MuZero) for the look-ahead, because the core steps in 0.05 ms on CPU with an exact chance law, while a forward costs about 0.3 ms of GPU.
