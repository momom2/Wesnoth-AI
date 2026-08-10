# Literature scan — unimplemented improvements (2026-08-10)

Two Opus research agents, each grounded in `docs/techniques.md` (the
code-verified inventory) and instructed to skip anything implemented
or empirically refuted. Reports verbatim below the synthesis.

## Synthesis (maintainer summary)

**The two scans converge on one strategic point:** nothing currently
holds the newly-won human prior in place once self-play fine-tuning
starts — the Gumbel distillation target is self-referential in the
model's own prior, and self-play is strictly current-vs-current. The
imitation checkpoint's +0.35-nat advantage is exactly the kind of
thing the documented prior-ratchet erodes. The three top imitation-
side items are complementary fixes for this one hole:
piKL-style anchoring to the frozen BC policy (trainer-side CE,
cheapest), the BC checkpoint as a permanent league opponent
(PFSP-lite), and RLPD-style rehearsal extending the existing
value-only `--human-anchor-file` to the policy heads.

**Search-side best value:** (1) debias the sampled-and-cut arms'
q-hat — variant (a), reserving one non-adaptive sim per halving
phase, is mctx-semantics-compatible; (5) power-mean backup is the
cheapest gateable knob (p=1 identity default). (3) Go-Exploit
archive starts attack strength-per-compute and the draw-saturated
data distribution.

**Cheap corpus wins:** KataGo-style opponent-reply auxiliary head
(the label is literally the next command in the stream, currently
discarded); HL-Gauss value targets (the shape-correct version of the
value_label_smoothing knob we already built).

**The standing caveat both scans respect and the search agent states
outright:** four fixes and +133 in-lineage Elo moved the external
0-0-30 RCA number not at all. Any adoption from this list must
pre-register an EXTERNAL observable (RCA probe, human-holdout CE),
not only in-lineage ones.

---


# Search / MCTS improvements (agent report)

## Ranked candidates

1. **Debias the cut-arm q-hat (PTP-analogue)** — S/M, low-medium risk.
   The -0.018 sampled-and-cut gap is the textbook negative bias of
   sample means under adaptive sampling (Nie et al. 2018, 1708.01977;
   Shin et al. 1905.11397). Game precedent: KataGo policy target
   pruning (1902.10565, 1.25x training efficiency). Remedies, mildest
   first: (a) reserve the LAST sim of each halving phase as a
   non-adaptive sample and use only those for the target q-hat —
   mctx-semantics-compatible; (b) shrinkage q-hat toward v_mix with a
   visit-count weight; (c) calibrated per-visit-bucket bias table
   from the existing instrument. Hook: _completed_q (tools/mcts.py
   ~1559); export to spool workers like distill_prior_discount.
2. **Progressive widening + Anytime Sequential Halving** — M, medium
   risk. Treat the 100-1300-action root as an infinitely-armed bandit
   (Carpentier-Valko ICML 2015); Anytime SH (2411.07171) removes the
   fixed-budget blocker; Sampled MuZero's importance correction
   (2104.06303) keeps the target unbiased over unsampled mass. The
   ONLY candidate that changes the coverage-pinned-at-m invariant;
   would un-refute the --mcts-sims lever. Hook: the unspent-budget
   spill path (tools/mcts.py ~1870).
3. **Go-Exploit archive of start states** (Trudeau-Bowling AAMAS 2023,
   2302.12359) — S/M, low risk. Start self-play from an archive of
   states of interest: shorter trajectories, more independent value
   targets, cuts the long-game tail that gates iterations. Plumbing
   exists (midgame setup tuples + roll_mix); add a self-play
   reservoir sampler at finalize_game. Caveats: novelty-weight the
   archive or it amplifies the passive-shuffle pathology; re-anchor
   the in-lineage Elo after the distribution change.
4. **MCTS as regularized policy optimization** (Grill et al. ICML
   2020, 2007.12509) — M, low-medium risk. Closed-form KL-regularized
   target/selection; the PRINCIPLED version of
   distill_prior_discount/temp for the prior self-ratchet. Uncertain
   marginal gain over a correct Gumbel; do not stack with the
   damping knobs.
5. **Power-mean backup** (Dam et al. IJCAI 2020, 1911.00384;
   stochastic variant 2406.02235) — S, lowest risk. One knob in
   _backup, p=1 reproduces today bit-for-bit. Counteracts the
   negative bias from the other end; calibrate with the item-1
   instrument, not by feel.
6. **Search-contempt** (2504.07757) — S/M, medium risk. Opponent-side
   Thompson sampling below a visit threshold shifts self-play toward
   decisive positions; attacks the 93-percent-draw data distribution.
   Untested with a Gumbel root; contaminates self-play-derived Elo
   (keep the eval ladder clean of it).
7. **Epistemic MCTS** (ICLR 2025, 2210.13455) — M/L, medium-high
   risk. Uncertainty-propagating search targeting the never-sampled
   shelter mass. WARNING: cliffness = C51 spread mixes aleatoric
   (dice) with epistemic uncertainty — using it as-is chases dice,
   not information; needs ensemble/RND to do properly. Cliffness has
   not paid off in this codebase yet (two OFF consumers).
8. **MAPLE multi-world aggregation** (2605.24139: +291 Elo Phantom
   Go over single-particle PIMC, which is structurally what our
   search is) — L, medium-high risk. Attacks the measured WYSIATI
   fog bias (boundary_sum +0.4..+0.65). Cheap diagnostic FIRST:
   measure whether a naive K=4 root ensemble moves boundary_sum
   toward 0 before building anything (a day of work; kills or funds
   the item).

Also noted: Multiagent Gumbel MuZero (AAAI 2024) is external
evidence FOR the already-implemented gumbel_hierarchical flag's
pre-registered A/B; Puppet Search fits the config-gated-strategies
goal but changes the legality contract; Elastic MCTS conflicts with
the bit-exact fidelity principle.

**Agent's pick-two:** item 1 variant (a) + item 3. **Standing
caveat (agent's own words):** nothing here is evidence that search
quality is what is binding on the external 0-0-30; pre-register
external observables before spending box-hours.

# Imitation / value / self-play improvements (agent report)

## Ranked candidates

1. **piKL — KL-anchor the policy to the frozen BC policy** (Jacob et
   al. ICML 2022; RL-DiL-piKL ICLR 2023 / Diplodocus; regularized
   two-player convergence ICML 2026, 2602.10894) — S/M, low risk.
   The Gumbel distillation target is self-referential in the model's
   own prior (the one-hot end_turn fixed point); anchor it to the
   FIXED external BC distribution instead, and/or add a CE term to
   the frozen BC policy trainer-side (cheapest: frozen BC model
   forwarded under no_grad on states already re-forwarded in
   step_mcts). Caveat the agent flags honestly: the LLM-finetuning
   literature (2510.18874, 2509.04259) disputes KL's retention role
   and credits on-policy rehearsal instead — but the PRO evidence is
   specifically in the search setting, which is ours. Pre-register a
   lambda sweep.
2. **PFSP league over past checkpoints** (AlphaStar-style; survey
   2408.01072) — M, medium risk. Self-play is strictly current-vs-
   current (techniques 7.6 calls it a genuine gap): mirror play makes
   the mutual-passivity equilibrium STABLE. Sample opponents over a
   checkpoint reservoir prioritized by loss rate, with the frozen BC
   checkpoint as a PERMANENT league member (the on-policy version of
   anti-forgetting). elo_ladder already dispatches two policies per
   side; the training rollout needs a side-dispatch wrapper +
   learner-side-only experience queueing. Costs about 2x throughput
   at fixed step budget; the game_weight floor needs re-checking.
3. **RLPD-style offline/online mixing for the POLICY heads** (Ball
   et al. ICML 2023) — S/M, low risk. --human-anchor-file is exactly
   this idea but VALUE-ONLY and default OFF; the policy — the thing
   that loses 0-30 — gets no rehearsal. Add policy-CE over the
   anchor states (human action indices already in the imitation
   dataset), reusing supervised_train's four-head CE.
4. **Advantage-weighted BC replacing binary winners-only** (AWR/AWAC
   family) — M, medium risk. winners_only discards 100 percent of
   loser policy data and equal-weights every winner action. Weight
   by exp(A/beta) from a FROZEN outcome-supervised value head,
   clipped (the [0.25, 4] per-game clip is precedent). Circularity
   managed by freezing the estimator.
5. **Opponent-reply auxiliary head** (KataGo 1902.10565, weight
   0.15) — S/M, low risk. The opponent's actual reply is literally
   the next command in the corpus stream — a dense label currently
   discarded. Highest information-per-line-of-code; mirror the
   existing optional-head pattern (aux_score / moves_left),
   respecting the strict=False checkpoint-compat trap.
6. **HL-Gauss value targets** (Farebrother et al. ICML 2024,
   2403.03950) — S, low risk. Gaussian-smeared categorical targets
   instead of hard two-hot C51 projection; the shape-correct version
   of value_label_smoothing, directly relevant to the measured Z
   entropy collapse (1.86 to 1.13). One function
   (_project_returns_to_atoms); keep eval CE unsmoothed.
7. **Reanalyse on human states — POLICY HALF ONLY** (MuZero
   Unplugged 2104.06294; ReZero 2404.16364) — L, medium-high risk.
   Search-improved soft targets over the 2.57M human states remove
   the BC/RL objective mismatch. The VALUE half would repeat the
   empirically-killed search-value distillation; build the
   policy/value distinction in from the start.
8. **Turn-permutation augmentation validated by the bit-exact sim**
   (dynamics-invariant augmentation, 2310.17786) — M, medium risk.
   The agent MEASURED the symmetry alternative closed: of 111 maps,
   only 5 are at least 0.99 point-symmetric; ladder maps sit at
   0.72-0.84 — so permutation is the only augmentation axis.
   Verify-or-reject each permutation via deep_state_fingerprint
   replay; measure the accepted fraction before investing.

Excluded with reasons: rating-conditioned BC (no player-rating
metadata in the datasets — L-effort scrape first); CQL-style
conservatism (no bootstrapped-Q failure mode here; targets are
terminal z).

Key sources: piKL (ICML 2022), RL-DiL-piKL (2210.05492), self-play
survey (2408.01072), RLPD (ICML 2023), AWAC (2006.09359), KataGo
(1902.10565), Stop Regressing / HL-Gauss (2403.03950), MuZero
Unplugged (2104.06294), Nie et al. (1708.01977), Anytime SH
(2411.07171), Sampled MuZero (2104.06303), Go-Exploit (2302.12359),
Grill et al. (2007.12509), Power-UCT (1911.00384), search-contempt
(2504.07757), E-MCTS (2210.13455), MAPLE (2605.24139).
