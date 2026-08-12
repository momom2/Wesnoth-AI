# Learned planning abstractions — literature review (2026-08-12)

Context: the 2026-08-12 diagnosis established that micro-action-granularity
search (one turn ≈ 14 decisions, branching ~100+, 16–64 sims) has an
effective depth of 1–2 decisions — a fraction of one turn — so the
policy-improvement operator has no tactical signal to distill and the
value head carries the entire burden. The user's constraint for any
remedy: **abstractions must be learned from the agent's own play and
falsifiable against the simulator; structural priors (locality,
goal-conditioning, turn boundaries) are acceptable, play priors are
not.** Three parallel literature dives (subgoal/option search; learned
reachability/backtracking; principled local decomposition) each
returned a design recommendation with a cheap falsifiable gate. This
file preserves the load-bearing conclusions; the full surveys lived in
the 2026-08-12 session.

A meta-finding common to all three threads: **no published system
combines adversarial two-player play + stochastic dynamics +
entity-set states + learned abstractions + search integration.** Our
setting is ahead of the literature in that combination — no recipe to
copy, no proven negative either. The bit-exact simulator removes each
family's dominant published failure mode (multi-step dynamics-model
error for options; unfalsifiable backward generation for reachability
— replaced by forward sim counterfactuals; unverifiable isolation for
region planning — replaced by measured closure/CE).

---

## Bet 1 — OptionZero-style self-distilled options (search depth)

**Anchor: OptionZero (Huang, Wu et al., ICLR 2025 oral, arXiv
2502.16634; code github.com/rlglab/optionzero).** An option head on
the shared trunk predicts, per state, the policy's own most likely
action *sequence*; the longest prefix with joint probability > 0.5
(cap L, and for us: capped at the turn boundary — a structural prior)
is the "dominant option", added as one extra candidate edge in MCTS
with two-stage selection and linked visit statistics. Option content
is learned purely from the agent's own executed trajectories (an
auxiliary CE target on data we already store). Atari evidence: +131pp
mean human-normalized over MuZero at 50 sims; learned option lengths
1.69–2.03; ~75% of options are repeated primitives ("harvests policy
confidence, does not invent strategy").

Why it fits: directly multiplies effective search depth by the option
length; the paper's dominant failure (multi-step *dynamics network*
error) does not exist for us — the sim executes options exactly, and
an illegal step truncates (itself a training signal). Options ending
at the pre-end_turn state also land the value head on a cleaner
evaluation point (full side commitment visible).

Known failure modes to defend against (from the option literature):
option collapse when content is shaped only by return (OptionZero's
grounding in executed sequences avoids this); dishonest accounting
(count simulator micro-steps, never tree edges — "What Matters in
Hierarchical Search", arXiv 2406.03361); hierarchy stops paying if
the value head becomes strong.

**GATE (free, hours, no training):** over logged self-play states,
run the current policy and measure the empirical dominant-sequence
length — largest l with product of policy probabilities of the
actually-executed next l actions > θ, sweeping θ ∈ {0.3, 0.5, 0.7},
recording turn-boundary truncations. Mean ~1.0–1.2 → shelve; ≥1.5
with a fat tail of 3–6 runs → the depth multiplication is real
(~1 week to implement).

Second-generation successor if this lands: Tuero/Buro/Lelis (ICML
2025, arXiv 2506.07255) — subgoal codes mined from own MCTS trees
(solved AND failed), VQ-VAE generator, no expert data.

Explicitly rejected for now: kSubS/AdaSubS generative *state*
subgoals (adversarial interleaving + heavy state generation);
Director/FeUdal/Hieros manager-worker stacks (no search integration;
three independent negative-results papers on hierarchical world
models); DADS (predictability objective = passivity anti-prior for
combat); Option-Critic (collapse).

## Bet 2 — Predicate-conditioned achievement head C(s, g, k) (backtracking)

**Formulation (composed from HER arXiv 1707.01495 / contrastive RL
arXiv 2206.07568 / HIQL arXiv 2307.11949 / GS-HER arXiv 2606.09476 /
DECSTR-line predicate goals arXiv 2204.05141 / HInt null
counterfactuals arXiv 2505.03172):** a small head on the existing
trunk estimating P(predicate g true within k turns | s, current
self-play distribution), g from a minimal predicate vocabulary —
initially `unit_dies(u)`, `unit_reaches(u, h)`,
`village_owner_flips(v)` — represented as (predicate embedding ⊕
entity-token pointer ⊕ params), k ∈ {1..8} explicit (not a
discount). Training labels are FREE: hindsight scans over stored
games (every death/arrival/flip is an achieved goal with known Δt);
~10^5–10^6 stored states yield 10^7+ tuples. Human corpus gives
human-play reachability as a bonus.

Key semantics choice: probability-of-achievement (contrastive/
occupancy family) NOT shortest-path distance (quasimetric family,
which is provably deterministic-only — wrong for stochastic combat).
Stratify sampling on ¬g(s)/contestedness or degenerate positives
dominate (HInt's warning).

**Backtracking without a backward generative model** (Recall-Traces-
style predecessor generation is unfalsifiable at our state
complexity): the screening-unit query is answered by **sim-exact
counterfactuals** — ΔC under entity-token masking proposes, actual
sim-side unit removal + policy rollouts disposes. The learned head
proposes, the simulator verifies: falsifiable by construction. This
is the deleted detector-advice faculty rebuilt from play data.

**GATE (few GPU-hours, laptop or one cheap box):** MVP =
`unit_dies(u)` only, k ∈ {1,2,3}, trunk frozen, pure predictor.
Pre-registered: (i) beats a logistic baseline on trivial features
(HP, adjacent enemies, terrain) with calibration below a set ECE
bar; (ii) counterfactual probe — on ~100 held-out low-C states, rank
screen candidates by masked ΔC, remove top-ranked in sim, N rollouts:
rank correlation between predicted ΔC and realized Δfrequency must be
significantly positive; (iii) any search integration gets a placebo
arm (shuffled C outputs) — mandatory given the detector-advice
history. Honest thin ice: entity-level partial goals in adversarial
games are unpublished (evidence ceiling: ≤5-block manipulation).

## Bet 3 — Influence-augmented battle-region search (locality)

**The good-regulator intuition is formalized, sharper than the
theorem itself, by Influence-Based Abstraction** (Oliehoek et al.,
AAAI 2012 + JAIR 2021 arXiv 1907.09278): a local model is
value-LOSSLESS iff augmented with the exact influence point — the
distribution of boundary-crossing events conditioned on a
d-separating statistic of *local* history (the required "model of
what impinges" is minimal: boundary flux, not a world model).
Degradation law (Congeduti et al., AAMAS 2021, arXiv 2011.01788):
value loss ≤ 2h²·|R|·√(2·KL) of the learned influence predictor —
so a CE-trained boundary predictor optimizes exactly the quantity
that bounds planning loss, and its held-out CE is a certified
monitor. The uncertainty-triggered fallback exists in the
literature: He et al. (IJCAI 2022, arXiv 2201.11404) gate
local-vs-global simulator choice per simulation on a KL-bound
statistic, with online predictor refresh (kills the distribution-
shift failure of offline training).

Architecture if built: regions = connected components of the k-turn
interaction graph (rules-derived, config k, no play prior); influence
vocabulary = per-turn boundary-crossing events; local MCTS over
region units with several sampled exterior continuations per node
(the depth-limited-solving robustness pattern, Brown & Sandholm
NeurIPS 2018); leaves scored by the GLOBAL value net — never local
material sums (Koller & Parr 1999: value does not factor even when
dynamics do). CGT decomposition search (Berlekamp/Müller) is the
exact-math precedent and warns: greedy hottest-region play is
provably suboptimal; cross-region allocation is the hard part.

**GATE (free, ~a day of scripting, no training):** run the region
detector over logged games; measure the **closure rate** — fraction
of k-turn windows (k ∈ {2,3,4}) in which detected regions see zero
consequential boundary crossings. Rare closure → the thread is
refuted for this meta at scripting cost. Where closure fails, the
tiny boundary predictor's held-out CE is the certified
value-at-risk proxy; pre-register that local/full search move
agreement degrades monotonically with that CE.

Biggest open risk (all threads agree): all guarantees fix the
exterior/opponent policy; self-play drift and two-way strategic
coupling (opponent responds to the local plan) have no end-to-end
guarantee anywhere in the literature. Staleness tolerance (DIALS
NeurIPS 2022 Thm 1) and menu-robustness are mitigations, not proofs.

---

## Synthesis: how they compose, and the order

The three bets are complementary layers, not rivals: options buy
tactical *depth* along confident lines; C(s,g,k) is the *goal/
backtracking* primitive (and the training signal any future subgoal
generator needs); region search is *where* deep tactics get computed
affordably. All three gates run on already-logged data at ~zero
cost. Sequencing decision 2026-08-12: run the gates first, build at
most one winner.
