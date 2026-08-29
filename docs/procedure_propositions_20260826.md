# Decision-procedure propositions — synthesis of four redteamed designs

Date: 2026-08-26. Inputs: four independently devised designs for the play-time decision procedure and its learning loop, each with an independent redteam verdict (scores 4, 5, 6, 6 of 10). This document ranks them for the owner's decision, states each verdict honestly, proposes repairs, and recommends the first validation spend. Register: nothing here is validated. Every benefit named is a hypothesis until its match runs; the only verdicts are games.

---

## 1. What all four designs converged on

All four arrived at the same chassis:

- **The search object is the whole side-turn plan, graded at commitment boundaries** — not a per-decision tree. Shared motivation: 100-400 forwards per side-turn buys no depth per decision; sim steps cost ~13% of a forward and forks ~0.5 ms; the repo's own evidence points the same way (imitation seed plays K≈12 actions/turn, per-decision self-play collapsed to K 2-4.5, the turn-commitment-search probe validated boundary-graded plan refinement).
- **The prior proposes, the value head ranks, the simulator is exact in between** (the sampled-policy-improvement condition: coverage from the prior, not correctness).
- **Sequential halving** allocates the leaf budget over the candidate population; leaves arrive as sibling batches.
- **Exact combat-outcome enumeration** (the repo's oracle-verified DP in tools/combat_outcomes.py) replaces sampling at chance nodes wherever affordable.
- **The distillation target is constructed so the leg-4 operator class** (per-step flattening; sheltered unvisited mass) **is impossible at the target level**, checkable in closed form before any run.
- **Value targets stay anchored on realized outcomes z.** The imitation seed appears only as opponent and yardstick, retiring by the standing Elo-gate rule; a league of frozen checkpoints supplies the cross-play that pure self-play's collinearity lacks (the leg-5 identification lesson).
- **Exploration and steerability live in config**; the same procedure generates data, is evaluated, and deploys.

The convergence is partly explained by shared inputs (same problem statement, same postmortems, turn-commitment search already in-repo), so it corroborates the chassis rather than proving it. The redteam killed no part of this chassis. Every fatal flaw lands in the trust layer above it: how far to believe the value head when it referees plan comparisons.

## 2. The failure axis the redteam found in all four

Each verdict's central attack is the same finding restated: **treatment-correlated value bias**. The head's measured defects — leg-5 trunk rotation onto raw unit count, leg-3 tempo blindness — are biases aligned with exactly the axes on which competing plans differ (trade vs hold, tempo spent vs pieces kept). From the verdict on design 3: "pairing cancels only bias SHARED across the challenger/incumbent pair. The dangerous class is treatment-correlated bias — bias that co-varies with what distinguishes the plans." From design 1's: "THE STATISTICAL TEST CONTROLS THE ERROR SOURCE THAT DOESN'T MATTER AND IS BLIND TO THE ONE THAT DOES. ... CRN then makes it WORSE: collapsed SE lets a biased A_hat clear the LCB test confidently."

Two consequences bind every proposition below:

1. **The only bias-sensitive referee is a played-out game.** Statistics internal to the search (variance, redraw agreement, martingale consistency, exactness at chance nodes) cannot see a consistent wrong ranking. Two external referees exist: the Elo gate (end verdict, standing rule) and **forked branch play-outs** — take a graded plan comparison, fork the game, play both branches to termination with the same procedure, compare the grader's margin to realized outcomes. The branch mechanism originates in design 4; its verdict exposed a censoring flaw (sampling only near-ties misses confident inversions — "confidently-wrong comparisons have large margins and never enter the ledger"), which the margin-stratified version fixes. **Margin-stratified branch play-outs are promoted here to a shared prerequisite: no training leg under any proposition until this instrument has measured the head's plan-ranking error against realized outcomes.**
2. **Target-operator algebra is necessary but not sufficient.** All four designs check their update's per-step mass algebra in closed form (the leg-4 methodology as a design-time gate — keep this). All four verdicts note the same limit: the realized update is SGD on a 15M shared trunk, and generalization moves the policy at states the algebra never touched. Gates and tripwires stay load-bearing; no proposition's argument replaces them.

## 3. Merge consideration

The chassis is common property; the distinct load-bearing ideas are: incumbent-anchored certify-or-abstain (design 3), exact-chance scenario grading + branch play-outs (design 4), the legality partition (design 2), and the exact-chance control variate on value targets (design 1). A full merge was considered and declined: propositions 1 and 2 differ in their grading engine (sampled projected redraws vs exact scenario mixtures), and that difference is what staged validation should adjudicate, not a design-time guess. Two partial merges are real and adopted across the board: the branch play-out instrument (from design 4) repairs the calibration flaw in proposition 1 and the censoring flaw in proposition 2 simultaneously; and the abstaining mixture update (proposition 1's) replaces proposition 2's per-decision credit assignment, which its verdict showed to be ill-defined. Designs 2 and 1 survive as component donors, not standalone propositions.

---

## Proposition 1 — incumbent-anchored turn-plan tournament with certify-or-abstain distillation

From: "Certified turn-plan tournament with abstaining mixture distillation" (redteam score 6). Rank: 1.

### Core idea

The incumbent plan is the policy's own sampled turn (the existing turn-commitment-search spine). Challengers come from prefix perturbations and whole-turn resamples. Sequential halving escalates grading **fidelity** — projection depth past the boundary (2 half-turns minimum, 6 for finalists), both sides played closed-loop by the raw policy — rather than visit counts. The final challenger is compared to the incumbent as a paired difference under shared randomness; the challenger plays only if its margin clears a noise band; otherwise the incumbent plays (abstention). Learning distills only certified turns, as a conservative-policy-iteration mixture (Kakade-Langford): target = (1−β)·prior + β·certified action, β scaled by the measured margin, zero on abstained turns. Value trains on terminal outcomes only. Opponents come from a config-weighted league. The safety shape: grading errors before certification can only forfeit an improvement; a wrong displacement requires a false certification.

### Redteam verdict

Fatal flaws, verbatim cores:

1. The improvement/non-degradation argument — a hard deliverable — "is literally an empty field," and when reconstructed it has the treatment-correlated-bias hole (quoted in §2). Also: "abstention is per-step safe but the certified data distribution is filtered toward exactly the positions where the head is already right, which is where improvement matters least."
2. "The false-certification rate alpha — the load-bearing calibrated quantity of the whole safety case — is calibrated circularly. ... Deep redraws of a biased evaluator agree with the bias with less variance."
3. "The specified implementation of the CPI step does not implement CPI." Verified against the repo: wesnoth_ai/trainer.py:1014 divides the policy loss by total visits, so a single experience with count = β has β cancel exactly — full-strength distillation, margin scaling gone. (The game-weight path at trainer.py:1069/1210 is the available fix.)
4. "The 'provably never net-flattening' claim rests on a false inequality" (entropy can rise when the certified action had low prior mass), and the operator algebra does not bind the SGD-generalized policy.

Weaknesses that matter: only one chance node per line is exactly branched, so multi-fight plans face repair storms and inflated noise bands — abstention concentrates on sharp tactical turns, where the per-decision baseline is strongest; the β_max derivation ("dElo/dKL from step-1 arm gaps") is incoherent as stated; proposal diversity can collapse as distillation sharpens the prior; several constants are picked, not derived.

Strengths the verdict granted: the budget-shape analysis matches the project's measured history; incumbent-relative grading is a genuine structural response to leg-5 ("never consume an absolute value read"); implementability is substantially grounded — tools/turn_search.py and tools/turn_policy.py exist and work, abstained value-only experiences already produce zero policy loss, and "the new code is the halving/certification layer and the league, not a rewrite"; the design metabolized legs 3/4/5 by name; validation is cheap-first and mostly killable.

### Repairs

1. **Write the improvement argument** (it is sketched here; the full statement is part of step 0). Play-time: committed play equals the policy's own plan except where a certified challenger displaces it; pre-certification errors forfeit upside only; displacement error rate is the false-certification rate, now **measured against played-out branch pairs** rather than assumed. Learning: per-update target moves only certified-turn mass, every non-certified action retains (1−β) of its prior mass, β = 0 without certification; conservative-policy-iteration bound on the visited distribution. Explicit assumptions: (i) false-certification rate at the audited level, per axis (unit-count delta, tempo/K delta, fog delta); (ii) projection depth materializes the consequences the head misprices — payoffs beyond ~6 half-turns are conceded as unimprovable by this procedure (missing upside, not degradation, contingent on (i)); (iii) all guarantees are visited-distribution and target-level; SGD generalization is guarded empirically (gates, tripwires, branch audit), not by the algebra.
2. **Fix the step size**: carry β as a per-sample game-weight multiplier outside the visit-normalized CE; add a unit test asserting the realized gradient scales with β through the production trainer path.
3. **Drop the entropy claim**; state and audit the correct property: uniform (1−β) scaling of all non-certified mass, no sheltered mass, order preserved — the closed-form audit over logged states (leg-4 methodology) checks this before any run.
4. **Replace the circular calibration** with the branch instrument: sample certified and near-margin-abstained comparisons stratified across the margin distribution, play both branches to termination, estimate false-certification rate vs margin and per axis. This is design 4's mechanism imported as measurement, not a change to the play procedure.
5. **β cap**: the step-size shape stays the conservative-policy-iteration closed form in measured-margin units; the per-leg KL cap is a pre-registered config constant, admitted as chosen — flagged as an open item against the no-hand-picked-constants rule, with a derivation path (from the audited false-certification rate through the CPI bound) available once the instrument has data.
6. **Chance handling**: v1 keeps sampled redraws with adaptive count; telemetry splits certification rate by fight count so the predicted tactical-turn abstention is observed, not assumed. Event-keyed shared tapes (design 1's mechanism) are imported only if a measured paired-SE reduction ≥2x justifies them. Exact outcome-class branching waits on the shared outcome-instantiation function (see proposition 2).
7. **Budget honesty**: the repair reserve counts inside the 400-forward cap; re-entry rates are predicted from the DP's class-mismatch probabilities per committed attack (computable) and the reserve sized from that.

### Validation, cheapest first

- **Step 0 — free, local.** (a) Closed-form operator audit of the target on logged states: kill/block on any flattening or sheltered mass. (b) β-scaling unit test through the real trainer path (catches the normalization cancellation). (c) Improvement argument written and pre-registered with its assumptions.
- **Step 1 — the mandated match, ~$10-25, days.** Seed net (2516k-b-294k-l4-0k) both sides. This procedure at the mid budget vs (A) per-decision Gumbel sequential-halving search at equal measured forwards per side-turn — the decisive arm, 200 games; (B) turn-commitment search as shipped with multi-turn projection, 60-100 games; (C) raw-policy floor, 40 games. Pre-registered one-sided test vs arm A (n=200 resolves roughly +40 Elo). **Kill the whole family** if ≤50% vs arm A cannot be rejected. Secondary kill: certification rate <5% of side-turns (no learning signal exists). Log true games/hour and per-turn forward accounting; every cost model below is re-based on the measurement.
- **Step 2 — branch-audit instrument, ~$2-6, CPU-dominant.** ~300 start states; finalist pairs stratified across margins; both branches played to termination; unit of independence = start state (the leg-5 instrument lesson). Deliverables: false-certification rate vs margin, uncensored sibling-ranking inversion rate, axis splits. **Kill any training leg under any proposition** if confident-margin comparisons on the trade/tempo axes invert at or above chance — the head cannot referee; run league cross-play value repair first.
- **Step 3 — short leg, ~$15-30.** 50-100k steps from the seed, full standing tripwires as crash barriers, new telemetry (certification rate and margins, β/KL per update, fog-differential, per-axis certification splits). Gate: pre-registered Elo vs seed, two-stage (40 games; extend to 120 if the interval straddles zero — rule fixed before launch). Kill on a non-positive gate or on certification rate trending to zero (proposal-diversity collapse); postmortem before redesign.
- **Step 4 — standing protocol.** ~250k-step leg and its 40-game verdict, per the resume plan. Then attribution ablations and the steerability smoke test (risk-functional and forced-opener config flips must change behavior measurably).

---

## Proposition 2 — exact-chance scenario grading with paired branch games

From: "Turn-plan halving with exact-chance grading and a near-tie branch league" (redteam score 6). Rank: 2.

### Core idea

Candidates are whole turn-plans sampled without replacement by stochastic beam search over the prior. Sequential halving's escalated resource is **exact-chance scenario resolution**: early rounds grade each plan at its modal combat outcome; each round doubles the enumerated outcome scenarios per survivor, with exact probabilities from the DP; a plan's score is the exact-probability mixture of scenario C51 reads — chance-sampling variance removed by construction. Execution is commit-and-replan. Plan comparisons that end close spawn **branch games**: fork the game, play both plans out to termination — simultaneously curriculum, decorrelated value data, and a standing measurement of the sibling-ranking assumption. Opponents by prioritized fictitious self-play weighting w(1−w) with a Bradley-Terry Fisher-information derivation. Update: regularized completed policy target (Grill et al.); value anchored on z plus a consistency term toward the exact scenario mixture.

### Redteam verdict

Fatal flaws, verbatim cores:

1. "Residual-mass-at-worst completion is a built-in anti-attack bias. ... This is the leg-3 passivity collapse rebuilt into the grader itself — and 'assign residual to worst' is a hand-picked rule with no derivation."
2. "Boundary-only grading with a tempo-blind value head repeats a known, already-countered failure. ... the plan is to detect the collapse after paying for it again" (the shipped counter — multi-turn projection — was dropped).
3. "The continuous A1 monitor is blind to the failure mode that actually occurred": near-tie-only sampling is censored; "confidently-wrong comparisons have large margins and never enter the ledger."
4. "The improvement property as stated is not the property the cited theorems give": plan-suffix scores are not action values, the factored projection loses correlations, and "'bounded per step, detectable, not compounding' is exactly the regime in which legs 3 and 4 each shed ~500 Elo."

Weaknesses that matter: prioritized-league weighting retires the frozen prior exactly when its decorrelation is most needed; execution-replan costs undercounted (realistic ~250-350 forwards/side-turn at M=16); the sim has **no outcome-forcing capability** — scenario replay needs new machinery touching parity-critical territory, weeks of work before step 1 can run; branch pairs are one Bernoulli comparison each, far weaker per pair than advertised.

Strengths the verdict granted: exact chance enumeration as the halving fidelity axis is "the one novelty claim that holds up"; the cost model is honest against measured historical throughput; the league/identifiability diagnosis is correct; the leg-4 operator class is structurally absent; three of five validation steps fire before serious GPU spend.

### Repairs

1. **Residual mass**: renormalize over evaluated scenarios (condition on the enumerated set) and report the residual; escalate finalists until residual is below a derived threshold (design 2's derivation: truncated mass δ shifts a score by <2δ; δ = C51 atom width / 4 puts truncation below value resolution). No completion-at-worst; no completion-at-best; the remaining representativeness error is measured in the instrument step.
2. **Restore multi-turn projection for finalists** (the leg-3 counter): the last two survivors are graded H half-turns past the boundary, not boundary-only. This moves its cost profile toward proposition 1's; that is the price of the documented failure.
3. **De-censor the monitor**: branch games sampled across the stratified margin distribution (the shared instrument of §2), not near-ties only. Near-ties remain the cheap bulk; a fixed fraction of confident-margin pairs makes the ledger bias-sensitive.
4. **Adopt the abstaining mixture update** from proposition 1 for the policy (population used for selection and for the branch ledger; distillation only of the selected plan's executed, certified decisions). This removes the ill-defined plan-suffix credit assignment its verdict identified; the consistency value term keeps its kill switch (dropped unless it improves cross-play holdout ranking).
5. **League**: keep the w(1−w) weighting for match-making; retired anchors are replaced by own-lineage checkpoints selected for measured style distance (K-median, attack rate, trade rate), with a config floor weight until the branch ledger demonstrably carries identifiability. The human seed still retires by the standing rule; no permanent tether.
6. **Outcome instantiation**: build `apply_outcome(fork, outcome_key)` as state surgery from the enumerated key (HP, status, XP, advancement) — not a change to the strike-by-strike combat path. Unit tests: instantiated class states match the DP's distribution and invariants; cross-check against sampled real resolutions. This function is shared with proposition 1's later exact-branching upgrade; build once, behind proposition 1's step-1 gate.

### Validation, cheapest first

- **Step 0 — free.** Operator audit (shared); DP class-probability vs Monte-Carlo test (largely exists); residual-threshold derivation written to docs/design_constants.md.
- **Step 1 — shared instrument, ~$2-6.** The margin-stratified branch audit of §2 does not require outcome instantiation (branches run real dice) and is bought under proposition 1's ladder regardless. Its axis-split results directly test this proposition's grader assumptions too.
- **Step 2 — outcome-instantiation function.** Weeks of implementation with parity tests. Kill: any distribution mismatch vs the DP blocks everything downstream.
- **Step 3 — its own mandated match, ~$10-25.** Same harness and arms as proposition 1's step 1, this proposer/grader swapped in; same pre-registered kill.
- **Step 4 — grader ablation, ~$5-10.** Exact scenario mixtures vs sampled redraws on identical candidate sets at equal forwards: does the exactness machinery pay Elo? Kill the machinery (not the chassis) if no.
- **Steps 5-6.** Short leg and standing protocol as in proposition 1, plus a branch-fraction ablation (does the curriculum/value data pay?).

Sequencing note: this proposition's first strength information sits behind weeks of parity-critical work. It is the routed next spend if proposition 1's match fails **diagnosably from redraw noise** (abstention concentrated on multi-fight turns) — exact grading is precisely the counter to that mechanism — or the follow-on spend if proposition 1 passes and the grader ablation is worth buying.

---

## Proposition 3 — legality-partition contingent plans (component donor)

From: "Contingent turn-plan tournament with exact chance grading" (redteam score 5). Rank: 3.

### Core idea

Branch a plan's evaluation only where the enumerated combat outcome changes the **legal continuation set** (deaths, advancements — not HP noise); within a cell, continue at conditional-expected HP. The same partition defines commitment: execute until a reveal crosses a cell, then re-plan. Truncate branch mass below a threshold derived from C51 atom resolution. Policy target: regularized completed target; value: z anchor plus exact one-turn bootstrap mixture.

### Redteam verdict

Fatal flaws, verbatim cores:

1. "The improvement argument's load-bearing assumption (A1: small boundary ranking error) is the exact assumption this project has twice measured to fail, and the design removes the shipped counter" (boundary-only grading, projection dropped, null plan seeded — the leg-3 trap restated).
2. The exactness claim fails on focus-fire plans: either cells proliferate (2-5x budget blowout) or HP folds within cells and later kill probabilities are computed at fictional fractional HP — "'exact' becomes 'exact for a fiction'."
3. "The objects being ranked are never trained. ... the loop's fixed point is self-consistency (value ranks what policy plays; policy plays what value ranks), not strength."

Also: the claimed observable-state fork does not exist (WesnothSim.fork is god-view; the cited primitive was misattributed), and the engineering scope before any kill gate is large.

### Disposition and salvage

Not a standalone proposition. After its natural repairs (projection restored, partition refined to split at later-attack kill-threshold crossings on the same defender, counterfactual boundaries grounded by branch play-outs) it converges into propositions 1-2. Three components are adopted now:

- **The legality partition as the replan trigger**: re-search exactly when a realized outcome changes the legal continuation set — the crispest definition among the four designs; adopted into proposition 1's execution loop.
- **The truncation-threshold derivation** (mass below atom-width/4 is below value resolution): adopted into proposition 2's residual rule; goes to docs/design_constants.md.
- **The within-cell smoothness probe** (value-head spread across same-cell HP/trait variants): a cheap calibration of how much boundary-state detail the head is sensitive to; added to the shared instrument battery.

A standalone leg is warranted only if the top two propositions' graders underperform in ablations for reasons the partition specifically addresses.

---

## Proposition 4 — paired-tape certification with fold bounds (component donor)

From: "Paired-tape turn-plan tournament with measured-advantage conservative distillation" (redteam score 4). Rank: 4.

### Core idea

Exact enumeration everywhere with worst-case bias bounds on unexpanded mass; paired comparisons under event-keyed shared random tapes; a challenger plays only if its paired-advantage lower confidence bound clears zero plus the exact fold-bias bounds; the same certified number sets the distillation step size through the conservative-policy-iteration closed form.

### Redteam verdict

Fatal flaws, verbatim cores:

1. "CERTIFICATION IS ARITHMETICALLY UNCLEARABLE ON THE TURNS THAT MATTER. ... the threshold is >= 0.8-2.4 on a value scale whose total range is 2. Realistic one-turn advantages are ~0.02-0.2. ... its own arithmetic predicts its own kill."
2. "THE STATISTICAL TEST CONTROLS THE ERROR SOURCE THAT DOESN'T MATTER AND IS BLIND TO THE ONE THAT DOES" (variance vs treatment-correlated bias; shared tapes make a biased margin certify confidently).
3. "THE FORWARDS BUDGET IS MISCOUNTED AT THE FIRST ROUND" (honest accounting ~500-900 forwards per contested turn, or the candidate population shrinks to where the coverage assumption fails).

### Disposition and salvage

Not viable as stated; its certification rule is unclearable by its own arithmetic, and repairing it requires a value-smoothness assumption of exactly the hand-tuned class the constraints forbid. Three components are adopted:

- **The exact-chance control variate on value targets** — target = z plus a correction with exactly zero mean under the enumerated combat probabilities, unbiased under any critic bias. The redteam: "salvageable independently of the rest of the design," matching the established control-variate lineage for poker-style variance reduction, on machinery that exists. Adopt in any training leg; unit test (zero-mean check) is free. One stated fix carries over: folded/unevaluated realizations map to their class representative so the zero-mean property survives.
- **Event-keyed shared random tapes** for paired projections — adopt only if the measured paired-SE reduction is ≥2x (its own criterion, which the redteam endorsed).
- **β = 0 when play could not certify** — already the spine of proposition 1.

---

## Recommendation: what to validate first

**Fund proposition 1's step 0 and step 1 now.** Reasoning in expected-Elo-per-dollar terms, stated with its honest limit — the Elo numerator of every proposition is unmeasured, so the ranking is dominated by the cost and ordering of decisive information:

1. **The first dollar should buy the family-level bit, and proposition 1 sells it cheapest.** All four propositions share the chassis; the mandated equal-compute match against per-decision Gumbel search adjudicates that chassis for all of them at once. Proposition 1 reaches that match on existing machinery (tools/turn_search.py, tools/turn_policy.py, projection, repair triggers — per its verdict, "the new code is the halving/certification layer and the league, not a rewrite"): roughly $10-25 and days. Proposition 2 needs weeks of parity-critical outcome-instantiation work before the same bit; propositions 3-4 are component donors.
2. **Conditional on a pass, its Elo ceiling is not obviously lower than the alternatives'.** It keeps the multi-turn projection that is the shipped counter to the one collapse mode (leg 3) the boundary-only variants re-expose, and its fatal flaws were disproportionately claim and implementation errors with identified fixes (the step-size cancellation is verified against trainer.py:1014 and fixed by the game-weight path), while proposition 2's flaws sit in the grader core and their repairs move its cost toward proposition 1's anyway.
3. **The failure branches are cheap and already routed.** Worst case ~$25-40 total: the match fails to beat per-decision search, the family dies, and the standing fallback (resume leg 5 from escrow with the shipped configuration) proceeds unchanged — with the operator audit, the control variate, and the league mechanism still applicable to it. A diagnosable redraw-noise failure routes the next spend to proposition 2's exact grading. A pass funds the ~$2-6 branch-audit instrument — which any proposition needs before any training leg, and which prices proposition 2's assumptions as a side effect — and then a ~$15-30 short leg behind two cheap gates. Cumulative cost to the first training-leg Elo verdict: ~$50-80 at historical box rates.

Sequencing after a pass: branch audit (shared), short leg with the repaired update, standing 250k protocol; build the outcome-instantiation function and run proposition 2's grader ablation only behind those gates.

## Decision records

- Rejected: worst-case fold-bias certification thresholds (proposition 4), because at affordable expansion they exceed realistic one-turn advantages by an order of magnitude — the rule abstains always.
- Rejected: residual-scenario-mass assigned to the worst evaluated outcome (proposition 2 as submitted), because it structurally penalizes attack-heavy plans — the leg-3 passivity mechanism relocated into the grader.
- Rejected: boundary-only grading without multi-turn projection (propositions 2 and 3 as submitted), because leg 3 measured the exploit and the counter is already shipped.
- Rejected: near-tie-only sampling for the ranking monitor, because the measured failure mode (leg 5) is confident inversion, which never enters a near-tie ledger.
- Rejected: certification calibrated by deep redraws of the same net, because it is circular under bias; calibration referee = played-out branch pairs.
- Rejected: full four-way design merge, because propositions 1 and 2 differ in the grading engine and staged matches should decide that, not synthesis.
- Open item: the per-leg KL cap on distillation is a pre-registered config constant without a derivation; a derivation path exists through the branch-audited false-certification rate and the conservative-policy-iteration bound once the instrument has data.