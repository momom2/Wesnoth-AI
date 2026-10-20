# Critique of round 6, challenge stance

**Verdict: repairable.** ψ works as a luck corrector. As a decision score it rests on an assumption that randomization does not identify (F1). D2's no-information null is not mean zero (F2), and no plant covers the decision path (F3). New script: `analysis/round6_challenge_critique_exchange_sign.py` (`.out` beside it).

## 1. Provenance audit

| Number | Class | Check |
|---|---|---|
| Dice share 0.161; context −0.0042; kill uncertainty 50.5%, variance 0.0878 | M | Reran `round6_challenge_dice_share.py`: identical. Subsample bias stated. |
| Setup 0.110; r 0.897, 0.747, 0.700, 0.732; 109.7 fights; band 76.1; look-ahead per flip | M/D | Read in the round-5 outputs: correct. |
| Attack top 0.197; pairs 8.0–24.3, 21.5–58.5 | M bounds | Reran `round6_challenge_band_attacks.py`: identical; both bounds valid (`parity3` shares × `own1`'s count). |
| V_coin 1.1e-4 to 5.1e-4; ratio 319–1,412 | D | The arithmetic is right; the inference drawn from it is not (F1). |
| Excess variance 0.0041 at 100,000 games | D | σ²d/N is right, but it is the corrector's excess risk. The score's reliability is unknown. |
| P-dice band 1 ± 0.136 | D | **Wrong.** The SE carries a factor 0.5 that the plant does not have, since s_e = ±1 on every fight, so the band is ±0.096. The variance comes from the biased subsample, although D1 logs the exact laws (F3). |
| D2 SE 5.14 / 5.82 per arm; 7.27 for real − anti | D | Correct. |
| Per-switch lines 0.0034–0.0105 | D on a guess | They assume each arm switches m/2 times. The record says the split is lopsided (F2). |
| "Null: alignment below 0.0017–0.0052" | misread | That is the 2-SE line when â reads 0; when it reads just under 2 SE, the bound doubles. In gain terms the line is √2·SE = 7.3 Elo for any m, so a null closes a lever worth up to 7–14 Elo, which D4 (SE 2.57) could resolve. |
| D3 forecast SE 3.1–5.9 Elo | D | The formula is right. L is taken from the census bounds; the switch count is unknown. |
| $13.09; $3.42; $10.62; $45 | D | Re-added: correct. The ruling-(f) fallback, the value head at fight children (m = 2.0–3.9), is not priced. |
| Reliability 0.8 → 100,000 more games | C | No consequence is derived; it carries $2.47. |

## 2. Faults that carry decisions

**F1. The dice identify the slopes, not the score.**
- **Identified vs used.** Randomization identifies W only through each fight's outcome deviations, Q_e(o) − E_o Q_e. X(a) = ⟨W, E_a φ ⊗ t̃(a)⟩ uses the level: it assumes an attack in which no strike lands is worth what every non-attack is worth (score 0), and that zero-hit states on different defenders, each from its own landing hex (`attack_hex`, `tools/wesnoth_sim.py`), are worth the same. Nothing randomized tests this.
- **The stance's own plant shows the gap.** P-dice, δ·s(1[dies] − P_law), changes no decision's value, yet the regression weights it δ and X would credit δ·s·P(kill).
- **Consequences.** "Without confounding" holds for ψ as a corrector, not for X as a decision value. §0.1 and §1.5.1 fall: the 320–1,400 ratio compares variance the dice explain with variance the decisions cause, two estimands. It shows fights matter, not that X estimates Δ.
- **What X is.** A one-ply exchange evaluation with contextual weights: round 4's rejected "valuation over a hand-chosen basis" (`round4_synthesis.md`) plus embedding context, and a learned form of the combat-oracle attack bias the user retired on 2026-07-16 ("time for the policy to learn to walk on its own legs", `wesnoth_ai/constants.py`). Randomization answers the confounding objection, not the basis one; ruling (f) should quote both records.
- **Where the level holds by construction.** Pairs that differ only in weapon (same attacker, target and landing hex) share their zero-hit state. Their count is not on disk; a census can read it.

**F2. D2's null is not mean zero.**
- **How the arms split.** Real switches toward the attack when it is the runner-up and X > 0, away when it is on top and X < 0; anti the reverse. On the recorded fights the expected exchange favours the attacker in 0.749 by HP fraction and 0.547 by kill probability, against 0.246 and 0.099 unfavourable (exchange_sign.out; counter-weapon subsample, starting HP for the maximum).
- **What â then reads.** About μ(other→attack) − μ(attack→other): whether the network under-attacks at near-ties, a type-level bias the factored head's mass splitting makes plausible (end_turn is the measured case). "â > 2 SE: decision information" and "position dominates" both misread it.
- **The stated remedy does not work.** A match yields one total per arm, so it cannot be read per pair type.
- **Repair.** Run one real/anti pair per subtype (other→attack, attack→other, attack→attack): six arms, $2.86. Or read κ with coins (§4).
- **A smaller point.** "Exactly 0" holds to first order only. The performance-difference lemma is exact under each arm's own state distribution, and the two arms diverge: one makes favourable trades, the other declines them.

**F3. The plants miss the decision path and the sign.**
- **What P-dice checks.** It enters as its own column with its own sign (hex parity), so it checks the regression and the kill law only. A wrong attacker sign s_e in the 455 real columns, a token from the wrong decision, a law in the wrong world or a stale PCA basis all pass.
- **What nothing checks.** The plant has mean zero, so it is irrelevant to decisions; nothing tests X → switch → gain against a known answer (J5 recomputes).
- **Repair.**
  - Plant through the real feature map: δ·s_e·(φ_kill − E φ_kill)·t̃_k for one coordinate, booked with the attacker's side read from the record by separate code, and read per side.
  - For D2, a level plant: add δ·1[kill on an even hex] to z. Its value at each logged switch, δ·ΔP_kill, follows from the laws, so real − anti has a predicted reading.
  - Take every band from D1's logs.

## 3. Other faults

- **Distillation contradicts the premise.** §0.2 says the network cannot compute what the law decides, yet D5 needs it to imitate choices made from the law. If ruling (e) is no, the gain exists only through a closure that the stance's own argument predicts will fail. P-distill tests closure on the decision count, not on law-based choices.
- **Novelty.** The corrector half of claims 1–2 is MIVAT ([White & Bowling 2009](https://www.ijcai.org/Abstract/09/326)): a linear value function with feature cross-products, fitted in closed form to minimise an advantage-sum estimator's variance on past games, which is this ridge. AIVAT builds on it. Claim 3 is SPSA's two-sided perturbation ([Spall](https://www.jhuapl.edu/SPSA/)), which Stockfish's tuning plays as θ+ against θ− (from memory, not re-read). Claim 4 is the stance's own citation. What remains new is scoring decisions with a dice-identified value, and that is F1's unidentified step.

## 4. Strongest part

ψ as a corrector: mean zero for any W fixed before a game (the luck walk's proof), CPU only, learned weights in place of the seven hand-set columns, its kept ratio read on held-out games for free, useful to the current design's labels and matches whatever D2 shows. Second, a few-dollar read precedes every costly stage.

**Repair path.**
- Keep D1 as the corrector stage, crediting MIVAT.
- Drop X as a play-time rule.
- Enter ΔX as one column of the current design's coin ridge, in the slot that design gives a certified critic column. There the coins read its coefficient per pair type without bias, at CPU cost, subject to ruling (f).
- If a pre-read is still wanted, run D2 per subtype, restricted first to weapon pairs.
