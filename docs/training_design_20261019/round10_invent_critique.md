# Round 10 critique of the invent stance (herded argmax)

**Verdict: repairable.** H's decision rule (one uncoined match, adopt or retire by play) stands. Three definitions behind it are wrong or missing. Two readings of H's result rest on numbers nobody has measured. New checks are in `analysis/round10_invent_critique_checks.py` (*cc*, output beside it).

## 1. Provenance audit

| Number | Class | Check |
|---|---|---|
| Collapse: turn-1 distinct types 1.345 / 1.356 / 3.164; HHI 0.514 / 0.274; repeat 0.916 / 0.313 | M | *comp2* rerun: output byte-identical. *comp* not rerun (3.4 min, over the guideline); read from its `.out` |
| Label by d; side-turn 1, d ≥ 2: +0.0700 ± 0.0103; slope within identity +0.00267 ± 0.00048; C +18.4 ± 3.2; R +19.9 ± 3.1; C − R −1.4 ± 3.1; R + d +4.1 ± 1.7; C by faction | M | *rules* rerun (15 s): output identical |
| R's count feedback −2.85 Elo | D | Code read: shift × slope, first order. Correct as a report |
| Bound \|n − Q\| < \|O\| − 1 | D | Correct without masks. With masks, masked types can hold the positive W, so the legal maximum can sit below 1/\|O\|: unproven, synthetic only. Nothing rests on it |
| Lines 5.1 and 3.6; 37,800 games for $0.98; box A $3.16; $57.55 spent, $24.79 left; o ≤ 0.54; $5.51 short; host ≤ $2.62e-5 | D | Redone from the current design plus $0.98: all correct |
| HHI SE 0.0038 | D | 0.003 × √(6,402/4,000): correct |
| "78× the labels' information per game" | D, misapplied | 78× is the ratio for the whole band's 158.1 variance. On C's and R's switch sets, the measured SEs give **5.1× and 4.8×** [*cc* §2]. H's case does not depend on it |
| HA collects C's +18.4 in band | D, on an unmeasured premise | §2.1 |
| κ_comp ≈ 1 | **guessed** | It is a target, not a derivation. It carries S4's decode rule (§2.6) |
| HHI shift 0.2 in κ_comp's SE | guessed, power only | H measures the shift first |

## 2. What would make it fail

**2.1 Inside the band, HA does not act like C.** After the addition W = Q_t − n, so HA switches from a0 to a1 iff n0 − n1 > Q_t(a0) − Q_t(a1). C switches iff n0 − n1 > 0. ΔQ sums over all of a side-game's 23 recruits, not over its 4.3 near-ties. Earlier recruits outside the band, where a0 was strongly preferred, make ΔQ large, and ΔQ is unmeasured. In HA's own play, n tracks Q within the bound. So at HA's band decisions d ≈ ΔQ, and the residual decides the switch, not d. The +18.4 is priced on argmax-world histories, and the stance's own *feedback* script finds R's 2.2 switches already shift d by 1.39; HA changes about 8 recruits. So "> 5.1: composition past the band" and "within: the band holds the lever" are not identified readings. H compares two rules, nothing more.

**2.2 HA's T → 0 limit is not the reference decode.** The reference decode in `tools/raw_player.py` (`_choose`, `pick_index`) takes the argmax over joint (slot, hex) priors. HA's q_t(o) is the slot's mass summed over hexes, and the two orderings differ wherever a type's mass is split across hexes. Batch 1 shows split mass is common: 2.14 same-type, two-hex band pairs per side-game, against 4.33 different-type pairs [*cc* §1, b00]. How often the two orderings pick different types is unmeasured. So HA can depart at its first recruit, with W = 0: part of HA is a marginal decode, the class of the record's end_turn actor rule (+193 ± 14), which H cannot separate from herding. R0's "T → 0 equals argmax" test passes on synthetic q while the identity fails in production, and the CI replay at type ratio ≥ 1.5 can fail on correct code.

Repair: take q_t(o) from the type's best-hex joint prior, so that W = 0 reproduces the reference decode exactly, and run the T → 0 test on the compact arrays.

**2.3 W under overrides is unspecified.** With HA in π_0, band coins and audit pairs (P 0.25) override HA, and the head vetoes it. The spec subtracts 1 from o*, the type HA chose, whatever was played. Either reading of that breaks something:
- **W follows the played type** (W = Q − n stays true). The proof's step "the chosen W ≥ 1/\|O\|" no longer applies. I-herd's "played argmax W" flags every override. Worse, HA re-recruits the vetoed type at the next recruit. An audit label then prices a reordering, a veto cannot remove HA's composition, and "HA proposes, the head selects" fails.
- **W follows o*.** W stops tracking n − Q, and I-herd's max \|n − Q\| measures nothing.

The gate is one-sided too: a coin playing a runner-up recruit where the argmax is a move changes n with no W update, and a coin playing a move where the argmax is a recruit subtracts for a recruit never made. H is uncoined and unaffected; S1 with HA is blocked until the update rule, and what an audit label prices, are written down.

**2.4 Section 7's third move misreads S0(a).** The current design rejects "coherence" from the hook's lost moves (they reappear on the opponent, +0.50 ± 2.59) and records that S0(a) never tested labels against uncoined play. The supportable reading (`round9_challenge_hook_flips.out`): decisions changed at margins up to 1 nat cost −0.0094 ± 0.0041 z more each than in-band ones (z −2.28), so in-band labels do not price changes outside the band. The conclusion (H needs play) stands on that reason, not the stated one.

**2.5 Memory.** `own2`'s memory was trained on its own collapsed trajectories. Under HA it sees histories it has not seen since imitation, and the record's causal confusion (the memory continues its history's patterns) makes Q endogenous: the bound holds, but "the network's own mixture" is not a fixed target. H's composition statistics are play statistics, not the network's mixture.

**2.6 κ_comp.** At SE 0.03, "short of 1 by 2 SE" fires at any κ below 0.94, so N nearly always keeps HA and herds a q already partly diversified, an uncertified combination. Report κ; let round 3's match choose N's decode.

**2.7 Decision-table gaps.**
- "< −5.1: retired" is unconditional. If (h) refuses R, J_T is irrelevant and HA with J_H > 3.6 should stay eligible, as the middle row allows.
- H has no per-faction reading, though the stated weakest point is a faction signal (Loyalists HHI 0.706, C +1.1 ± 1.0). The stratified reading costs nothing.

**2.8 Literature.** My search found no herding decode of a game policy either; the mechanism is herding ([Welling 2009](https://icml.cc/imls/conferences/2009/papers/447.pdf); [herded Gibbs](https://arxiv.org/abs/1301.4168)), as cited. Uncited evidence against the premise: purifying a mixed strategy toward its mode often helps in poker, depending on its support ([Ganzfried, Sandholm & Waugh 2012](https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2012/papers/3E_4.pdf), abstract-level), and the record's argmax beats T = 1 sampling by +412. For recruits, only the in-band labels say a learned mixture is worth realising.

**Rulings.** HA is a hand-designed decode on a hand-chosen option set, the class of the −1.5 offset the user wants replaced by learning. Ruling (i) asks correctly; HA's learned end state is the reset, which waits for the optimizer discussion.

## 3. Strongest part

The diagnosis, from records already paid for: the same collapse in `own1` and `own2` against a human contrast; a count gradient inside unit identity (z 5.6); a parameter-free C that matches R's 43 values (−1.4 ± 3.1) and shares half its switches. Much of R's "unit identity" spread is "recruit the type you have fewer of". HA is deterministic, so coin exactness survives while W is a fixed function of the history, and H costs $0.98 with a branch for every outcome. Both scripts I reran reproduced exactly.

## 4. Repairs, in order

1. Define q_t from the best-hex joint priors, and test T → 0 on production arrays.
2. Specify W's update under coins, audit pairs and vetoes; state what an audit label prices; define I-herd on that.
3. Read H as a comparison of two rules, per faction, with the bottom row conditional on (h).
4. Correct section 7's S0(a) reason, the 78×, and the κ_comp rule.
