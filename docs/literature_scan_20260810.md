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



# Search / MCTS improvements (agent report, verbatim)

None


# Imitation / value / self-play improvements (agent report, verbatim)

None
