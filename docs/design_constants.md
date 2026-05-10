# Design constants — derivations and rationale

This document collects numerical constants used in our model,
training, and search code that AREN'T arbitrary tuning knobs but
are DERIVED from a specific assumption or measurement. It exists
because re-deriving a constant from a docstring fragment costs
real time, and because constants get copy-pasted across files
faster than their justifications do.

## How to use this document

**When a constant is derived (from math, from a measurement, or
from a fixed external standard), add an entry here.** Required:

- The constant's name + value (e.g. `cliffness_max = 0.577`)
- Where it's defined in code (file:line or symbol name)
- The derivation: math, measurement protocol, or external source
- A "why this number specifically" note — what it would mean if it
  were different, what bounds it on each side

**When you find yourself writing "where does this value come from?"
in a code comment, the rationale belongs HERE, not in the
comment.** A two-line code comment + cross-reference here is fine
and preferred.

**Tuning knobs that are arbitrary defaults DON'T belong here.**
This doc is for derived / measured / canonical constants only.
Things like learning rate, c_puct, dirichlet_alpha live in
`constants.py` with their own comment block; if they're
defended by experiment, the experiment goes in BACKLOG.md.

---

## Table of contents

- [Value head / cliffness](#value-head--cliffness)
- [Encoder normalizations](#encoder-normalizations)

---

## Value head / cliffness

### `cliffness_max = 0.577` (≈ 1/√3)

**Defined:** `tools/mcts.py` (`MCTSConfig.cliffness_max`), also
referenced indirectly via `_BOOTSTRAP_PRIOR_VAR = 1/3` (the same
quantity squared).

**Derivation:** `cliffness = std(Z(s))` is the standard deviation
of the network's predicted categorical value distribution over
atoms in [V_MIN, V_MAX] = [-1, +1]. The MAXIMUM possible std for
ANY distribution supported on [-1, +1] is achieved by the
two-point distribution placing mass 0.5 on each endpoint, which
has std = 1.0. The maximum std for a UNIFORM-ish distribution
(maximum entropy under a fixed support) is the std of the
continuous uniform on [-1, +1]:

```
σ_uniform = sqrt((V_MAX - V_MIN)² / 12) = sqrt(4/12) = 1/√3 ≈ 0.5774
```

The discrete uniform on K=51 atoms over [-1, +1] gets to within
3 decimal places of this; verified in
`test_distributional_value.test_cliffness_high_when_distribution_spread`.

**Why this number specifically:** 0.577 is the practical "I have
no idea what's going to happen" upper bound — corresponding to
the network outputting uniform-over-atoms logits, which is the
max-entropy state of a freshly-initialized C51 head. Cliffness
above 0.577 means the network's distribution is BIMODAL or
otherwise more spread than uniform — possible in principle but
unusual to see during training. We use 0.577 as the normalizer
for the adaptive sim budget (cliffness/cliffness_max in [0, 1])
and as the "fully uncertain" reference point for the Bayesian
bootstrap shrinkage (where cliffness² ≈ 1/3 = prior variance
gives 50/50 blend with the prior).

**Cross-references:**

- `tools/mcts.py` `MCTSConfig.cliffness_max`: comment cites
  this doc for the full derivation.
- `tools/mcts.py` `_BOOTSTRAP_PRIOR_VAR = 1/3`: same quantity
  squared; the variance of the uniform prior on [-1, +1].
- `model.py` `VALUE_V_MIN, VALUE_V_MAX, VALUE_N_ATOMS`: define
  the support that 1/√3 is computed against. If the support
  changes, this number changes (proportional to support range).

---

## Encoder normalizations

### `HP_NORM, MOVES_NORM, EXP_NORM, ...`

**Defined:** `constants.py`, re-exported from `encoder.py`.

**Derivation:** these are ROUGH normalizers ("typical max"
values), not derived constants — chosen to keep encoded inputs
in [0, ~1] for stable training. They live in `constants.py`
with their own comment block explaining "scale rationale".
Listed here only for completeness; if you wonder where these
come from, check the `constants.py` block — they're era-mod
overridable in one place.
