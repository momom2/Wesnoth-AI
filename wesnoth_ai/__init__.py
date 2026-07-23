"""Wesnoth AI — core library package.

Shared source for both training paths (in-process simulator and the
eval-only live-Wesnoth bridge): game state (`classes`), the encoder,
the transformer model, trainers, reward shaping, combat/visibility
logic, and the policy adapters. See CLAUDE.md "Architecture" for the
module map.

Application/entry code lives outside this package: `tools/` holds the
scripts (self-play, eval, replay tooling), `main.py` is the setup CLI,
and `tests/` holds the test suite.
"""
