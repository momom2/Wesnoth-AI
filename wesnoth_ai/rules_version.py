"""The rules version of pre-encoded records: what built their positions,
besides the observation epoch (wesnoth_ai/constants.py) and the encoding's
switches, which the records name on their own.

The Rust core's phase (`game_core._CORE_PHASE`, the wheel's `__phase__`)
changes with every change to the core, most of which build every position
as before: a faster path, a new query, a view built in Rust instead of
Python. The rules version changes only when a rule the core applies or what
an encoded position holds changes. Each phase maps to its rules version
here, a version being named after the first phase that built positions
under it. Whoever bumps the phase adds it to the table and says which kind
of change it is: tests/test_rules_version.py fails while the adapter's
phase is missing.

A sequence manifest records the phase whose wheel built it (`core_phase`);
its fingerprint carries that phase's rules version
(`tools/preencode_sequences.encoding_fingerprint`). The trainer refuses
records whose rules version is not the running adapter's, and loads records
of another phase under the same rules version. Phases before 31 are not in
the table: records they built are refused.
"""
from __future__ import annotations

from typing import Dict

# Core phase -> the rules version its positions are built under.
RULES_VERSION_OF_PHASE: Dict[int, int] = {
    # Every unit of a side rests from each of its turn starts, the first
    # included (commit f52900c).
    31: 31,
    # The core builds the decision's packed legality masks beside the
    # encoding when asked for them (3009a12, the encoding's arrays
    # unchanged), and the view's units (f1cdbed, equal to the Python
    # builder's, tests/test_core_view.py).
    32: 31,
    # The long calls compute without the GIL (669280c): the same
    # computations, moved out of the Python-facing code.
    33: 31,
    # The inference server's requests parsed and staged in the core, its
    # memory books, and the view cache that reuses a unit's Python object
    # while its record is unchanged (feature/generation-speed-4; every
    # refresh checked against a whole rebuild under WESNOTH_CHECK_VIEWS):
    # nothing that builds a position.
    34: 31,
    # The label builder in the core and the record's core encoded in place
    # (feature/record-walk-rust) build the same positions and labels, byte
    # for byte (tests/test_core_label.py).
    35: 31,
    # The inference server's mask staging without the hand-set prior
    # biases (fix/remove-hand-priors): nothing that builds a position.
    36: 31,
}


def rules_version_of(phase: int) -> int:
    """The rules version of positions core phase `phase` built; ValueError
    for a phase the table does not hold."""
    try:
        return RULES_VERSION_OF_PHASE[int(phase)]
    except KeyError:
        raise ValueError(f"core phase {phase} has no rules version in wesnoth_ai/rules_version.py: "
                         f"add it, with whether it changed how a position is built") from None


def current_rules_version() -> int:
    """The rules version of the adapter this process runs (its `_CORE_PHASE`)."""
    from wesnoth_ai.game_core import _CORE_PHASE
    return rules_version_of(_CORE_PHASE)


def check_records_rules(manifest: Dict, where: str) -> None:
    """SystemExit unless the records of `manifest` (a sequence manifest,
    its `core_phase`) were built under the running adapter's rules."""
    phase = manifest.get("core_phase")
    if phase is None:
        raise SystemExit(f"{where}: the manifest names no core phase; encode the records again")
    want = current_rules_version()
    try:
        got = rules_version_of(phase)
    except ValueError as e:
        raise SystemExit(f"{where}: {e}; this code reads rules version {want}") from None
    if got != want:
        raise SystemExit(f"{where} was built under rules version {got} (core phase {phase}); this code reads "
                         f"rules version {want}: encode the records again")
