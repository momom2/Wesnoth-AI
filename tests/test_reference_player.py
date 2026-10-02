"""configs/reference_player.json through tools/reference_player.py: the
reference's memory travels in the match flags of its side, so no script
spells it out."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import reference_player  # noqa: E402


def test_the_reference_flags_carry_its_memory():
    ref = dict(reference_player.load())
    flags = reference_player.batch_flags("b", ref)
    assert flags[flags.index("--memory-b") + 1] == str(ref["memory_slots"])
    without = {k: v for k, v in ref.items() if k != "memory_slots"}
    assert "--memory-b" not in reference_player.batch_flags("b", without)
