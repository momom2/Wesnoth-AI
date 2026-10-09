"""The core's Mersenne Twister draws equal `combat.MTRng`'s, the
generator replay extraction reads a recorded command's draws with: the
same seed and call count give the same integer."""
from __future__ import annotations

import random

import pytest

from wesnoth_ai import combat as cb

try:
    import wesnoth_core as _core
    _HAS_KERNEL = hasattr(_core, "random_int")
except ImportError:
    _HAS_KERNEL = False
pytestmark = pytest.mark.skipif(not _HAS_KERNEL, reason="wesnoth_core.random_int not available")


def test_random_int_equals_the_python_rng():
    import wesnoth_core
    rng = random.Random(7)
    for _ in range(500):
        seed = "%08x" % rng.getrandbits(32)
        calls = rng.choice([0, 1, 5])
        n = rng.randint(1, 6)
        assert wesnoth_core.random_int(int(seed, 16), calls, 0, n - 1) == \
            cb.MTRng(seed, calls).get_random_int(0, n - 1)
