"""Tests for the seeded RNG utility."""

from dungeon_builder.utils.rng import SeededRNG


def test_deterministic():
    rng1 = SeededRNG(42)
    rng2 = SeededRNG(42)
    for _ in range(100):
        assert rng1.randint(0, 1000) == rng2.randint(0, 1000)


def test_different_seeds_differ():
    rng1 = SeededRNG(42)
    rng2 = SeededRNG(99)
    values1 = [rng1.randint(0, 1000) for _ in range(20)]
    values2 = [rng2.randint(0, 1000) for _ in range(20)]
    assert values1 != values2


def test_fork_deterministic():
    rng1 = SeededRNG(42)
    rng2 = SeededRNG(42)
    f1 = rng1.fork("geology")
    f2 = rng2.fork("geology")
    for _ in range(50):
        assert f1.uniform(0, 1) == f2.uniform(0, 1)


def test_fork_different_labels_differ():
    rng = SeededRNG(42)
    f1 = rng.fork("geology")
    f2 = rng.fork("intruders")
    values1 = [f1.randint(0, 1000) for _ in range(20)]
    values2 = [f2.randint(0, 1000) for _ in range(20)]
    assert values1 != values2
