"""Seeded random number generator for reproducible game runs."""

from __future__ import annotations

import random


class SeededRNG:
    """Wrapper around random.Random for deterministic generation.

    Use fork() to create child RNGs for subsystems so their
    random streams don't interfere with each other.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._rng = random.Random(seed)

    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)

    def uniform(self, a: float, b: float) -> float:
        return self._rng.uniform(a, b)

    def random(self) -> float:
        return self._rng.random()

    def choice(self, seq):
        return self._rng.choice(seq)

    def shuffle(self, seq) -> None:
        self._rng.shuffle(seq)

    def gauss(self, mu: float, sigma: float) -> float:
        return self._rng.gauss(mu, sigma)

    def fork(self, label: str) -> SeededRNG:
        """Create a child RNG with a derived seed for a subsystem."""
        derived_seed = hash((self.seed, label)) & 0xFFFFFFFF
        return SeededRNG(derived_seed)
