"""Wesnoth's synced random numbers, and the combat tables the Python
side reads.

The Rust core resolves every fight (rust/wesnoth_core/src/combat.rs,
core_attack.rs). What stays here:

  - `MTRng`, a bit-exact `std::mt19937` seeded as `mt_rng::seed_random`
    (src/mt_rng.cpp) seeds it, which replay extraction uses to read the
    draws a recorded command made; `seed_int_of`, a synced command's seed
    as the integer the core seeds its generator with.
  - `DAMAGE_TYPES`, the damage types in the order of the scraped
    resistance tables, and `TOD_DEFAULT_CYCLE`, the default time-of-day
    cycle with each step's lawful bonus.

The C++ uses `std::mt19937` (32-bit Mersenne Twister) seeded by a single
`uint32_t`. numpy's `np.random.MT19937(seed)` cannot stand in for it:
numpy seeds via `SeedSequence`, which expands the int through additional
mixing before filling the state, producing a DIFFERENT output stream from
`std::mt19937(uint32)`. Verified empirically:
  std::mt19937(0x9260e745) first output = 3761859111
  numpy MT19937(0x9260e745) first output = 499511960
"""

from __future__ import annotations


def seed_int_of(seed_hex: str) -> int:
    """The 32-bit seed of a synced command's hex string; 42 when the
    string is not hex (`if (!(s >> std::hex >> new_seed)) { new_seed = 42; }`)."""
    try:
        return int(seed_hex, 16) & 0xFFFFFFFF
    except (ValueError, TypeError):
        return 42


class MTRng:
    """Bit-exact std::mt19937 — Wesnoth's combat RNG.

    Per the Wesnoth source, one synced command (e.g. one [attack])
    creates a fresh `mt_rng`, calls `seed_random(seed_str, call_count)`,
    and from then on all draws come from that same MT state. Each draw
    is a raw `uint32_t` from `mt_()`; combat hit-rolls take it mod 100.

    We implement std::mt19937 directly with the canonical Knuth seeding
    (https://en.wikipedia.org/wiki/Mersenne_Twister), which matches what
    `std::mt19937 mt(seed)` produces in C++. We can NOT use numpy's
    `MT19937(seed)`: numpy seeds via `SeedSequence`, producing a
    different stream.
    """

    # std::mt19937 constants
    _N         = 624
    _M         = 397
    _MATRIX_A  = 0x9908B0DF
    _UPPER     = 0x80000000
    _LOWER     = 0x7FFFFFFF
    _MULT_INIT = 1812433253

    def __init__(self, seed_hex: str, call_count: int = 0):
        seed_int = seed_int_of(seed_hex)
        self.seed_int = seed_int          # replayed by the Rust kernel
        self._mt = [0] * self._N
        self._idx = self._N           # forces twist on first draw
        self._seed(seed_int)
        # mt_.discard(call_count): pull and drop `call_count` outputs.
        for _ in range(call_count):
            self._next_uint32()
        self.calls = call_count

    def _seed(self, seed: int) -> None:
        """Knuth init: state[0]=seed, state[i]=(C * (state[i-1] xor
        (state[i-1]>>30)) + i) & 0xFFFFFFFF for i in 1..N-1."""
        self._mt[0] = seed & 0xFFFFFFFF
        for i in range(1, self._N):
            prev = self._mt[i - 1]
            self._mt[i] = (self._MULT_INIT * (prev ^ (prev >> 30)) + i) & 0xFFFFFFFF
        self._idx = self._N

    def _twist(self) -> None:
        for i in range(self._N):
            y = (self._mt[i] & self._UPPER) | (self._mt[(i + 1) % self._N] & self._LOWER)
            self._mt[i] = self._mt[(i + self._M) % self._N] ^ (y >> 1)
            if y & 1:
                self._mt[i] ^= self._MATRIX_A
        self._idx = 0

    def _next_uint32(self) -> int:
        if self._idx >= self._N:
            self._twist()
        y = self._mt[self._idx]
        self._idx += 1
        # Tempering
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= (y >> 18)
        return y & 0xFFFFFFFF

    def get_next_random(self) -> int:
        """One raw uint32 draw, mirroring `mt_rng::get_next_random()`."""
        v = self._next_uint32()
        self.calls += 1
        return v

    def get_random_int(self, low: int, high: int) -> int:
        """Inclusive [low, high]. Wesnoth uses
        `next_random() % (max+1)` for the [0, max] form (`rng.cpp`
        `get_random_int_in_range_zero_to`), which we mirror.
        """
        span = high - low + 1
        return low + (self.get_next_random() % span)


# Damage type indexing. Order matches Wesnoth's WML keys order in our
# scraped resistance tables.
DAMAGE_TYPES = ["blade", "pierce", "impact", "fire", "cold", "arcane"]


# Time-of-day cycle (default 6-step). lawful_bonus is +25 in day,
# -25 at night, 0 at twilight (the engine's `generic_combat_modifier`
# reads it).
TOD_DEFAULT_CYCLE = [
    ("dawn",          0),
    ("morning",      25),
    ("afternoon",    25),
    ("dusk",          0),
    ("first_watch", -25),
    ("second_watch",-25),
]


__all__ = ["MTRng", "seed_int_of", "DAMAGE_TYPES", "TOD_DEFAULT_CYCLE"]
