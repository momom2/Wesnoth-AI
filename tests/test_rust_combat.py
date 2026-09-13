"""The Rust combat kernel equals the Python resolver.

`wesnoth_core.resolve_attack` (rust/wesnoth_core/src/combat.rs) must
reproduce `combat._resolve_attack_python` on fuzzed fights, field for
field including the per-strike checkup records and the RNG draws, and
must pass the `[mp_checkup]` fixture of test_combat_seed_alignment
bit-exactly. Skipped without the phase-6 wheel.
"""
from __future__ import annotations

import copy
import random

import pytest

from wesnoth_ai import combat as cb

try:
    import wesnoth_core as _core
    _HAS_KERNEL = hasattr(_core, "resolve_attack")
except ImportError:
    _HAS_KERNEL = False
pytestmark = pytest.mark.skipif(not _HAS_KERNEL, reason="wesnoth_core.resolve_attack not available")

_SPECIALS = ["magical", "marksman", "deflect", "backstab", "charge", "swarm", "drains",
             "plague", "poison", "slow", "petrifies", "firststrike", "berserk"]
_TYPES = ["blade", "pierce", "impact", "fire", "cold", "arcane"]


def _weapon(rng) -> cb.Weapon:
    k = rng.choice([0, 0, 1, 1, 2, 3])
    return cb.Weapon(name="w", damage=rng.randint(1, 14), number=rng.randint(1, 5),
                     range=rng.choice(["melee", "ranged"]), type=rng.choice(_TYPES),
                     specials=rng.sample(_SPECIALS, k), accuracy=rng.choice([0, 0, 10, -10]),
                     parry=rng.choice([0, 0, 10]))


def _unit(rng, side) -> cb.CombatUnit:
    max_hp = rng.randint(10, 60)
    return cb.CombatUnit(
        side=side, hp=rng.randint(1, max_hp), max_hp=max_hp, level=rng.choice([0, 1, 1, 2, 3]),
        experience=rng.randint(0, 40), max_experience=rng.randint(20, 80),
        alignment=rng.choice(list(cb.Alignment)),
        weapons=[_weapon(rng) for _ in range(rng.randint(1, 3))],
        resistance={t: rng.choice([60, 80, 100, 100, 120, 140]) for t in _TYPES},
        defense_pct=rng.choice([30, 40, 50, 60, 70]),
        is_slowed=rng.random() < 0.1, is_poisoned=rng.random() < 0.1,
        is_invulnerable=rng.random() < 0.03, is_fearless=rng.random() < 0.1,
        abilities=(["steadfast"] if rng.random() < 0.15 else []),
        is_undrainable=rng.random() < 0.1, is_unpoisonable=rng.random() < 0.1)


def _fight(rng):
    att, dfd = _unit(rng, 1), _unit(rng, 2)
    a_w = rng.randrange(len(att.weapons))
    d_w = rng.choice([None, -1] + list(range(len(dfd.weapons))))
    seed = "%08x" % rng.getrandbits(32)
    kw = dict(a_weapon_idx=a_w, d_weapon_idx=d_w,
              a_lawful_bonus=rng.choice([-25, 0, 25]), d_lawful_bonus=rng.choice([-25, 0, 25]),
              a_leadership_bonus=rng.choice([0, 0, 25, 50]), d_leadership_bonus=rng.choice([0, 0, 25]),
              a_backstab_active=rng.random() < 0.3, d_backstab_active=rng.random() < 0.3)
    return att, dfd, seed, rng.choice([0, 0, 3]), kw


def _same(a, b):
    return all(getattr(a, f) == getattr(b, f) for f in a.__dataclass_fields__)


def test_fuzzed_fights_equal_the_python_resolver():
    rng = random.Random(20260912)
    import wesnoth_core
    kernel = wesnoth_core.resolve_attack
    n_strikes = n_deaths = n_status = 0
    for _ in range(3000):
        att, dfd, seed, calls, kw = _fight(rng)
        py_att, py_dfd = copy.deepcopy(att), copy.deepcopy(dfd)
        py = cb._resolve_attack_python(py_att, py_dfd, rng=cb.MTRng(seed, calls), **kw)
        rs = cb._resolve_attack_rust(kernel, att, dfd, kw["a_weapon_idx"], kw["d_weapon_idx"],
                                     kw["a_lawful_bonus"], kw["d_lawful_bonus"], cb.MTRng(seed, calls),
                                     kw["a_leadership_bonus"], kw["d_leadership_bonus"],
                                     kw["a_backstab_active"], kw["d_backstab_active"])
        assert py.checkup_strikes == rs.checkup_strikes, (seed, kw)
        assert _same(py, rs), (seed, kw, py, rs)
        for a, b in ((py_att, att), (py_dfd, dfd)):
            assert (a.hp, a.experience, a.is_slowed, a.is_poisoned, a.is_petrified) == \
                (b.hp, b.experience, b.is_slowed, b.is_poisoned, b.is_petrified), (seed, kw)
        n_strikes += len(py.checkup_strikes) // 2
        n_deaths += (not py.attacker_alive) + (not py.defender_alive)
        n_status += py.defender_poisoned + py.defender_slowed + py.defender_petrified
    assert n_strikes > 10000 and n_deaths > 300 and n_status > 100


def test_random_int_equals_the_python_rng():
    import wesnoth_core
    rng = random.Random(7)
    for _ in range(500):
        seed = "%08x" % rng.getrandbits(32)
        calls = rng.choice([0, 1, 5])
        n = rng.randint(1, 6)
        assert wesnoth_core.random_int(int(seed, 16), calls, 0, n - 1) == \
            cb.MTRng(seed, calls).get_random_int(0, n - 1)


def test_mp_checkup_fixture_bit_exact_through_the_kernel():
    """The strict-sync fixture of test_combat_seed_alignment through
    the kernel: every attack's checkup records (chance, hit, the
    unclamped damage stat, death) must equal Wesnoth's recorded
    strikes and the Python resolver's."""
    import wesnoth_core
    from tests.test_combat_seed_alignment import FIXTURE
    from tools.replay_dataset import (_apply_command, _build_initial_gamestate, _find_unit_at,
                                      _setup_scenario_events, build_attack_context)
    from tools.replay_extract import extract_replay
    from tools.verify_mp_checkup import parse_replay
    recorded = parse_replay(FIXTURE)
    data = extract_replay(FIXTURE)
    gs = _build_initial_gamestate(data)
    _setup_scenario_events(gs, data.get("scenario_id", ""))
    k = checked = strikes = 0
    for cmd in data["commands"]:
        if cmd[0] == "attack":
            rec = recorded[k]
            k += 1
            att = _find_unit_at(gs, cmd[1], cmd[2])
            dfd = _find_unit_at(gs, cmd[3], cmd[4])
            seed = cmd[7] if len(cmd) > 7 and cmd[7] else "00000000"
            ctx = build_attack_context(gs, att, dfd, cmd[5], cmd[6] if len(cmd) > 6 else -1)
            args = dict(a_weapon_idx=ctx.a_weapon,
                        d_weapon_idx=ctx.d_weapon if ctx.d_weapon >= 0 else None,
                        a_lawful_bonus=ctx.a_lawful, d_lawful_bonus=ctx.d_lawful,
                        a_leadership_bonus=ctx.a_leadership, d_leadership_bonus=ctx.d_leadership,
                        a_backstab_active=ctx.a_backstab, d_backstab_active=ctx.d_backstab)
            py = cb._resolve_attack_python(copy.deepcopy(ctx.att_cu), copy.deepcopy(ctx.dfd_cu),
                                           rng=cb.MTRng(seed), **args)
            rs = cb._resolve_attack_rust(wesnoth_core.resolve_attack, ctx.att_cu, ctx.dfd_cu,
                                         args["a_weapon_idx"], args["d_weapon_idx"],
                                         args["a_lawful_bonus"], args["d_lawful_bonus"], cb.MTRng(seed),
                                         args["a_leadership_bonus"], args["d_leadership_bonus"],
                                         args["a_backstab_active"], args["d_backstab_active"])
            assert rs.checkup_strikes == py.checkup_strikes, k
            ours = rs.checkup_strikes
            assert len(ours) == 2 * len(rec.strikes), (k, len(ours), len(rec.strikes))
            for i, r in enumerate(rec.strikes):
                s_, d_ = ours[2 * i], ours[2 * i + 1]
                assert s_["chance"] == r.chance and s_["hits"] == r.hits, (k, i, s_, r)
                if r.hits:
                    assert s_["damage"] == r.damage, (k, i, s_, r)
                if r.dies is not None:
                    assert d_["dies"] == r.dies, (k, i, d_, r)
                strikes += 1
            checked += 1
        _apply_command(gs, cmd)
    assert checked == len(recorded) > 20 and strikes > 100


def test_kernel_path_raises_on_an_out_of_range_defender_weapon():
    """The kernel path must reject a defender weapon index past the end
    of the weapon list, as `_resolve_attack_python` does, instead of
    resolving the fight with no counter-attack. Needs the wheel because
    the assertion is that the REAL dispatch raises before the kernel
    runs -- tests/test_combat_rules.py covers the bridge on a stub."""
    rng = random.Random(20260913)
    att, dfd = _unit(rng, 1), _unit(rng, 2)
    bad = len(dfd.weapons)                       # one past the end
    assert cb.rust_combat_kernel() is not None, "kernel gate let a stale wheel through"

    with pytest.raises(IndexError):
        cb.resolve_attack(copy.deepcopy(att), copy.deepcopy(dfd), 0, bad, 0, 0,
                          cb.MTRng("deadbeef"))
    with pytest.raises(IndexError):
        cb._resolve_attack_python(copy.deepcopy(att), copy.deepcopy(dfd), 0, bad, 0, 0,
                                  cb.MTRng("deadbeef"))

    # Positive control: the in-range index still resolves through the
    # kernel and matches the oracle, so the raises above are about the
    # index and not about the fixture or the dispatch being broken.
    k_att, k_dfd = copy.deepcopy(att), copy.deepcopy(dfd)
    p_att, p_dfd = copy.deepcopy(att), copy.deepcopy(dfd)
    k = cb.resolve_attack(k_att, k_dfd, 0, bad - 1, 0, 0, cb.MTRng("deadbeef"))
    p = cb._resolve_attack_python(p_att, p_dfd, 0, bad - 1, 0, 0, cb.MTRng("deadbeef"))
    assert _same(k, p), "kernel and oracle disagree on the in-range fight"
