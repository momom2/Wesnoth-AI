"""The unit row of version 2 (docs/input_facts_design_20261124.md), which the
Rust core builds under `unit_row_version` 2: the sidebar's columns against
unit_stats.json on hand-built positions; the attack window's lines against
the engine's arithmetic worked here, and on played games against the core's
own fight statistics and the strikes the fights then made; version 1's rows
as the prefix; the legal attacks, which are on a hex that touches the
attacker's; and a checkpoint widened to version 2, which computes what its
source computed and which the loop's tools play, pass and step."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from wesnoth_ai import encoder as enc  # noqa: E402
from wesnoth_ai.game_core import game_core_class  # noqa: E402

pytestmark = pytest.mark.skipif(game_core_class() is None, reason="wesnoth_core.GameCore not available")

from helpers.parity_games import (  # noqa: E402
    FACTION_IDS, core_of, record, unit_id_at, unit_row, unit_types, vocab_of,
)

# A movetype's key for each terrain class, in the hex tokens' class order
# (classes.Terrain, then the mushroom grove and the reef).
CLASS_KEYS = ("castle", "cave", "deep_water", "flat", "forest", "frozen", "hills", "impassable", "mountains",
              "sand", "shallow_water", "swamp_water", "unwalkable", "village", "fungus", "reef")
UNREACHABLE = 99              # a movement cost the unit cannot pay (movetype.hpp UNREACHABLE)
NIGHT = 4                     # tod_start_index of first watch: turn 1 at lawful bonus -25
RECIPE = dict(relevant_set=True, fog_hides_enemy_villages=True, terrain_multi_hot=True, observation_parity=True,
              relevant_set_version=2)
F, D, W = enc.ROWS2_FIGHT_AT, enc.FIGHT_DIRECTIONS, enc.PARITY_WEAPON_SLOTS


def rows(cs, type_to_id, version=2, **more):
    return cs.encode_raw(type_to_id=type_to_id, faction_to_id=FACTION_IDS, unit_row_version=version,
                         **RECIPE, **more)


def fights_of(row) -> np.ndarray:
    """A row's fight columns as [direction, weapon slot, column]."""
    return row[F:].reshape(D, W, enc.FIGHT_COLS)


def expected_tables(unit_type: dict, slowed: bool = False) -> tuple:
    """(chance to be hit / 100, one over the movement cost) per terrain
    class from a unit_stats.json type. A negative defence entry is a cap,
    read as its size (movetype.cpp `config_to_min`)."""
    defence = [abs(unit_type["defense"].get(k, 100)) / 100 for k in CLASS_KEYS]
    move = []
    for k in CLASS_KEYS:
        cost = unit_type["movement_costs"].get(k, UNREACHABLE)
        cost = 2 * cost if slowed and cost < UNREACHABLE else cost
        move.append(0.0 if cost >= UNREACHABLE else 1.0 / cost)
    return np.asarray(defence, dtype=np.float32), np.asarray(move, dtype=np.float32)


def test_the_layout_adds_up():
    assert (enc.UNIT_FEAT_DIM_PARITY, enc.UNIT_FACTS_COLS, enc.UNIT_FIGHTS_COLS) == (109, 34, 198)
    assert enc.UNIT_FEAT_DIM_ROWS2 == 341 == enc.ROWS2_FIGHT_AT + enc.UNIT_FIGHTS_COLS
    assert enc.observation_widths(True, 2).unit_feats == 341
    assert enc.observation_widths(True, 1) is enc.PARITY_WIDTHS
    with pytest.raises(ValueError, match="observation_parity"):
        enc.observation_widths(False, 2)
    with pytest.raises(ValueError, match="unit_row_version"):
        enc.observation_widths(True, 3)


def test_the_sidebar_columns_are_the_units_level_defence_and_movement():
    """Each unit's level, its chance to be hit where it stands and on each
    terrain class, and one over its movement costs, from unit_stats.json:
    a level-2 leader on its keep, a level-1 Spearman on grass and one on
    hills, a level-0 Vampire Bat, a Merman in shallow water, a slowed
    Horseman (whose costs double, and whose forest entry is a cap), and an
    enemy's row, which carries them too."""
    types = unit_types()
    special = {(1, 3): "Kh", (7, 2): "Hh", (9, 4): "Ww", (17, 3): "Kh"}
    units = [("Lieutenant", 1, 1, 3, True), ("Spearman", 1, 5, 3, False), ("Spearman", 1, 7, 2, False),
             ("Vampire Bat", 1, 3, 5, False), ("Merman Fighter", 1, 9, 4, False), ("Horseman", 1, 11, 3, False),
             ("Lieutenant", 2, 17, 3, True), ("Skeleton", 2, 13, 3, False)]
    data = record(units, special=special, fog=False)
    cs = core_of(data)
    cs.apply_command(["init_side", 1])
    cs.core.update_unit(unit_id_at(cs, 11, 3), {"statuses": ["slowed"]})
    raw = rows(cs, vocab_of(u[0] for u in units))
    assert raw.unit_feats.shape == (len(units), enc.UNIT_FEAT_DIM_ROWS2)
    stands = {(1, 3): "castle", (5, 3): "flat", (7, 2): "hills", (3, 5): "flat", (9, 4): "shallow_water",
              (11, 3): "flat", (17, 3): "castle", (13, 3): "flat"}
    for name, _side, x, y, _leader in units:
        t = types[name]
        row = unit_row(raw, unit_id_at(cs, x, y))
        defence, move = expected_tables(t, slowed=(x, y) == (11, 3))
        assert row[enc.ROWS2_LEVEL_AT] == np.float32(t["level"] / enc.LEVEL_NORM), name
        assert row[enc.ROWS2_STANDS_AT] == np.float32(abs(t["defense"][stands[(x, y)]]) / 100), (name, x, y)
        assert np.array_equal(row[enc.ROWS2_DEFENCE_AT:enc.ROWS2_MOVE_AT], defence), name
        assert np.array_equal(row[enc.ROWS2_MOVE_AT:enc.ROWS2_FIGHT_AT], move), name
    assert types["Vampire Bat"]["level"] == 0 and types["Lieutenant"]["level"] == 2
    horse = unit_row(raw, unit_id_at(cs, 11, 3))
    assert horse[enc.ROWS2_MOVE_AT + CLASS_KEYS.index("flat")] == np.float32(0.5), "slowed: flat costs 2"
    assert horse[enc.ROWS2_MOVE_AT + CLASS_KEYS.index("mountains")] == 0.0, "it cannot enter mountains"
    assert horse[enc.ROWS2_DEFENCE_AT + CLASS_KEYS.index("forest")] == np.float32(0.7), "the forest cap of 70"


def test_recruit_rows_carry_the_types_level_and_tables():
    """A recruit option shows its type's level and tables; it stands
    nowhere, so "where it stands" and its fights are 0."""
    types = unit_types()
    data = record([("Lieutenant", 1, 0, 3, True), ("Lieutenant", 2, 19, 3, True)],
                  recruits={1: ["Spearman", "Mage", "Merman Fighter", "Horseman"]})
    cs = core_of(data)
    cs.apply_command(["init_side", 1])
    names = ["Lieutenant", "Spearman", "Mage", "Merman Fighter", "Horseman"]
    raw = rows(cs, vocab_of(names))
    old = rows(cs, vocab_of(names), version=1)
    assert raw.recruit_feats.shape == (4, enc.UNIT_FEAT_DIM_ROWS2)
    assert np.array_equal(raw.recruit_feats[:, :enc.UNIT_FEAT_DIM_PARITY], old.recruit_feats)
    for row, name in zip(raw.recruit_feats, raw.recruit_types):
        defence, move = expected_tables(types[name])
        assert row[enc.ROWS2_LEVEL_AT] == np.float32(types[name]["level"] / enc.LEVEL_NORM), name
        assert row[enc.ROWS2_STANDS_AT] == 0.0
        assert np.array_equal(row[enc.ROWS2_DEFENCE_AT:enc.ROWS2_MOVE_AT], defence), name
        assert np.array_equal(row[enc.ROWS2_MOVE_AT:enc.ROWS2_FIGHT_AT], move), name
        assert not row[enc.ROWS2_FIGHT_AT:].any()


# ---------------------------------------------------------------------------
# The attack window's lines, by the engine's arithmetic
# ---------------------------------------------------------------------------

def combat_modifier(alignment: str, lawful_bonus: int) -> int:
    """src/actions/attack.cpp `generic_combat_modifier` (1.18.4)."""
    return {"lawful": lawful_bonus, "neutral": 0, "chaotic": -lawful_bonus,
            "liminal": 25 - abs(lawful_bonus)}[alignment]


def round_damage(base: int, bonus: int, divisor: int) -> int:
    """src/utils/math.hpp `round_damage` (1.18.4): to the nearest, a half
    away from the base damage, at least 1."""
    if base == 0:
        return 0
    rounding = divisor // 2 - (0 if bonus < divisor or divisor == 1 else 1)
    return max(1, (base * bonus + rounding) // divisor)


def strikes_of(attacker: dict, weapon: dict, defender: dict, lawful_bonus: int, *, leadership: int = 0,
               slowed: bool = False, terrain: str = "flat") -> list:
    """[damage per hit, strikes, chance to hit] / their divisors for a
    weapon without a special that changes them: the time of day and the
    leadership added, times the defender's resistance (attack.cpp
    `battle_context_unit_stats`), halved for a slowed striker; the chance
    to hit is the defender's chance to be hit on its terrain."""
    modifier = 100 + combat_modifier(attacker["alignment"], lawful_bonus) + leadership
    damage = round_damage(weapon["damage"], modifier * defender["resistance"][weapon["type"]],
                          20000 if slowed else 10000)
    return [damage / enc.WEAPON_DAMAGE_NORM, weapon["number"] / enc.WEAPON_STRIKES_NORM,
            defender["defense"][terrain] / 100]


def expected_line(attacker: dict, w: int, defender: dict, lawful_bonus: int, **attacker_kw) -> np.ndarray:
    """One weapon slot's 11 columns against one enemy; the enemy answers
    with its one weapon of the attack's range, if it has one."""
    weapon = attacker["attacks"][w]
    line = np.zeros(enc.FIGHT_COLS, dtype=np.float32)
    line[0] = 1.0
    line[1:4] = strikes_of(attacker, weapon, defender, lawful_bonus, **attacker_kw)
    answers = [k for k, a in enumerate(defender["attacks"]) if a["range"] == weapon["range"]]
    assert len(answers) <= 1, "this helper reads a defender with one weapon per range"
    if answers:
        line[4] = 1.0
        line[5:8] = strikes_of(defender, defender["attacks"][answers[0]], attacker, lawful_bonus)
        line[8 + answers[0]] = 1.0
    return line


def _contact(extra_units=(), **record_kw):
    """A Spearman of side 1 at (5, 3) with a Skeleton north of it and an
    Elvish Archer south-east of it, at first watch."""
    units = [("Lieutenant", 1, 1, 3, True), ("Spearman", 1, 5, 3, False), ("Lieutenant", 2, 18, 3, True),
             ("Skeleton", 2, 5, 2, False), ("Elvish Archer", 2, 6, 4, False), *extra_units]
    data = record(units, fog=False, tod_start_index=NIGHT, **record_kw)
    cs = core_of(data)
    cs.apply_command(["init_side", 1])
    return cs, vocab_of(u[0] for u in units)


def test_the_fight_columns_are_the_attack_windows_lines():
    """At first watch a lawful Spearman fights at -25 and a chaotic
    Skeleton at +25. Its spear does 7 x 75% x 40% = 2.1, so 2 a hit, three
    times at 60%, and the Skeleton's axe answers 7 x 125% = 8.75, so 9;
    its javelin does 6 x 30% = 1.8, so 2, once, and the Skeleton, which has
    no ranged weapon, does not answer. The Elvish Archer answers the spear
    with its sword and the javelin with its bow. The lines sit in the
    slots of the hexes the enemies stand on, north and south-east."""
    from tools.abilities import hex_neighbors
    types = unit_types()
    spearman, skeleton, archer = types["Spearman"], types["Skeleton"], types["Elvish Archer"]
    cs, vocab = _contact()
    raw = rows(cs, vocab)
    assert hex_neighbors(5, 3)[0] == (5, 2) and hex_neighbors(5, 3)[2] == (6, 4)
    lines = fights_of(unit_row(raw, unit_id_at(cs, 5, 3)))
    board = -25
    spear = np.array([1, 2 / 40, 3 / 6, 0.6, 1, 9 / 40, 3 / 6, 0.6, 1, 0, 0], dtype=np.float32)
    javelin = np.array([1, 2 / 40, 1 / 6, 0.6, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    assert np.array_equal(lines[0, 0], spear) and np.array_equal(lines[0, 1], javelin)
    assert np.array_equal(lines[0, 0], expected_line(spearman, 0, skeleton, board))
    assert np.array_equal(lines[0, 1], expected_line(spearman, 1, skeleton, board))
    assert np.array_equal(lines[2, 0], expected_line(spearman, 0, archer, board))
    assert np.array_equal(lines[2, 1], expected_line(spearman, 1, archer, board))
    assert lines[2, 0, 8] == 1.0 and lines[2, 1, 9] == 1.0, "the sword answers the spear, the bow the javelin"
    assert not lines[:, 2].any(), "the Spearman has two weapons"
    assert not lines[[1, 3, 4, 5]].any(), "no enemy stands on the other four hexes"
    # Only the side's units with an attack left carry lines.
    for x, y in ((5, 2), (6, 4), (1, 3), (18, 3)):
        assert not fights_of(unit_row(raw, unit_id_at(cs, x, y))).any(), (x, y)
    assert not raw.recruit_feats[:, F:].any()


def test_slow_leadership_and_a_spent_attack_show_in_the_lines():
    """A slowed Spearman's hit is halved (7 x 30% / 2 = 1.05, so 1); beside
    its Lieutenant it fights at +25 (7 x 100% x 40% = 2.8, so 3); once it
    has attacked it carries no line."""
    types = unit_types()
    spearman, skeleton = types["Spearman"], types["Skeleton"]
    cs, vocab = _contact()
    uid = unit_id_at(cs, 5, 3)
    cs.core.update_unit(uid, {"statuses": ["slowed"]})
    line = fights_of(unit_row(rows(cs, vocab), uid))[0, 0]
    assert line[1] == np.float32(1 / 40)
    assert np.array_equal(line[1:4], np.asarray(strikes_of(spearman, spearman["attacks"][0], skeleton, -25,
                                                           slowed=True), dtype=np.float32))
    cs.core.update_unit(uid, {"statuses": [], "has_attacked": True})
    assert not fights_of(unit_row(rows(cs, vocab), uid)).any()

    led, vocab = _contact(extra_units=[("Lieutenant", 1, 4, 3, False)])
    line = fights_of(unit_row(rows(led, vocab), unit_id_at(led, 5, 3)))[0, 0]
    assert line[1] == np.float32(3 / 40)
    assert np.array_equal(line[1:4], np.asarray(strikes_of(spearman, spearman["attacks"][0], skeleton, -25,
                                                           leadership=25), dtype=np.float32))


def test_a_line_is_there_for_exactly_the_enemies_the_mask_offers():
    """Under fog, an ambusher in forest on a hex that touches the unit's is
    uncovered by that unit, so the side sees it and its line is there; a
    petrified enemy beside the unit cannot be attacked and has none. The
    core's packed mask offers the same one target."""
    from wesnoth_ai.server_priors import core_packed_masks
    units = [("Lieutenant", 1, 1, 3, True), ("Spearman", 1, 5, 3, False), ("Lieutenant", 2, 18, 3, True),
             ("Elvish Ranger", 2, 5, 2, False), ("Skeleton", 2, 6, 4, False)]
    data = record(units, fog=True, special={(5, 2): "Gs^Fp"})
    cs = core_of(data)
    cs.apply_command(["init_side", 1])
    cs.core.update_unit(unit_id_at(cs, 6, 4), {"statuses": ["petrified"]})
    raw = rows(cs, vocab_of(u[0] for u in units), masks=True)
    assert unit_id_at(cs, 5, 2) in raw.unit_ids, "a unit beside an ambusher uncovers it"
    lines = fights_of(unit_row(raw, unit_id_at(cs, 5, 3)))
    assert lines[0, 0, 0] == 1.0 and lines[0, 1, 0] == 1.0, "the Ranger, north"
    assert not lines[2].any(), "the petrified Skeleton, south-east"
    slot = raw.unit_ids.index(unit_id_at(cs, 5, 3))
    bits = np.unpackbits(np.asarray(core_packed_masks(raw).attack_valid[slot], dtype=np.uint8))
    offered = {(raw.hex_positions[t].x, raw.hex_positions[t].y)
               for t in np.flatnonzero(bits[:len(raw.hex_positions)])}
    assert offered == {(5, 2)}


# ---------------------------------------------------------------------------
# Played games
# ---------------------------------------------------------------------------

def _closing_play(seed: int, turns: int):
    """A ladder game played by a walker that closes on the enemy and
    attacks when it can: yields (sim, the encoder's view of the decision)
    before each action, and applies it."""
    from tools.wesnoth_sim import PvPDefaults, WesnothSim
    from wesnoth_ai.action_sampler import _build_legality_masks
    from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate, random_setup
    rng = random.Random(seed)
    setup = random_setup(rng, category="ladder")
    gs = build_scenario_gamestate(setup, base_income=PvPDefaults().base_income)
    sim = WesnothSim(gs, scenario_id=setup.scenario_id, max_turns=turns)
    e = enc.GameStateEncoder(d_model=8, relevant_set_hexes=True, fog_hides_enemy_villages=True,
                             terrain_multi_hot=True, observation_parity=True, relevant_set_version=2,
                             unit_row_version=2)
    for _ in range(4000):
        if sim.done:
            return
        e.register_names(sim.gs)
        encoded = e.encode(sim.gs)
        masks = _build_legality_masks(encoded, sim.gs)
        yield sim, e, encoded, masks
        side = sim.gs.global_info.current_side
        U, R = encoded.unit_tokens.size(1), encoded.recruit_tokens.size(1)
        acting = np.flatnonzero(masks.actor_valid[0].numpy())
        leaders = {u.id for u in sim.gs.map.units if u.is_leader}
        attacks = [(a, t) for a in acting if a < U for t in np.flatnonzero(masks.target_valid_attack[a].numpy())]
        # The leaders stay on their keeps and recruit, so the games last.
        moves = [(a, t) for a in acting if a < U and encoded.unit_ids[a] not in leaders
                 for t in np.flatnonzero(masks.target_valid_move[a].numpy())]
        recruits = [(a, t) for a in acting if U <= a < U + R for t in np.flatnonzero(masks.target_valid[a].numpy())]
        foes = [(u.position.x, u.position.y) for u in sim.gs.map.units if u.side not in (side,) and u.side in (1, 2)]
        r = rng.random()
        if recruits and r < 0.9:
            a, t = rng.choice(recruits)
            action = {"type": "recruit", "unit_type": encoded.recruit_types[a - U],
                      "target_hex": encoded.hex_positions[t]}
        elif attacks and r < 0.97:
            a, t = rng.choice(attacks)
            unit = next(u for u in sim.gs.map.units if u.id == encoded.unit_ids[a])
            action = {"type": "attack", "start_hex": encoded.unit_positions[a],
                      "target_hex": encoded.hex_positions[t], "attack_index": rng.randrange(len(unit.attacks))}
        elif moves and r < 0.97:
            def gap(move):                         # the hexes left to the nearest enemy after the move
                h = encoded.hex_positions[move[1]]
                return min((abs(h.x - fx) + abs(h.y - fy) for fx, fy in foes), default=0)
            best = min(gap(m) for m in moves)
            a, t = rng.choice([m for m in moves if gap(m) <= best + 1])
            action = {"type": "move", "start_hex": encoded.unit_positions[a], "target_hex": encoded.hex_positions[t]}
        else:
            action = {"type": "end_turn"}
        sim.step(action)


def _schedule(first: str, a_n: int, d_n: int) -> list:
    """The order of a one-round fight's strikes: the sides alternate from
    `first` while either has a strike left (attack.cpp `perform`)."""
    left = {"a": a_n, "d": d_n}
    order, who = [], first
    while left["a"] or left["d"]:
        if left[who]:
            order.append(who)
            left[who] -= 1
        who = "d" if who == "a" else "a"
    return order


def test_on_played_games_the_lines_are_the_fights_own():
    """Over two games played into contact: version 1's rows are version
    2's prefix and every other stream is unchanged; a unit on a hex of one
    terrain class is hit there as its table says for that class; every
    line equals the core's fight statistics for that attacker, enemy and
    weapon, the enemy answering as the engine chooses; no attack is offered
    on a hex that does not touch the attacker's; and each attack then
    played strikes as its line said: every strike's chance to hit, and
    every hit's damage until a slowing hit lands."""
    from tools.abilities import hex_neighbors
    seen = {"positions": 0, "lines": 0, "one_class": 0}
    for seed in (11, 12):
        for sim, e, encoded, masks in _closing_play(seed, turns=16):
            cs = sim.core
            core = cs.core
            vocab = dict(type_to_id=e.unit_type_to_id, faction_to_id=e.faction_to_id)
            r1 = cs.encode_raw(**vocab, **RECIPE)
            r2 = cs.encode_raw(**vocab, unit_row_version=2, **RECIPE)
            seen["positions"] += 1
            assert np.array_equal(r1.unit_feats, r2.unit_feats[:, :enc.UNIT_FEAT_DIM_PARITY])
            assert np.array_equal(r1.recruit_feats, r2.recruit_feats[:, :enc.UNIT_FEAT_DIM_PARITY])
            for name in ("hex_terrain_ids", "hex_dynamic_flags", "global_feats", "unit_type_ids", "unit_xs"):
                assert np.array_equal(getattr(r1, name), getattr(r2, name)), name
            side = sim.gs.global_info.current_side
            units = {(u.position.x, u.position.y): u for u in sim.gs.map.units}
            visible = set(r2.unit_ids)
            terrain = {(p.x, p.y): int(m) for p, m in zip(r2.hex_positions, r2.hex_terrain_ids)}
            for row, uid, pos in zip(r2.unit_feats, r2.unit_ids, r2.unit_positions):
                u = units[(pos.x, pos.y)]
                mask = terrain.get((pos.x, pos.y), 0)
                if mask and mask & (mask - 1) == 0:              # one terrain class
                    k = mask.bit_length() - 1
                    assert row[enc.ROWS2_STANDS_AT] == row[enc.ROWS2_DEFENCE_AT + k], (uid, k)
                    seen["one_class"] += 1
                lines = fights_of(row)
                can = u.side == side and not u.has_attacked and "petrified" not in (u.statuses or set())
                for d, (nx, ny) in enumerate(hex_neighbors(pos.x, pos.y)):
                    foe = units.get((nx, ny))
                    if not (can and foe is not None and foe.id in visible and foe.side != side
                            and "petrified" not in (foe.statuses or set())):
                        assert not lines[d].any(), (uid, d)
                        continue
                    for w in range(W):
                        if w >= len(u.attacks):
                            assert not lines[d, w].any()
                            continue
                        counter = int(core.counter_weapon_choice(pos.x, pos.y, nx, ny, w)[0])
                        a_st, d_st = core.fight_stats(pos.x, pos.y, nx, ny, w, counter)
                        want = np.zeros(enc.FIGHT_COLS, dtype=np.float32)
                        want[0] = 1.0
                        hit = a_st["slow_damage"] if "slowed" in (u.statuses or set()) else a_st["damage"]
                        want[1:4] = (hit / enc.WEAPON_DAMAGE_NORM, a_st["n_attacks"] / enc.WEAPON_STRIKES_NORM,
                                     a_st["cth"] / 100)
                        if d_st is not None:
                            hit = d_st["slow_damage"] if "slowed" in (foe.statuses or set()) else d_st["damage"]
                            want[4:8] = (1.0, hit / enc.WEAPON_DAMAGE_NORM,
                                         d_st["n_attacks"] / enc.WEAPON_STRIKES_NORM, d_st["cth"] / 100)
                            if counter < W:
                                want[8 + counter] = 1.0
                        assert np.array_equal(lines[d, w], want), (uid, d, w, lines[d, w], want)
                        seen["lines"] += 1
            # The legal attacks: every one on a hex that touches the attacker's.
            for a in range(encoded.unit_tokens.size(1)):
                for t in np.flatnonzero(masks.target_valid_attack[a].numpy()):
                    s, h = encoded.unit_positions[a], encoded.hex_positions[t]
                    assert (h.x, h.y) in hex_neighbors(s.x, s.y), "an attack offered from a distance"
    assert seen["positions"] > 200 and seen["lines"] > 100 and seen["one_class"] > 500, seen


def _played_attacks(seed: int, turns: int):
    """(the line the attacker's row held, the attack's recorded strikes,
    the two weapons' specials) for each attack of a closing game."""
    from tools.abilities import hex_neighbors
    types = unit_types()

    def specials(name: str, weapon: int) -> set:
        return set(types[name]["attacks"][weapon].get("specials") or ()) if weapon >= 0 else set()

    out = []
    pending = None
    for sim, e, _encoded, _masks in _closing_play(seed, turns):
        history = sim.command_history
        if pending is not None:
            n_before, lines, units = pending
            new = history[n_before:]
            if len(new) == 1 and new[0].kind == "attack" and new[0].extras.get("checkup_strikes"):
                ax, ay, dx, dy, w, dw = (int(v) for v in new[0].cmd[1:7])
                d = hex_neighbors(ax, ay).index((dx, dy))
                (a_id, a_name), (_d_id, d_name) = units[(ax, ay)], units[(dx, dy)]
                out.append((lines[a_id][d, w], new[0].extras["checkup_strikes"], specials(a_name, w),
                            specials(d_name, dw), dw))
        raw = sim.core.encode_raw(type_to_id=e.unit_type_to_id, faction_to_id=e.faction_to_id,
                                  unit_row_version=2, **RECIPE)
        pending = (len(history), {uid: fights_of(row).copy() for row, uid in zip(raw.unit_feats, raw.unit_ids)},
                   {(u.position.x, u.position.y): (u.id, u.name) for u in sim.gs.map.units})
    return out


def test_each_attack_played_strikes_as_its_line_said():
    """The fights of two closing games, strike by strike from the
    simulator's record of them (the engine's [mp_checkup] data: chance,
    hit, damage): every strike is at its side's chance to hit, the sides
    strike in the order their counts give, and a hit does its side's
    damage, until a slowing hit lands, which halves the damage of what
    follows. A berserk fight is read on its first round."""
    checked = {"attacks": 0, "strikes": 0, "answered": 0}
    for seed in (11, 12):
        for line, strikes, a_specials, d_specials, dw in _played_attacks(seed, turns=16):
            assert line[0] == 1.0
            a = dict(damage=round(float(line[1]) * enc.WEAPON_DAMAGE_NORM), n=round(float(line[2]) * 6),
                     cth=round(float(line[3]) * 100))
            answers = line[4] == 1.0
            assert answers == (dw >= 0)
            dfd = dict(damage=round(float(line[5]) * enc.WEAPON_DAMAGE_NORM), n=round(float(line[6]) * 6),
                       cth=round(float(line[7]) * 100))
            if answers and dw < W:
                assert line[8 + dw] == 1.0 and line[8:].sum() == 1.0
            first = "d" if answers and "firststrike" in d_specials and "firststrike" not in a_specials else "a"
            order = _schedule(first, a["n"], dfd["n"] if answers else 0)
            made = [(strikes[k], strikes[k + 1]) for k in range(0, len(strikes) - 1, 2)]
            assert len(made) <= len(order) or "berserk" in a_specials | d_specials
            slowed = False
            for who, (s, end) in zip(order, made):
                side = a if who == "a" else dfd
                assert s["chance"] == side["cth"], (who, s, line)
                if s["hits"] and not slowed:
                    assert s["damage"] == side["damage"], (who, s, line)
                if s["hits"] and "slow" in (a_specials if who == "a" else d_specials):
                    slowed = True
                checked["strikes"] += 1
                if end["dies"]:
                    break
            checked["attacks"] += 1
            checked["answered"] += bool(answers)
    assert checked["attacks"] > 25 and checked["strikes"] > 100 and checked["answered"] > 10, checked


# ---------------------------------------------------------------------------
# The legal attacks
# ---------------------------------------------------------------------------

def test_an_attack_is_legal_only_on_a_hex_that_touches_the_attackers():
    """A Spearman with its moves left, a Skeleton two hexes away that it
    could walk up to, and an Elvish Archer beside it: the mask offers the
    Archer and not the Skeleton, in the Python builder and in the core's
    packed masks alike."""
    from tools.abilities import hex_neighbors
    from wesnoth_ai.action_sampler import _build_legality_masks
    from wesnoth_ai.game_core import view_of
    from wesnoth_ai.server_priors import core_packed_masks
    units = [("Lieutenant", 1, 1, 3, True), ("Spearman", 1, 5, 3, False), ("Lieutenant", 2, 18, 3, True),
             ("Skeleton", 2, 7, 3, False), ("Elvish Archer", 2, 6, 4, False)]
    data = record(units, fog=False)
    cs = core_of(data)
    cs.apply_command(["init_side", 1])
    assert (7, 3) not in hex_neighbors(5, 3) and (6, 4) in hex_neighbors(5, 3)
    e = enc.GameStateEncoder(d_model=8, relevant_set_hexes=True, fog_hides_enemy_villages=True,
                             terrain_multi_hot=True, observation_parity=True, relevant_set_version=2)
    gs = view_of(cs)
    e.register_names(gs)
    encoded = e.encode(gs)
    masks = _build_legality_masks(encoded, gs)
    a = encoded.unit_ids.index(unit_id_at(cs, 5, 3))
    targets = {(encoded.hex_positions[t].x, encoded.hex_positions[t].y)
               for t in np.flatnonzero(masks.target_valid_attack[a].numpy())}
    assert targets == {(6, 4)}
    raw = cs.encode_raw(type_to_id=e.unit_type_to_id, faction_to_id=e.faction_to_id, masks=True, **RECIPE)
    packed = core_packed_masks(raw)
    slot = raw.unit_ids.index(unit_id_at(cs, 5, 3))
    bits = np.unpackbits(np.asarray(packed.attack_valid[slot], dtype=np.uint8))[:len(raw.hex_positions)]
    assert {(raw.hex_positions[t].x, raw.hex_positions[t].y) for t in np.flatnonzero(bits)} == {(6, 4)}


# ---------------------------------------------------------------------------
# A checkpoint widened to version 2
# ---------------------------------------------------------------------------

def _tiny_checkpoint(tmp_path):
    from helpers.pg_games import save_network, tiny_network
    vocab, encoder, model = tiny_network()
    return save_network(tmp_path / "pi.pt", vocab, encoder, model), vocab


def test_a_widened_checkpoint_computes_what_its_source_computed(tmp_path):
    """The widened network reads version 2's rows and gives the source's
    output on the same positions, to the last bit: the added columns go
    through a projection of their own, at zero, so each unit token gains a
    sum of zeros. The widening names its source, refuses a checkpoint
    already widened, and a policy of one version refuses a checkpoint of
    the other."""
    from helpers.pg_games import SLOTS, initial_forward
    from tools.eval_players import _load_policy, peek_checkpoint_arch
    from tools.eval_provenance import file_sha256
    from tools.pg_pass import load_network
    from tools.widen_unit_rows import WidenRefused, widen
    from wesnoth_ai.transformer_policy import TransformerPolicy
    source, _vocab = _tiny_checkpoint(tmp_path)
    wide = tmp_path / "pi_rows2.pt"
    rec = widen(source, wide)
    assert rec["base_sha256"] == file_sha256(source) and rec["checkpoint_sha256"] == file_sha256(wide)
    ck = torch.load(wide, map_location="cpu", weights_only=True)
    assert ck["unit_row_version"] == 2 and ck["unit_rows_widened"]["base_sha256"] == rec["base_sha256"]
    source_state = torch.load(source, map_location="cpu", weights_only=True)["encoder_state"]
    added = ck["encoder_state"]["unit_feat_proj2.weight"]
    assert added.shape == (source_state["unit_feat_proj.weight"].shape[0], enc.UNIT_FEAT_DIM_ROWS2 - 109)
    assert not added.any() and list(ck["encoder_state"])[-1] == "unit_feat_proj2.weight"
    for name, weight in source_state.items():
        assert torch.equal(ck["encoder_state"][name], weight), name
    with pytest.raises(WidenRefused, match="version 2"):
        widen(wide, tmp_path / "again.pt")
    assert peek_checkpoint_arch(wide)["unit_row_version"] == 2
    assert "unit_row_version" not in peek_checkpoint_arch(source) or peek_checkpoint_arch(source)["unit_row_version"] == 1

    cpu = torch.device("cpu")
    enc1, model1 = load_network(source, cpu)
    enc2, model2 = load_network(wide, cpu)
    assert (enc1.unit_row_version, enc2.unit_row_version) == (1, 2)
    raws1, raws2 = [], []
    for sim, e, _encoded, _masks in _closing_play(11, turns=7):
        if len(raws1) >= 24 or sim.gs.global_info.turn_number < 4:
            continue
        vocab = dict(type_to_id=enc1.unit_type_to_id, faction_to_id=enc1.faction_to_id)
        raws1.append(sim.core.encode_raw(**vocab, **RECIPE))
        raws2.append(sim.core.encode_raw(**vocab, unit_row_version=2, **RECIPE))
    assert len(raws1) == 24 and any(r.unit_feats[:, F:].any() for r in raws2), "positions with a fight in view"
    with torch.no_grad():
        out1 = initial_forward(model1, enc1, raws1, SLOTS)
        out2 = initial_forward(model2, enc2, raws2, SLOTS)
    for name in ("actor_logits", "type_logits", "target_logits", "weapon_logits", "value", "value_logits"):
        assert torch.equal(getattr(out1, name), getattr(out2, name)), name

    policy = _load_policy(wide, cpu, label="wide", infer_bf16=False, infer_compile=False)
    assert policy._inference_encoder.unit_row_version == 2
    assert policy._inference_encoder.unit_feat_proj.weight.shape[1] == enc.UNIT_FEAT_DIM_PARITY
    assert policy._inference_encoder.unit_feat_proj2.weight.shape[1] == enc.UNIT_FEAT_DIM_ROWS2 - 109
    arch = {k: v for k, v in peek_checkpoint_arch(source).items()}
    with pytest.raises(RuntimeError, match="unit_row_version"):
        TransformerPolicy(**{**arch, "unit_row_version": 2}).load_checkpoint(source)
    with pytest.raises(RuntimeError, match="unit_row_version"):
        TransformerPolicy(**{**peek_checkpoint_arch(wide), "unit_row_version": 1}).load_checkpoint(wide)


def test_the_served_network_and_the_passed_one_read_the_new_columns_alike(tmp_path):
    """With the added projection's weights far from zero, the pass must
    still reproduce the probabilities the players logged through the
    inference server: both forwards read the new columns, and alike. A
    forward that left them out would be off by the whole of their effect,
    which is shown to be large here."""
    import json
    from helpers.pg_games import SLOTS, initial_forward, selfplay_games
    import tools.pg_game_pass as gp
    from tools.game_record import read_records, walk_core
    from tools.pg_pass import load_network
    from tools.widen_unit_rows import NEW_WEIGHT, widen
    source, _vocab = _tiny_checkpoint(tmp_path)
    wide = tmp_path / "pi_rows2.pt"
    widen(source, wide)
    ck = torch.load(wide, map_location="cpu", weights_only=True)
    torch.manual_seed(7)
    ck["encoder_state"][NEW_WEIGHT] = 0.3 * torch.randn_like(ck["encoder_state"][NEW_WEIGHT])
    loud = tmp_path / "pi_loud.pt"
    torch.save(ck, loud)

    games = tmp_path / "games"
    records = selfplay_games(loud, games, [(1, 13), (1, 47)])
    grad = tmp_path / "grad.pt"
    assert gp.main(["--checkpoint", str(loud), "--games", str(games), "--out", str(grad), "--device", "cpu",
                    "--workers", "0"]) == 0
    replay = json.loads(grad.with_suffix(".json").read_text(encoding="utf-8"))["replay"]
    assert replay["half_mean_square"] < 1e-9, replay

    # The new columns matter to this network: without them its output moves.
    cpu = torch.device("cpu")
    enc_loud, model_loud = load_network(loud, cpu)
    enc_wide, model_wide = load_network(wide, cpu)
    vocab = dict(type_to_id=enc_loud.unit_type_to_id, faction_to_id=enc_loud.faction_to_id)
    raws = []
    for path in records.values():
        [rec] = list(read_records(path))
        for _k, cs, cmd in walk_core(rec):
            if int(cs.core.current_side) in (1, 2) and cmd[0] in ("move", "attack", "recruit", "end_turn"):
                raws.append(cs.encode_raw(unit_row_version=2, **vocab, **RECIPE))
    assert len(raws) > 20
    with torch.no_grad():
        out_loud = initial_forward(model_loud, enc_loud, raws, SLOTS)
        out_wide = initial_forward(model_wide, enc_wide, raws, SLOTS)
    finite = torch.isfinite(out_wide.actor_logits)
    assert (out_loud.actor_logits[finite] - out_wide.actor_logits[finite]).abs().max() > 0.05


def test_the_loops_tools_play_pass_and_step_a_widened_checkpoint(tmp_path):
    """A round of the simple loop on each side of the widening, on a tiny
    network: games played and passed and one Adam step at version 1; the
    stepped checkpoint and its optimizer state widened; then games played
    by the widened checkpoint, whose pass reproduces the players' logged
    probabilities (the exact check), and a step from the carried state.
    The optimizer meets the added projection as a parameter it has never
    stepped: its weights move by that step, by the learning rate at most
    (Adam's first step), while every other parameter is at its second."""
    from helpers.pg_games import selfplay_games
    import tools.pg_game_pass as gp
    from tools.eval_provenance import file_sha256
    from tools.pg_adam_step import adam_step
    from tools.pg_positions import read_record
    from tools.widen_unit_rows import PARAMETER, widen
    lr = 1e-5
    theta0, _vocab = _tiny_checkpoint(tmp_path)

    def round_of(theta, name, state=None):
        games = tmp_path / f"games_{name}"
        records = selfplay_games(theta, games, [(1, 13), (2, 7)])
        grad = tmp_path / f"grad_{name}.pt"
        assert gp.main(["--checkpoint", str(theta), "--games", str(games), "--out", str(grad), "--device", "cpu",
                        "--workers", "0"]) == 0
        out, state_out = tmp_path / f"pi_{name}.pt", tmp_path / f"adam_{name}.pt"
        step = adam_step(theta, grad, state, lr, out, state_out)
        return out, state_out, step, records

    theta1, adam1, _step1, _ = round_of(theta0, "a")
    wide, wide_state = tmp_path / "pi_wide.pt", tmp_path / "adam_wide.pt"
    rec = widen(theta1, wide, adam1, wide_state)
    state = torch.load(wide_state, map_location="cpu", weights_only=True)
    assert state["checkpoint_sha256"] == file_sha256(wide) == rec["checkpoint_sha256"]
    source_state = torch.load(adam1, map_location="cpu", weights_only=True)
    assert state["names"] == list(source_state["names"]) + [PARAMETER]
    new_index = len(source_state["names"])
    assert state["optimizer"]["param_groups"][0]["params"] == list(range(new_index + 1))
    assert new_index not in state["optimizer"]["state"], "no moments yet: Adam builds them at its first step"
    assert set(state["optimizer"]["state"]) == set(source_state["optimizer"]["state"])

    theta2, _adam2, step2, records = round_of(wide, "b", wide_state)
    assert step2["step"] == 2
    for path in records.values():
        assert int(read_record(path)["format"]) >= 6
    stepped = torch.load(theta2, map_location="cpu", weights_only=True)
    moved = stepped["encoder_state"]["unit_feat_proj2.weight"].abs()
    assert moved.max() > 0, "the added columns' weights train"
    assert moved.max() <= lr * (1 + 1e-6), float(moved.max())
    assert stepped["unit_row_version"] == 2
    after = torch.load(_adam2, map_location="cpu", weights_only=True)["optimizer"]["state"]
    assert int(after[new_index]["step"]) == 1 and int(after[0]["step"]) == 2
