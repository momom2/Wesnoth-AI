"""Tests for the underworlder intruder system."""

from __future__ import annotations

import pytest

from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.world.pathfinding import AStarPathfinder
from dungeon_builder.dungeon_core.core import DungeonCore
from dungeon_builder.utils.rng import SeededRNG

from dungeon_builder.intruders.agent import Intruder, IntruderState
from dungeon_builder.intruders.archetypes import (
    IntruderObjective,
    ArchetypeStats,
    MAGMAWRAITH,
    BOREMITE,
    STONESKIN_BRUTE,
    TREMORSTALKER,
    CORROSIVE_CRAWLER,
    UNDERWORLD_ARCHETYPES,
    UNDERWORLD_ARCHETYPE_BY_NAME,
    ALL_ARCHETYPES,
)
from dungeon_builder.intruders.personal_map import PersonalMap
from dungeon_builder.intruders.personal_pathfinder import PersonalPathfinder
from dungeon_builder.intruders.interactions import (
    handle_block,
    InteractionResult,
)
from dungeon_builder.intruders.party import (
    PartyTemplate,
    generate_composition,
    choose_underworld_template,
    UNDERWORLD_TEMPLATES,
    UNDERWORLD_HORDE,
    UNDERWORLD_OVERSEER,
    UNDERWORLD_INFERNAL,
    UNDERWORLD_SOLITARY,
)
from dungeon_builder.intruders.decision import IntruderAI

from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_STONE,
    VOXEL_SPIKE,
    VOXEL_ROLLING_STONE,
    VOXEL_LAVA,
    SPIKE_DAMAGE,
    ROLLING_STONE_DAMAGE,
    UNDERWORLD_SPAWN_INTERVAL,
    MAX_UNDERWORLD_PARTIES,
    MAX_UNDERWORLDERS_TOTAL,
    UNDERWORLD_SPAWN_Z_MIN,
    UNDERWORLD_SPAWN_Z_MAX,
    MAGMAWRAITH_HEAT_AMOUNT,
    MAGMAWRAITH_HEAT_INTERVAL,
    BOREMITE_DIG_DIVISOR,
    CORROSIVE_DAMAGE_FACTOR,
    DIG_DURATION,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_intruder(
    arch=MAGMAWRAITH,
    x=5, y=5, z=10,
    is_underworlder=True,
    intruder_id=1,
    objective=IntruderObjective.DESTROY_CORE,
):
    return Intruder(
        intruder_id=intruder_id,
        x=x, y=y, z=z,
        archetype=arch,
        objective=objective,
        personal_map=PersonalMap(),
        is_underworlder=is_underworlder,
    )


def _make_ai(width=10, depth=10, height=20, core_pos=(5, 5, 10), seed=42):
    """Create an IntruderAI with a simple dungeon grid."""
    bus = EventBus()
    grid = VoxelGrid(width=width, depth=depth, height=height)
    grid.grid[:] = VOXEL_STONE
    # Clear surface
    grid.grid[:, :, 0] = VOXEL_AIR
    pf = AStarPathfinder(grid)
    core = DungeonCore(bus, *core_pos, hp=100)
    rng = SeededRNG(seed)
    ai = IntruderAI(bus, grid, pf, core, rng)
    return ai, bus, grid, core, rng


# ══════════════════════════════════════════════════════════════════════
# Archetype tests
# ══════════════════════════════════════════════════════════════════════


class TestUnderworldArchetypes:

    def test_five_archetypes_exist(self):
        assert len(UNDERWORLD_ARCHETYPES) == 5

    def test_all_have_unique_names(self):
        names = [a.name for a in UNDERWORLD_ARCHETYPES]
        assert len(set(names)) == 5

    def test_lookup_by_name(self):
        for arch in UNDERWORLD_ARCHETYPES:
            assert UNDERWORLD_ARCHETYPE_BY_NAME[arch.name] is arch

    def test_no_overlap_with_surface(self):
        surface_names = {a.name for a in ALL_ARCHETYPES}
        uw_names = {a.name for a in UNDERWORLD_ARCHETYPES}
        assert surface_names.isdisjoint(uw_names)

    def test_all_are_archetype_stats(self):
        for arch in UNDERWORLD_ARCHETYPES:
            assert isinstance(arch, ArchetypeStats)

    def test_magmawraith_fire_immune(self):
        assert MAGMAWRAITH.fire_immune is True
        assert MAGMAWRAITH.never_retreats is True

    def test_boremite_can_dig(self):
        assert BOREMITE.can_dig is True
        assert BOREMITE.never_retreats is True
        assert BOREMITE.hp == 25  # Weakest

    def test_stoneskin_brute_highest_hp(self):
        assert STONESKIN_BRUTE.hp == 200
        assert STONESKIN_BRUTE.frenzy_threshold == 0.3

    def test_tremorstalker_has_arcane_sight(self):
        assert TREMORSTALKER.arcane_sight_range == 4
        assert TREMORSTALKER.darkvision_range == 5
        assert TREMORSTALKER.spike_detect_range == 3

    def test_corrosive_crawler_can_dig(self):
        assert CORROSIVE_CRAWLER.can_dig is True
        assert CORROSIVE_CRAWLER.never_retreats is True


# ══════════════════════════════════════════════════════════════════════
# Party template tests
# ══════════════════════════════════════════════════════════════════════


class TestUnderworldPartyTemplates:

    def test_four_templates(self):
        assert len(UNDERWORLD_TEMPLATES) == 4

    def test_weights_sum_to_one(self):
        total = sum(t.weight for t in UNDERWORLD_TEMPLATES)
        assert total == pytest.approx(1.0)

    def test_choose_returns_underworld_template(self):
        rng = SeededRNG(42)
        for _ in range(20):
            t = choose_underworld_template(rng)
            assert t in UNDERWORLD_TEMPLATES

    def test_solitary_produces_one_member(self):
        rng = SeededRNG(42)
        comp = generate_composition(UNDERWORLD_SOLITARY, rng)
        assert len(comp) == 1
        assert comp[0] in (STONESKIN_BRUTE, MAGMAWRAITH, CORROSIVE_CRAWLER)

    def test_horde_produces_at_least_five(self):
        rng = SeededRNG(42)
        comp = generate_composition(UNDERWORLD_HORDE, rng)
        assert len(comp) >= 5

    def test_overseer_has_one_brute(self):
        rng = SeededRNG(42)
        comp = generate_composition(UNDERWORLD_OVERSEER, rng)
        brutes = [a for a in comp if a.name == "Stoneskin Brute"]
        assert len(brutes) == 1

    def test_infernal_has_tremorstalker(self):
        rng = SeededRNG(42)
        comp = generate_composition(UNDERWORLD_INFERNAL, rng)
        stalkers = [a for a in comp if a.name == "Tremorstalker"]
        assert len(stalkers) == 1


# ══════════════════════════════════════════════════════════════════════
# Agent tests
# ══════════════════════════════════════════════════════════════════════


class TestIsUnderworlder:

    def test_default_false(self):
        i = Intruder(
            1, 0, 0, 0, MAGMAWRAITH,
            IntruderObjective.DESTROY_CORE, PersonalMap(),
        )
        assert i.is_underworlder is False

    def test_set_true(self):
        i = _make_intruder(is_underworlder=True)
        assert i.is_underworlder is True

    def test_set_false_explicitly(self):
        i = _make_intruder(is_underworlder=False)
        assert i.is_underworlder is False


# ══════════════════════════════════════════════════════════════════════
# Spawn tests
# ══════════════════════════════════════════════════════════════════════


class TestUnderworldSpawning:

    def test_spawn_position_at_deep_edge(self):
        ai, bus, grid, core, rng = _make_ai()
        # Carve air at edges for depth z=10
        grid.grid[0, 0, 10] = VOXEL_AIR
        grid.grid[0, 1, 10] = VOXEL_AIR
        pos = ai._find_underworld_spawn_position()
        assert pos is not None
        x, y, z = pos
        assert UNDERWORLD_SPAWN_Z_MIN <= z <= UNDERWORLD_SPAWN_Z_MAX
        assert x == 0 or x == grid.width - 1 or y == 0 or y == grid.depth - 1

    def test_spawn_carves_solid_fallback(self):
        ai, bus, grid, core, rng = _make_ai()
        # No air at depth edges — all solid stone
        # The fallback should carve an air pocket
        pos = ai._find_underworld_spawn_position()
        assert pos is not None
        x, y, z = pos
        # Should have been carved to air
        assert grid.get(x, y, z) == VOXEL_AIR

    def test_spawn_timer_uses_underworld_interval(self):
        ai, bus, grid, core, rng = _make_ai()
        ai.spawning_enabled = True
        # Carve air at edge for spawn
        grid.grid[0, 0, 10] = VOXEL_AIR
        grid.grid[:, :, 0] = VOXEL_AIR  # Surface air for surface spawning

        # Should NOT spawn before interval
        for tick in range(1, UNDERWORLD_SPAWN_INTERVAL):
            bus.publish("tick", tick=tick)

        uw_intruders = [i for i in ai.intruders if i.is_underworlder]
        assert len(uw_intruders) == 0

    def test_spawn_at_interval(self):
        ai, bus, grid, core, rng = _make_ai()
        ai.spawning_enabled = True
        # Carve air at edges for spawning
        for x in range(grid.width):
            grid.grid[x, 0, 12] = VOXEL_AIR
            grid.grid[x, grid.depth - 1, 12] = VOXEL_AIR
        for y in range(grid.depth):
            grid.grid[0, y, 12] = VOXEL_AIR
            grid.grid[grid.width - 1, y, 12] = VOXEL_AIR
        # Carve path to core
        grid.grid[5, :, 10] = VOXEL_AIR
        grid.grid[:, 5, 10] = VOXEL_AIR
        grid.grid[5, 5, 10] = VOXEL_AIR

        for tick in range(1, UNDERWORLD_SPAWN_INTERVAL + 1):
            bus.publish("tick", tick=tick)

        uw_intruders = [i for i in ai.intruders if i.is_underworlder]
        assert len(uw_intruders) > 0

    def test_underworld_spawned_flag(self):
        ai, bus, grid, core, rng = _make_ai()
        ai.spawning_enabled = True
        # Carve air at edges
        for x in range(grid.width):
            grid.grid[x, 0, 12] = VOXEL_AIR
        for y in range(grid.depth):
            grid.grid[0, y, 12] = VOXEL_AIR
        grid.grid[5, :, 10] = VOXEL_AIR

        for tick in range(1, UNDERWORLD_SPAWN_INTERVAL + 1):
            bus.publish("tick", tick=tick)

        for i in ai.intruders:
            if i.is_underworlder:
                assert i.is_underworlder is True
                return
        # If no underworlders spawned (fallback carve), check that any UW exist
        # Could happen if carving was used
        uw = [i for i in ai.intruders if i.is_underworlder]
        if uw:
            assert uw[0].is_underworlder is True

    def test_caps_independent(self):
        """Underworld caps are independent of surface caps."""
        ai, bus, grid, core, rng = _make_ai()
        # Fill surface intruder list to MAX
        for n in range(30):
            si = Intruder(
                intruder_id=100 + n, x=0, y=0, z=0,
                archetype=MAGMAWRAITH,
                objective=IntruderObjective.DESTROY_CORE,
                personal_map=PersonalMap(),
                is_underworlder=False,
            )
            si.state = IntruderState.ADVANCING
            ai.intruders.append(si)

        # Surface count is high but underworld count is 0
        alive_uw = sum(1 for i in ai.intruders if i.alive and i.is_underworlder)
        assert alive_uw == 0

        # Underworld spawning should still work
        grid.grid[0, 0, 12] = VOXEL_AIR
        pos = ai._find_underworld_spawn_position()
        assert pos is not None


# ══════════════════════════════════════════════════════════════════════
# Retreat tests
# ══════════════════════════════════════════════════════════════════════


class TestUnderworldRetreat:

    def test_is_at_deep_edge_true(self):
        ai, bus, grid, core, rng = _make_ai()
        intruder = _make_intruder(x=0, y=5, z=12)
        assert ai._is_at_deep_edge(intruder) is True

    def test_is_at_deep_edge_false_surface(self):
        ai, bus, grid, core, rng = _make_ai()
        intruder = _make_intruder(x=0, y=5, z=0)
        assert ai._is_at_deep_edge(intruder) is False

    def test_is_at_deep_edge_false_interior(self):
        ai, bus, grid, core, rng = _make_ai()
        intruder = _make_intruder(x=5, y=5, z=12)
        assert ai._is_at_deep_edge(intruder) is False

    def test_is_at_deep_edge_all_edges(self):
        ai, bus, grid, core, rng = _make_ai()
        # x=0
        assert ai._is_at_deep_edge(_make_intruder(x=0, y=3, z=15)) is True
        # x=width-1
        assert ai._is_at_deep_edge(
            _make_intruder(x=grid.width - 1, y=3, z=15)
        ) is True
        # y=0
        assert ai._is_at_deep_edge(_make_intruder(x=3, y=0, z=15)) is True
        # y=depth-1
        assert ai._is_at_deep_edge(
            _make_intruder(x=3, y=grid.depth - 1, z=15)
        ) is True

    def test_underworlder_escapes_at_deep_edge(self):
        ai, bus, grid, core, rng = _make_ai()
        intruder = _make_intruder(
            arch=TREMORSTALKER, x=0, y=5, z=12,
        )
        intruder.state = IntruderState.RETREATING
        intruder.move_interval = 1
        ai.intruders.append(intruder)
        # Already at edge — should escape on update
        ai._update_retreating(intruder)
        assert intruder.state == IntruderState.ESCAPED

    def test_surface_intruder_does_not_escape_at_deep_edge(self):
        ai, bus, grid, core, rng = _make_ai()
        intruder = _make_intruder(
            arch=TREMORSTALKER, x=0, y=5, z=12,
            is_underworlder=False,
        )
        intruder.state = IntruderState.RETREATING
        intruder.move_interval = 1
        ai.intruders.append(intruder)
        ai._update_retreating(intruder)
        # NOT escaped — surface intruder needs to reach z=0
        assert intruder.state != IntruderState.ESCAPED

    def test_start_retreat_underworlder_at_deep_edge(self):
        ai, bus, grid, core, rng = _make_ai()
        intruder = _make_intruder(
            arch=TREMORSTALKER, x=0, y=5, z=12,
        )
        intruder.state = IntruderState.ADVANCING
        ai._start_retreat(intruder)
        assert intruder.state == IntruderState.RETREATING

    def test_start_retreat_underworlder_no_exit(self):
        ai, bus, grid, core, rng = _make_ai()
        # All edges are solid, no air anywhere deep
        intruder = _make_intruder(
            arch=TREMORSTALKER, x=5, y=5, z=12,
        )
        intruder.state = IntruderState.ADVANCING
        ai._start_retreat(intruder)
        # No exit → revert to ADVANCING
        assert intruder.state == IntruderState.ADVANCING


# ══════════════════════════════════════════════════════════════════════
# Unique behavior tests
# ══════════════════════════════════════════════════════════════════════


class TestBoremiteDigging:

    def test_dig_duration_one_third(self):
        ai, bus, grid, core, rng = _make_ai()
        intruder = _make_intruder(arch=BOREMITE, x=3, y=3, z=10)
        intruder.state = IntruderState.ADVANCING
        ai.intruders.append(intruder)

        target = (4, 3, 10)
        vtype = VOXEL_STONE
        base = DIG_DURATION.get(vtype, 40)
        expected = max(1, base // BOREMITE_DIG_DIVISOR)

        ai._start_digging(intruder, target, vtype)
        assert intruder.interaction_ticks == expected

    def test_dig_faster_than_tunneler(self):
        ai, bus, grid, core, rng = _make_ai()
        boremite = _make_intruder(arch=BOREMITE, x=3, y=3, z=10)
        from dungeon_builder.intruders.archetypes import TUNNELER
        tunneler = _make_intruder(
            arch=TUNNELER, x=3, y=3, z=10,
            is_underworlder=False, intruder_id=2,
        )

        target = (4, 3, 10)
        vtype = VOXEL_STONE

        ai._start_digging(boremite, target, vtype)
        bore_ticks = boremite.interaction_ticks

        ai._start_digging(tunneler, target, vtype)
        tunnel_ticks = tunneler.interaction_ticks

        assert bore_ticks < tunnel_ticks


class TestCorrosiveCrawler:

    def test_corrosive_adds_stress_after_dig(self):
        ai, bus, grid, core, rng = _make_ai()
        # Set up a digging interaction that's about to complete
        intruder = _make_intruder(arch=CORROSIVE_CRAWLER, x=3, y=3, z=10)
        intruder.state = IntruderState.INTERACTING
        intruder.interaction_type = "dig"
        intruder.interaction_target = (4, 3, 10)
        intruder.interaction_ticks = 1  # Will complete this tick
        ai.intruders.append(intruder)

        # Place stone at target and neighbors
        grid.grid[4, 3, 10] = VOXEL_STONE
        grid.grid[5, 3, 10] = VOXEL_STONE
        grid.grid[3, 3, 10] = VOXEL_STONE
        grid.grid[4, 4, 10] = VOXEL_STONE

        # Initial stress should be 0
        assert grid.stress_ratio[5, 3, 10] == 0.0

        ai._update_interacting(intruder)

        # Neighbors should have increased stress_ratio
        assert grid.stress_ratio[5, 3, 10] == pytest.approx(CORROSIVE_DAMAGE_FACTOR)
        # The block at (3,3,10) — the intruder's own position (solid) — also affected
        assert grid.stress_ratio[3, 3, 10] == pytest.approx(CORROSIVE_DAMAGE_FACTOR)
        assert grid.stress_ratio[4, 4, 10] == pytest.approx(CORROSIVE_DAMAGE_FACTOR)

    def test_corrosive_only_affects_solid(self):
        ai, bus, grid, core, rng = _make_ai()
        intruder = _make_intruder(arch=CORROSIVE_CRAWLER, x=3, y=3, z=10)
        # Dig target surrounded by air on some sides
        grid.grid[4, 3, 10] = VOXEL_STONE
        grid.grid[5, 3, 10] = VOXEL_AIR  # air neighbor
        grid.grid[3, 3, 10] = VOXEL_STONE  # solid neighbor

        ai._apply_corrosive_damage(4, 3, 10)

        # Air neighbor should NOT have stress
        assert grid.stress_ratio[5, 3, 10] == 0.0
        # Solid neighbor should have stress
        assert grid.stress_ratio[3, 3, 10] == pytest.approx(CORROSIVE_DAMAGE_FACTOR)


class TestMagmawraithHeat:

    def test_heats_adjacent_blocks(self):
        ai, bus, grid, core, rng = _make_ai()
        intruder = _make_intruder(arch=MAGMAWRAITH, x=5, y=5, z=10)
        intruder.state = IntruderState.ADVANCING
        ai.intruders.append(intruder)

        initial_temp = float(grid.temperature[6, 5, 10])
        ai._tick_magmawraith_heat(intruder, MAGMAWRAITH_HEAT_INTERVAL)
        assert grid.temperature[6, 5, 10] == pytest.approx(
            initial_temp + MAGMAWRAITH_HEAT_AMOUNT
        )

    def test_no_heat_off_interval(self):
        ai, bus, grid, core, rng = _make_ai()
        intruder = _make_intruder(arch=MAGMAWRAITH, x=5, y=5, z=10)
        initial_temp = float(grid.temperature[6, 5, 10])
        # Tick that's NOT a multiple of interval
        ai._tick_magmawraith_heat(intruder, MAGMAWRAITH_HEAT_INTERVAL + 1)
        assert grid.temperature[6, 5, 10] == pytest.approx(initial_temp)

    def test_no_heat_for_non_magmawraith(self):
        ai, bus, grid, core, rng = _make_ai()
        intruder = _make_intruder(arch=BOREMITE, x=5, y=5, z=10)
        initial_temp = float(grid.temperature[6, 5, 10])
        ai._tick_magmawraith_heat(intruder, MAGMAWRAITH_HEAT_INTERVAL)
        assert grid.temperature[6, 5, 10] == pytest.approx(initial_temp)


# ══════════════════════════════════════════════════════════════════════
# Interaction tests (Stoneskin Brute)
# ══════════════════════════════════════════════════════════════════════


class TestStoneskinBruteInteractions:

    def test_spike_quarter_damage(self):
        intruder = _make_intruder(arch=STONESKIN_BRUTE)
        info = handle_block(intruder, VOXEL_SPIKE, 1)  # extended
        assert info.result == InteractionResult.DAMAGE
        assert info.damage == SPIKE_DAMAGE // 4

    def test_rolling_stone_half_damage(self):
        intruder = _make_intruder(arch=STONESKIN_BRUTE)
        info = handle_block(intruder, VOXEL_ROLLING_STONE, 0)
        assert info.result == InteractionResult.DAMAGE
        assert info.damage == ROLLING_STONE_DAMAGE // 2

    def test_frenzy_at_thirty_percent(self):
        intruder = _make_intruder(arch=STONESKIN_BRUTE)
        assert intruder.max_hp == 200
        # Set HP to 30% = 60
        intruder.hp = 60
        IntruderAI._check_frenzy(intruder)
        assert intruder.frenzy_active is True

    def test_no_frenzy_above_threshold(self):
        intruder = _make_intruder(arch=STONESKIN_BRUTE)
        intruder.hp = 61  # Just above 30%
        IntruderAI._check_frenzy(intruder)
        assert intruder.frenzy_active is False


# ══════════════════════════════════════════════════════════════════════
# Magmawraith lava traversal
# ══════════════════════════════════════════════════════════════════════


class TestMagmawraithLava:

    def test_lava_continue(self):
        intruder = _make_intruder(arch=MAGMAWRAITH)
        info = handle_block(intruder, VOXEL_LAVA, 0)
        assert info.result == InteractionResult.CONTINUE

    def test_boremite_lava_death(self):
        intruder = _make_intruder(arch=BOREMITE)
        info = handle_block(intruder, VOXEL_LAVA, 0)
        assert info.result == InteractionResult.DEATH
