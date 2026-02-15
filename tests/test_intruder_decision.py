"""Tests for the full intruder decision engine (Phase 6 integration)."""

import pytest

from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.world.pathfinding import AStarPathfinder
from dungeon_builder.dungeon_core.core import DungeonCore
from dungeon_builder.intruders.agent import Intruder, IntruderState
from dungeon_builder.intruders.archetypes import (
    IntruderObjective,
    VANGUARD,
    SHADOWBLADE,
    TUNNELER,
    PYREMANCER,
    WINDCALLER,
    WARDEN,
    GORECLAW,
    GLOOMSEER,
)
from dungeon_builder.intruders.personal_map import PersonalMap
from dungeon_builder.intruders.decision import IntruderAI
from dungeon_builder.intruders.party import Party
from dungeon_builder.utils.rng import SeededRNG
from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_STONE,
    VOXEL_DOOR,
    VOXEL_SPIKE,
    VOXEL_TREASURE,
    VOXEL_TARP,
    VOXEL_LAVA,
    VOXEL_DIRT,
    VOXEL_REINFORCED_WALL,
    SURFACE_Z,
    DOOR_BASH_TICKS,
    DOOR_LOCKPICK_TICKS,
    TREASURE_GRAB_TICKS,
    DIG_DURATION,
    SPIKE_DAMAGE,
    PYREMANCER_HEAT_AMOUNT,
    PYREMANCER_HEAT_INTERVAL,
    INTRUDER_PARTY_SPAWN_INTERVAL,
    MAX_PARTIES,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _make_grid(width=10, depth=10, height=5):
    """Create a grid of stone with a full air surface."""
    grid = VoxelGrid(width=width, depth=depth, height=height)
    grid.grid[:] = VOXEL_STONE
    grid.grid[:, :, 0] = VOXEL_AIR
    return grid


def _make_corridor_grid():
    """Grid with surface, shaft, and corridor to core.

    Layout:
      z=0: full air surface
      (5,0-5) z=0-3: vertical shaft
      (5,0-5) z=3: horizontal corridor
      Core at (5,5,3)
    """
    grid = _make_grid()
    for z in range(0, 4):
        grid.grid[5, 0, z] = VOXEL_AIR
    for y in range(0, 6):
        grid.grid[5, y, 3] = VOXEL_AIR
    return grid


def _make_ai(grid=None, core_pos=(5, 5, 3), core_hp=100, seed=42):
    """Create an IntruderAI with the given grid setup."""
    bus = EventBus()
    if grid is None:
        grid = _make_corridor_grid()
    pf = AStarPathfinder(grid)
    core = DungeonCore(bus, *core_pos, hp=core_hp)
    rng = SeededRNG(seed)
    ai = IntruderAI(bus, grid, pf, core, rng)
    return ai, bus, grid, core, rng


def _make_intruder(
    intruder_id=1, x=0, y=0, z=0, arch=VANGUARD,
    objective=IntruderObjective.DESTROY_CORE,
):
    return Intruder(intruder_id, x, y, z, arch, objective, PersonalMap())


# ── Party spawning ─────────────────────────────────────────────────────


class TestPartySpawning:
    def test_spawn_party_creates_members(self):
        ai, bus, grid, core, rng = _make_ai()
        ai._spawn_party()
        assert len(ai.intruders) > 0
        assert len(ai.parties) == 1
        assert all(i.state == IntruderState.ADVANCING for i in ai.intruders)

    def test_spawn_party_publishes_events(self):
        ai, bus, grid, core, rng = _make_ai()
        spawned = []
        bus.subscribe("intruder_spawned", lambda **kw: spawned.append(kw))
        ai._spawn_party()
        assert len(spawned) == len(ai.intruders)

    def test_spawn_party_assigns_party_id(self):
        ai, bus, grid, core, rng = _make_ai()
        ai._spawn_party()
        party = ai.parties[0]
        for m in party.members:
            assert m.party_id == party.id

    def test_tick_spawning_respects_max_parties(self):
        """The _tick_spawning method enforces MAX_PARTIES limit."""
        ai, bus, grid, core, rng = _make_ai()
        ai.spawning_enabled = True
        # Manually spawn MAX_PARTIES parties
        for _ in range(MAX_PARTIES):
            ai._spawn_party()
        assert len(ai.parties) == MAX_PARTIES

        # Now _tick_spawning should refuse to spawn more
        ai._spawn_timer = INTRUDER_PARTY_SPAWN_INTERVAL  # Ready to spawn
        ai._tick_spawning()
        assert len(ai.parties) == MAX_PARTIES  # No new party

    def test_spawning_disabled_by_default(self):
        ai, bus, grid, core, rng = _make_ai()
        bus.publish("tick", tick=1)
        assert len(ai.intruders) == 0


# ── Vision integration ─────────────────────────────────────────────────


class TestVisionIntegration:
    def test_intruder_reveals_nearby_cells(self):
        ai, bus, grid, core, rng = _make_ai()
        intruder = _make_intruder(1, 5, 0, 0)
        intruder.state = IntruderState.ADVANCING
        ai.intruders.append(intruder)

        ai._update_vision(intruder)

        assert intruder.personal_map.is_revealed(5, 0, 0)
        assert len(intruder.personal_map) > 1

    def test_gloomseer_arcane_sight(self):
        ai, bus, grid, core, rng = _make_ai()
        intruder = _make_intruder(1, 5, 0, 0, arch=GLOOMSEER)
        intruder.state = IntruderState.ADVANCING
        ai.intruders.append(intruder)

        ai._update_vision(intruder)

        # Gloomseer sees through walls via arcane sight (range=6)
        assert intruder.personal_map.is_revealed(5, 0, 3)

    def test_pyremancer_thermal_vision(self):
        ai, bus, grid, core, rng = _make_ai()
        grid.grid[3, 0, 0] = VOXEL_LAVA
        grid.temperature[3, 0, 0] = 1000.0

        intruder = _make_intruder(1, 5, 0, 0, arch=PYREMANCER)
        intruder.state = IntruderState.ADVANCING
        ai.intruders.append(intruder)

        ai._update_vision(intruder)

        # Within thermal range (dist 2 <= 4) and above threshold
        assert intruder.personal_map.is_revealed(3, 0, 0)


# ── Movement and path following ────────────────────────────────────────


class TestMovement:
    def test_intruder_follows_path_to_core(self):
        ai, bus, grid, core, rng = _make_ai()
        intruder = _make_intruder(1, 5, 0, 0)
        path = ai.pathfinder.find_path((5, 0, 0), (5, 5, 3))
        assert path is not None
        intruder.path = path
        intruder.path_index = 1
        intruder.state = IntruderState.ADVANCING
        intruder.move_interval = 1
        ai.intruders.append(intruder)

        for _ in range(100):
            ai._update_intruder(intruder)
            if intruder.state == IntruderState.ATTACKING:
                break

        assert intruder.state == IntruderState.ATTACKING

    def test_intruder_moves_publishes_event(self):
        ai, bus, grid, core, rng = _make_ai()
        intruder = _make_intruder(1, 5, 0, 0)
        intruder.path = [(5, 0, 0), (5, 1, 0)]
        intruder.path_index = 1
        intruder.state = IntruderState.ADVANCING
        intruder.move_interval = 1  # Now uses instance variable
        ai.intruders.append(intruder)

        moves = []
        bus.subscribe("intruder_moved", lambda **kw: moves.append(kw))

        ai._update_advancing(intruder)

        assert len(moves) == 1
        assert intruder.pos == (5, 1, 0)


# ── Retreat behavior ───────────────────────────────────────────────────


class TestRetreat:
    def test_vanguard_retreats_below_threshold(self):
        ai, bus, grid, core, rng = _make_ai()
        intruder = _make_intruder(1, 5, 3, 3)
        intruder.state = IntruderState.ADVANCING
        intruder.path = ai.pathfinder.find_path((5, 3, 3), (5, 5, 3))
        intruder.path_index = 1
        intruder.move_interval = 1
        ai.intruders.append(intruder)

        # Vanguard retreat_threshold=0.15, max_hp=120 -> retreat at <18 HP
        intruder.hp = 10
        ai._update_intruder(intruder)
        assert intruder.state == IntruderState.RETREATING

    def test_goreclaw_never_retreats(self):
        ai, bus, grid, core, rng = _make_ai()
        intruder = _make_intruder(1, 5, 3, 3, arch=GORECLAW)
        intruder.state = IntruderState.ADVANCING
        intruder.path = [(5, 3, 3), (5, 4, 3)]
        intruder.path_index = 1
        intruder.move_interval = 1
        ai.intruders.append(intruder)

        intruder.hp = 1
        ai._update_intruder(intruder)
        assert intruder.state != IntruderState.RETREATING

    def test_retreating_intruder_escapes_at_surface(self):
        ai, bus, grid, core, rng = _make_ai()
        # Intruder already at surface, path points just past end
        intruder = _make_intruder(1, 5, 0, 0)
        intruder.state = IntruderState.RETREATING
        intruder.move_interval = 1
        intruder.path = [(5, 0, 0)]
        intruder.path_index = 1  # Past end of path
        ai.intruders.append(intruder)

        escaped = []
        bus.subscribe("intruder_escaped", lambda **kw: escaped.append(kw))

        # First call: ticks_since_move goes 0->1, since 1 < 1 is false,
        # _advance_along_path does nothing (path_index past end), then z==0 check
        ai._update_retreating(intruder)

        assert intruder.state == IntruderState.ESCAPED
        assert len(escaped) == 1


# ── Frenzy behavior ───────────────────────────────────────────────────


class TestFrenzy:
    def test_goreclaw_enters_frenzy_at_half_hp(self):
        ai, bus, grid, core, rng = _make_ai()
        intruder = _make_intruder(1, 5, 0, 0, arch=GORECLAW)
        intruder.state = IntruderState.ADVANCING
        ai.intruders.append(intruder)

        intruder.hp = 45  # 45/90 = 0.5, threshold is 0.5
        IntruderAI._check_frenzy(intruder)
        assert intruder.frenzy_active is True

    def test_frenzy_doubles_speed(self):
        intruder = _make_intruder(1, 5, 0, 0, arch=GORECLAW)
        intruder.frenzy_active = True
        assert intruder.effective_speed == GORECLAW.speed * 2
        assert intruder.effective_move_interval == max(1, GORECLAW.move_interval // 2)

    def test_frenzy_boosts_damage(self):
        intruder = _make_intruder(1, 5, 0, 0, arch=GORECLAW)
        intruder.frenzy_active = True
        assert intruder.effective_damage == int(GORECLAW.damage * 1.5)


# ── Door interaction ───────────────────────────────────────────────────


class TestDoorInteraction:
    def _setup_door(self):
        """Grid with air corridor at z=3, closed door at (5,2,3)."""
        grid = _make_grid()
        for z in range(0, 4):
            grid.grid[5, 0, z] = VOXEL_AIR
        for y in range(0, 6):
            grid.grid[5, y, 3] = VOXEL_AIR
        # Place closed door at (5, 2, 3)
        grid.grid[5, 2, 3] = VOXEL_DOOR
        grid.block_state[5, 2, 3] = 1  # Closed
        return grid

    def test_vanguard_bashes_closed_door(self):
        grid = self._setup_door()
        ai, bus, _, core, rng = _make_ai(grid=grid)

        intruder = _make_intruder(1, 5, 1, 3, arch=VANGUARD)
        intruder.state = IntruderState.ADVANCING
        intruder.path = [(5, 1, 3), (5, 2, 3), (5, 3, 3)]
        intruder.path_index = 1
        intruder.move_interval = 1
        ai.intruders.append(intruder)

        ai._update_advancing(intruder)

        assert intruder.state == IntruderState.INTERACTING
        assert intruder.interaction_type == "bash_door"
        assert intruder.interaction_ticks == DOOR_BASH_TICKS

    def test_shadowblade_lockpicks_closed_door(self):
        grid = self._setup_door()
        ai, bus, _, core, rng = _make_ai(grid=grid)

        intruder = _make_intruder(1, 5, 1, 3, arch=SHADOWBLADE)
        intruder.state = IntruderState.ADVANCING
        intruder.path = [(5, 1, 3), (5, 2, 3), (5, 3, 3)]
        intruder.path_index = 1
        intruder.move_interval = 1
        ai.intruders.append(intruder)

        ai._update_advancing(intruder)

        assert intruder.state == IntruderState.INTERACTING
        assert intruder.interaction_type == "lockpick"
        assert intruder.interaction_ticks == DOOR_LOCKPICK_TICKS

    def test_door_opens_after_interaction(self):
        grid = self._setup_door()
        ai, bus, _, core, rng = _make_ai(grid=grid)

        intruder = _make_intruder(1, 5, 1, 3, arch=VANGUARD)
        intruder.state = IntruderState.ADVANCING
        intruder.path = [(5, 1, 3), (5, 2, 3), (5, 3, 3)]
        intruder.path_index = 1
        intruder.move_interval = 1
        ai.intruders.append(intruder)

        ai._update_advancing(intruder)
        assert intruder.state == IntruderState.INTERACTING

        for _ in range(DOOR_BASH_TICKS):
            ai._update_interacting(intruder)

        assert grid.block_state[5, 2, 3] == 0
        assert intruder.state == IntruderState.ADVANCING


# ── Spike interaction ──────────────────────────────────────────────────


class TestSpikeInteraction:
    def test_vanguard_takes_half_spike_damage(self):
        grid = _make_grid()
        for y in range(0, 5):
            grid.grid[5, y, 0] = VOXEL_AIR
        grid.grid[5, 2, 0] = VOXEL_SPIKE
        grid.block_state[5, 2, 0] = 1  # Extended

        ai, bus, _, core, rng = _make_ai(grid=grid)

        intruder = _make_intruder(1, 5, 1, 0, arch=VANGUARD)
        intruder.state = IntruderState.ADVANCING
        intruder.path = [(5, 1, 0), (5, 2, 0), (5, 3, 0)]
        intruder.path_index = 1
        intruder.move_interval = 1
        ai.intruders.append(intruder)

        initial_hp = intruder.hp
        ai._update_advancing(intruder)

        assert intruder.hp == initial_hp - SPIKE_DAMAGE // 2
        assert intruder.pos == (5, 2, 0)


# ── Tunneler digging ───────────────────────────────────────────────────


class TestTunnelerDigging:
    def test_tunneler_digs_through_stone(self):
        grid = _make_grid()
        for y in range(0, 4):
            grid.grid[5, y, 0] = VOXEL_AIR
        grid.grid[5, 2, 0] = VOXEL_STONE

        ai, bus, _, core, rng = _make_ai(grid=grid)

        intruder = _make_intruder(1, 5, 1, 0, arch=TUNNELER)
        intruder.state = IntruderState.ADVANCING
        intruder.path = [(5, 1, 0), (5, 2, 0), (5, 3, 0)]
        intruder.path_index = 1
        intruder.move_interval = 1
        ai.intruders.append(intruder)

        digging_events = []
        bus.subscribe("intruder_digging", lambda **kw: digging_events.append(kw))

        ai._update_advancing(intruder)

        assert intruder.state == IntruderState.INTERACTING
        assert intruder.interaction_type == "dig"
        assert len(digging_events) == 1

        expected_ticks = max(1, DIG_DURATION[VOXEL_STONE] // 2)
        assert intruder.interaction_ticks == expected_ticks

    def test_tunneler_completes_dig(self):
        grid = _make_grid()
        for y in range(0, 4):
            grid.grid[5, y, 0] = VOXEL_AIR
        grid.grid[5, 2, 0] = VOXEL_DIRT

        ai, bus, _, core, rng = _make_ai(grid=grid)

        intruder = _make_intruder(1, 5, 1, 0, arch=TUNNELER)
        intruder.state = IntruderState.ADVANCING
        intruder.path = [(5, 1, 0), (5, 2, 0), (5, 3, 0)]
        intruder.path_index = 1
        intruder.move_interval = 1
        ai.intruders.append(intruder)

        ai._update_advancing(intruder)
        assert intruder.state == IntruderState.INTERACTING

        dig_ticks = max(1, DIG_DURATION[VOXEL_DIRT] // 2)
        for _ in range(dig_ticks):
            ai._update_interacting(intruder)

        assert grid.get(5, 2, 0) == VOXEL_AIR
        assert intruder.state == IntruderState.ADVANCING

    def test_tunneler_cannot_dig_reinforced(self):
        grid = _make_grid()
        for y in range(0, 4):
            grid.grid[5, y, 0] = VOXEL_AIR
        grid.grid[5, 2, 0] = VOXEL_REINFORCED_WALL

        ai, bus, _, core, rng = _make_ai(grid=grid)

        intruder = _make_intruder(1, 5, 1, 0, arch=TUNNELER)
        intruder.state = IntruderState.ADVANCING
        intruder.path = [(5, 1, 0), (5, 2, 0), (5, 3, 0)]
        intruder.path_index = 1
        intruder.move_interval = 1
        ai.intruders.append(intruder)

        ai._update_advancing(intruder)

        assert intruder.state != IntruderState.INTERACTING


# ── Treasure collection ────────────────────────────────────────────────


class TestTreasureCollection:
    def test_shadowblade_collects_treasure(self):
        grid = _make_grid()
        for y in range(0, 4):
            grid.grid[5, y, 0] = VOXEL_AIR
        grid.grid[5, 2, 0] = VOXEL_TREASURE

        ai, bus, _, core, rng = _make_ai(grid=grid)

        intruder = _make_intruder(1, 5, 1, 0, arch=SHADOWBLADE)
        intruder.state = IntruderState.ADVANCING
        intruder.path = [(5, 1, 0), (5, 2, 0), (5, 3, 0)]
        intruder.path_index = 1
        intruder.move_interval = 1
        ai.intruders.append(intruder)

        ai._update_advancing(intruder)

        assert intruder.state == IntruderState.INTERACTING
        assert intruder.interaction_type == "grab_treasure"

    def test_treasure_collected_after_interaction(self):
        grid = _make_grid()
        for y in range(0, 4):
            grid.grid[5, y, 0] = VOXEL_AIR
        grid.grid[5, 2, 0] = VOXEL_TREASURE

        ai, bus, _, core, rng = _make_ai(grid=grid)

        collected_events = []
        bus.subscribe(
            "intruder_collected_treasure",
            lambda **kw: collected_events.append(kw),
        )

        intruder = _make_intruder(1, 5, 1, 0, arch=SHADOWBLADE)
        intruder.state = IntruderState.ADVANCING
        intruder.path = [(5, 1, 0), (5, 2, 0), (5, 3, 0)]
        intruder.path_index = 1
        intruder.move_interval = 1
        ai.intruders.append(intruder)

        ai._update_advancing(intruder)
        assert intruder.state == IntruderState.INTERACTING

        for _ in range(TREASURE_GRAB_TICKS):
            ai._update_interacting(intruder)

        assert grid.get(5, 2, 0) == VOXEL_AIR
        assert intruder.loot_count == 1
        assert len(collected_events) == 1


# ── Lava death ─────────────────────────────────────────────────────────


class TestLavaDeath:
    def test_non_fire_immune_dies_in_lava(self):
        grid = _make_grid()
        for y in range(0, 4):
            grid.grid[5, y, 0] = VOXEL_AIR
        grid.grid[5, 2, 0] = VOXEL_LAVA

        ai, bus, _, core, rng = _make_ai(grid=grid)

        died_events = []
        bus.subscribe("intruder_died", lambda **kw: died_events.append(kw))

        intruder = _make_intruder(1, 5, 1, 0, arch=VANGUARD)
        intruder.state = IntruderState.ADVANCING
        intruder.path = [(5, 1, 0), (5, 2, 0)]
        intruder.path_index = 1
        intruder.move_interval = 1
        ai.intruders.append(intruder)

        ai._update_advancing(intruder)

        assert intruder.state == IntruderState.DEAD
        assert len(died_events) == 1

    def test_pyremancer_walks_through_lava(self):
        grid = _make_grid()
        for y in range(0, 4):
            grid.grid[5, y, 0] = VOXEL_AIR
        grid.grid[5, 2, 0] = VOXEL_LAVA

        ai, bus, _, core, rng = _make_ai(grid=grid)

        intruder = _make_intruder(1, 5, 1, 0, arch=PYREMANCER)
        intruder.state = IntruderState.ADVANCING
        intruder.path = [(5, 1, 0), (5, 2, 0)]
        intruder.path_index = 1
        intruder.move_interval = 1
        ai.intruders.append(intruder)

        ai._update_advancing(intruder)

        assert intruder.alive
        assert intruder.pos == (5, 2, 0)


# ── Tarp fall ──────────────────────────────────────────────────────────


class TestTarpFall:
    def test_vanguard_falls_through_tarp(self):
        grid = _make_grid()
        for y in range(0, 4):
            grid.grid[5, y, 0] = VOXEL_AIR
        grid.grid[5, 2, 0] = VOXEL_TARP

        ai, bus, _, core, rng = _make_ai(grid=grid)

        fell_events = []
        bus.subscribe("intruder_fell", lambda **kw: fell_events.append(kw))

        intruder = _make_intruder(1, 5, 1, 0, arch=VANGUARD)
        intruder.state = IntruderState.ADVANCING
        intruder.path = [(5, 1, 0), (5, 2, 0)]
        intruder.path_index = 1
        intruder.move_interval = 1
        ai.intruders.append(intruder)

        ai._update_advancing(intruder)

        assert len(fell_events) == 1
        assert grid.get(5, 2, 0) == VOXEL_AIR  # Tarp destroyed
        assert (5, 2, 0) in intruder.personal_map.hazards


# ── Pyremancer heating ─────────────────────────────────────────────────


class TestPyremancerHeating:
    def test_pyremancer_heats_adjacent_blocks(self):
        grid = _make_grid()
        for y in range(0, 4):
            grid.grid[5, y, 0] = VOXEL_AIR

        ai, bus, _, core, rng = _make_ai(grid=grid)

        intruder = _make_intruder(1, 5, 1, 0, arch=PYREMANCER)
        intruder.state = IntruderState.ADVANCING
        intruder.path = [(5, 1, 0), (5, 2, 0)]
        intruder.path_index = 1
        intruder.move_interval = 100
        ai.intruders.append(intruder)

        initial_temp = float(grid.temperature[5, 2, 0])
        ai._tick_pyremancer_heat(intruder, PYREMANCER_HEAT_INTERVAL)

        assert grid.temperature[5, 2, 0] == pytest.approx(
            initial_temp + PYREMANCER_HEAT_AMOUNT
        )


# ── Attacking behavior ────────────────────────────────────────────────


class TestAttacking:
    def test_ranged_attacker_attacks_from_distance(self):
        ai, bus, grid, core, rng = _make_ai()

        # Pyremancer has attack_range=3, core at (5,5,3)
        # Place intruder at (5,2,3) with path to (5,3,3)
        intruder = _make_intruder(1, 5, 2, 3, arch=PYREMANCER)
        intruder.state = IntruderState.ADVANCING
        intruder.path = [(5, 2, 3), (5, 3, 3)]
        intruder.path_index = 1
        intruder.move_interval = 1
        ai.intruders.append(intruder)

        # Need to first move to (5,3,3), then check attack range
        ai._update_advancing(intruder)

        # After moving to (5,3,3), distance to core (5,5,3) is 2 <= 3
        assert intruder.pos == (5, 3, 3)
        assert intruder.state == IntruderState.ATTACKING

    def test_core_takes_damage_from_attacking_intruder(self):
        ai, bus, grid, core, rng = _make_ai()

        intruder = _make_intruder(1, 5, 5, 3, arch=VANGUARD)
        intruder.state = IntruderState.ATTACKING
        intruder.attack_interval = 1
        ai.intruders.append(intruder)

        for _ in range(5):
            ai._update_attacking(intruder)

        assert core.hp == 60  # 5 * 8 = 40 damage

    def test_frenzy_goreclaw_does_extra_damage(self):
        ai, bus, grid, core, rng = _make_ai()

        intruder = _make_intruder(1, 5, 5, 3, arch=GORECLAW)
        intruder.state = IntruderState.ATTACKING
        intruder.attack_interval = 1
        intruder.frenzy_active = True
        ai.intruders.append(intruder)

        for _ in range(3):
            ai._update_attacking(intruder)

        expected_hp = 100 - 3 * int(GORECLAW.damage * 1.5)
        assert core.hp == expected_hp


# ── Death handling ─────────────────────────────────────────────────────


class TestDeathHandling:
    def test_death_publishes_event(self):
        ai, bus, grid, core, rng = _make_ai()
        died = []
        bus.subscribe("intruder_died", lambda **kw: died.append(kw))

        intruder = _make_intruder(1, 5, 0, 0)
        ai.intruders.append(intruder)
        ai._on_intruder_death(intruder)

        assert intruder.state == IntruderState.DEAD
        assert len(died) == 1

    def test_party_notified_on_member_death(self):
        ai, bus, grid, core, rng = _make_ai()

        m1 = _make_intruder(1, 5, 0, 0, arch=WARDEN)
        m2 = _make_intruder(2, 5, 1, 0, arch=VANGUARD)
        party = Party(1, [m1, m2])
        ai.parties.append(party)
        ai.intruders.extend([m1, m2])

        ai._on_intruder_death(m1)

        assert m2.loyalty_modifier < 0


# ── Game over ──────────────────────────────────────────────────────────


class TestGameOver:
    def test_game_over_stops_ai(self):
        ai, bus, grid, core, rng = _make_ai()
        bus.publish("game_over", reason="core_destroyed")
        assert ai._game_over is True

        ai.spawning_enabled = True
        ai._spawn_timer = 9999
        bus.publish("tick", tick=1)
        assert len(ai.intruders) == 0


# ── Cleanup ────────────────────────────────────────────────────────────


class TestCleanup:
    def test_cleanup_removes_dead(self):
        ai, bus, grid, core, rng = _make_ai()
        m1 = _make_intruder(1, 5, 0, 0)
        m2 = _make_intruder(2, 5, 1, 0)
        ai.intruders.extend([m1, m2])

        m1.state = IntruderState.DEAD
        ai._cleanup()

        assert len(ai.intruders) == 1
        assert ai.intruders[0].id == 2

    def test_cleanup_removes_wiped_parties(self):
        ai, bus, grid, core, rng = _make_ai()
        m1 = _make_intruder(1, 5, 0, 0)
        party = Party(1, [m1])
        ai.parties.append(party)

        m1.state = IntruderState.DEAD
        ai._cleanup()

        assert len(ai.parties) == 0


# ── Pillaging ──────────────────────────────────────────────────────────


class TestPillaging:
    def test_betrayed_intruder_heads_to_treasure(self):
        grid = _make_grid()
        for y in range(0, 6):
            grid.grid[5, y, 0] = VOXEL_AIR
        grid.grid[5, 4, 0] = VOXEL_TREASURE

        ai, bus, _, core, rng = _make_ai(grid=grid)

        intruder = _make_intruder(1, 5, 1, 0, arch=SHADOWBLADE,
                                   objective=IntruderObjective.PILLAGE)
        intruder.state = IntruderState.PILLAGING
        intruder.move_interval = 1
        ai.intruders.append(intruder)

        # Reveal the corridor and treasure
        for y in range(0, 6):
            intruder.personal_map.reveal(5, y, 0, VOXEL_AIR)
        intruder.personal_map.reveal(5, 4, 0, VOXEL_TREASURE)

        ai._update_pillaging(intruder)

        # Should have found a path to the treasure
        assert intruder.path is not None
        assert len(intruder.path) > 1

    def test_pillager_retreats_when_no_treasure(self):
        grid = _make_grid()
        for y in range(0, 4):
            grid.grid[5, y, 0] = VOXEL_AIR

        ai, bus, _, core, rng = _make_ai(grid=grid)

        intruder = _make_intruder(1, 5, 1, 0, arch=SHADOWBLADE,
                                   objective=IntruderObjective.PILLAGE)
        intruder.state = IntruderState.PILLAGING
        intruder.move_interval = 1
        # Reveal some cells so retreat pathfinding works
        for y in range(0, 4):
            intruder.personal_map.reveal(5, y, 0, VOXEL_AIR)
        ai.intruders.append(intruder)

        ai._update_pillaging(intruder)

        assert intruder.state == IntruderState.RETREATING
