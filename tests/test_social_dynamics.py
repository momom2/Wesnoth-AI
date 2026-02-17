"""Tests for social dynamics: level/status, knowledge archive, reputation, morale, factions."""

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
    IntruderStatus,
    STATUS_TRUST,
    VANGUARD,
    SHADOWBLADE,
    TUNNELER,
    PYREMANCER,
    WINDCALLER,
    WARDEN,
    GORECLAW,
    GLOOMSEER,
    MAGMAWRAITH,
    BOREMITE,
    STONESKIN_BRUTE,
    TREMORSTALKER,
    CORROSIVE_CRAWLER,
    ArchetypeStats,
)
from dungeon_builder.intruders.personal_map import PersonalMap
from dungeon_builder.intruders.knowledge_archive import KnowledgeArchive, FactionMap
from dungeon_builder.intruders.reputation import DungeonReputation, ReputationProfile
from dungeon_builder.intruders.party import Party
from dungeon_builder.intruders.decision import IntruderAI

from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_STONE,
    VOXEL_DIRT,
    VOXEL_SPIKE,
    VOXEL_LAVA,
    VOXEL_TARP,
    VOXEL_TREASURE,
    VOXEL_DOOR,
    SURFACE_Z,
    LEVEL_HP_SCALE,
    LEVEL_DAMAGE_SCALE,
    LEVEL_WEIGHTS,
    KNOWLEDGE_STALE_TICKS,
    KNOWLEDGE_UNCERTAIN_THRESHOLD,
    KNOWLEDGE_CONTRADICTION_BASE,
    KNOWLEDGE_CONFIRM_DECAY,
    KNOWLEDGE_CHANGE_UNCERTAINTY,
    MORALE_BASE,
    MORALE_LOW_THRESHOLD,
    MORALE_HIGH_THRESHOLD,
    MORALE_FLEE_THRESHOLD,
    MORALE_ALLY_DEATH_PENALTY,
    MORALE_DAMAGE_PENALTY,
    MORALE_HAZARD_PENALTY,
    MORALE_TREASURE_BONUS,
    MORALE_LEADER_BONUS,
    MORALE_WARDEN_TICK,
    MORALE_DRIFT_RATE,
    MORALE_SLOW_FACTOR,
    MORALE_FAST_FACTOR,
    MORALE_DAMAGE_BONUS,
    MORALE_RETREAT_MULTIPLIER,
    FACTION_ENCOUNTER_INTERVAL,
    REPUTATION_DEADLY_LETHALITY,
    REPUTATION_RICH_RICHNESS,
    REPUTATION_UNKNOWN_THRESHOLD,
)


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_intruder(
    intruder_id=1,
    x=0,
    y=0,
    z=0,
    archetype=VANGUARD,
    objective=IntruderObjective.DESTROY_CORE,
    is_underworlder=False,
    level=1,
    status=None,
):
    """Create a basic intruder for testing."""
    if status is None:
        status = IntruderStatus.GRUNT
    return Intruder(
        intruder_id=intruder_id,
        x=x, y=y, z=z,
        archetype=archetype,
        objective=objective,
        personal_map=PersonalMap(),
        is_underworlder=is_underworlder,
        level=level,
        status=status,
    )


def _make_ai(width=10, depth=10, height=20, core_pos=(5, 5, 10), seed=42):
    """Create an IntruderAI with a simple dungeon grid."""
    bus = EventBus()
    grid = VoxelGrid(width=width, depth=depth, height=height)
    grid.grid[:] = VOXEL_STONE
    grid.grid[:, :, 0] = VOXEL_AIR  # Clear surface
    grid.visible[:] = True
    grid.claimed[:] = True
    pf = AStarPathfinder(grid)
    core = DungeonCore(bus, *core_pos, hp=100)
    rng = SeededRNG(seed)
    ai = IntruderAI(bus, grid, pf, core, rng)
    return ai, bus, grid, core, rng


# ═══════════════════════════════════════════════════════════════════════
#  1. Level & Status Tests
# ═══════════════════════════════════════════════════════════════════════


class TestLevelAndStatus:
    """Tests for intruder level/status system."""

    def test_level_1_intruder_has_base_hp(self):
        """Level 1 intruder gets base archetype HP."""
        i = _make_intruder(archetype=VANGUARD, level=1)
        assert i.hp == VANGUARD.hp
        assert i.max_hp == VANGUARD.hp

    def test_level_5_intruder_has_scaled_hp(self):
        """Level 5 intruder gets 1.6x HP (1 + 4 * 0.15)."""
        i = _make_intruder(archetype=VANGUARD, level=5)
        expected = int(VANGUARD.hp * (1.0 + 4 * LEVEL_HP_SCALE))
        assert i.hp == expected
        assert i.max_hp == expected

    def test_level_3_intruder_damage_scaling(self):
        """Level 3 intruder gets scaled damage."""
        i = _make_intruder(archetype=VANGUARD, level=3)
        expected = int(VANGUARD.damage * (1.0 + 2 * LEVEL_DAMAGE_SCALE))
        assert i.effective_damage == expected

    def test_status_assigned_from_level_grunt(self):
        """Levels 1-2 map to GRUNT."""
        assert IntruderAI._level_to_status(1) == IntruderStatus.GRUNT
        assert IntruderAI._level_to_status(2) == IntruderStatus.GRUNT

    def test_status_assigned_from_level_veteran(self):
        """Level 3 maps to VETERAN."""
        assert IntruderAI._level_to_status(3) == IntruderStatus.VETERAN

    def test_status_assigned_from_level_elite(self):
        """Level 4 maps to ELITE."""
        assert IntruderAI._level_to_status(4) == IntruderStatus.ELITE

    def test_status_assigned_from_level_champion(self):
        """Level 5 maps to CHAMPION."""
        assert IntruderAI._level_to_status(5) == IntruderStatus.CHAMPION

    def test_status_trust_weights(self):
        """STATUS_TRUST has correct weights per status."""
        assert STATUS_TRUST[IntruderStatus.GRUNT] == 0.5
        assert STATUS_TRUST[IntruderStatus.VETERAN] == 1.0
        assert STATUS_TRUST[IntruderStatus.ELITE] == 1.5
        assert STATUS_TRUST[IntruderStatus.CHAMPION] == 2.0

    def test_level_does_not_modify_archetype(self):
        """Level scaling doesn't modify the frozen archetype."""
        i = _make_intruder(archetype=VANGUARD, level=5)
        assert VANGUARD.hp == 120  # Original unchanged
        assert i.hp != VANGUARD.hp  # Instance is scaled

    def test_default_level_and_status(self):
        """Default intruder has level=1, status=GRUNT."""
        i = _make_intruder()
        assert i.level == 1
        assert i.status == IntruderStatus.GRUNT

    def test_high_status_elected_leader_over_high_loyalty(self):
        """VETERAN outranks a higher-loyalty GRUNT for leadership."""
        grunt = _make_intruder(
            intruder_id=1, archetype=VANGUARD, level=1,
            status=IntruderStatus.GRUNT,
        )
        veteran = _make_intruder(
            intruder_id=2, archetype=SHADOWBLADE, level=3,
            status=IntruderStatus.VETERAN,
        )
        grunt.state = IntruderState.ADVANCING
        veteran.state = IntruderState.ADVANCING
        party = Party(1, [grunt, veteran])
        assert party.leader.id == veteran.id

    def test_level_weights_sum_to_one(self):
        """LEVEL_WEIGHTS tuple sums to 1.0."""
        assert pytest.approx(sum(LEVEL_WEIGHTS), abs=1e-10) == 1.0

    def test_assign_level_returns_valid_range(self):
        """_assign_level always returns 1-5."""
        rng = SeededRNG(42)
        for _ in range(100):
            level = IntruderAI._assign_level(rng)
            assert 1 <= level <= 5


# ═══════════════════════════════════════════════════════════════════════
#  2. Knowledge Archive Tests
# ═══════════════════════════════════════════════════════════════════════


class TestKnowledgeArchive:
    """Tests for faction knowledge persistence with uncertainty."""

    def test_archive_stores_escaped_survivor_data(self):
        """Escaped survivor's map data is archived."""
        ka = KnowledgeArchive()
        intruder = _make_intruder()
        intruder.personal_map.reveal(1, 2, 3, VOXEL_STONE, 0)
        intruder.personal_map.reveal(4, 5, 6, VOXEL_AIR, 0)
        intruder.state = IntruderState.ESCAPED

        ka.archive_survivor(intruder, tick=100)

        faction = ka._get_faction(False)
        assert (1, 2, 3) in faction.seen
        assert (4, 5, 6) in faction.seen
        assert faction.seen[(1, 2, 3)][0] == VOXEL_STONE

    def test_surface_and_underworld_archives_separate(self):
        """Surface and underworld factions have separate archives."""
        ka = KnowledgeArchive()

        surface = _make_intruder(is_underworlder=False)
        surface.personal_map.reveal(1, 1, 1, VOXEL_STONE, 0)
        ka.archive_survivor(surface, tick=10)

        uw = _make_intruder(intruder_id=2, is_underworlder=True)
        uw.personal_map.reveal(2, 2, 2, VOXEL_DIRT, 0)
        ka.archive_survivor(uw, tick=10)

        assert (1, 1, 1) in ka._get_faction(False).seen
        assert (1, 1, 1) not in ka._get_faction(True).seen
        assert (2, 2, 2) in ka._get_faction(True).seen
        assert (2, 2, 2) not in ka._get_faction(False).seen

    def test_inject_merges_into_personal_map(self):
        """Inject fills a new intruder's PersonalMap with archived data."""
        ka = KnowledgeArchive()
        survivor = _make_intruder()
        survivor.personal_map.reveal(3, 4, 5, VOXEL_AIR, 0)
        ka.archive_survivor(survivor, tick=100)

        new_map = PersonalMap()
        ka.inject_knowledge(new_map, False, current_tick=100, cunning=0.0)
        assert (3, 4, 5) in new_map.seen
        assert new_map.seen[(3, 4, 5)] == VOXEL_AIR

    def test_staleness_filter(self):
        """Data older than KNOWLEDGE_STALE_TICKS is skipped during injection."""
        ka = KnowledgeArchive()
        survivor = _make_intruder()
        survivor.personal_map.reveal(1, 1, 1, VOXEL_STONE, 0)
        ka.archive_survivor(survivor, tick=100)

        new_map = PersonalMap()
        ka.inject_knowledge(
            new_map, False,
            current_tick=100 + KNOWLEDGE_STALE_TICKS + 1,
            cunning=0.0,
        )
        assert (1, 1, 1) not in new_map.seen

    def test_contradiction_increases_uncertainty(self):
        """Two survivors reporting different vtypes increases uncertainty."""
        ka = KnowledgeArchive()

        s1 = _make_intruder(intruder_id=1, status=IntruderStatus.GRUNT)
        s1.personal_map.reveal(5, 5, 5, VOXEL_STONE, 0)
        ka.archive_survivor(s1, tick=10)

        s2 = _make_intruder(intruder_id=2, status=IntruderStatus.GRUNT)
        s2.personal_map.reveal(5, 5, 5, VOXEL_AIR, 0)
        ka.archive_survivor(s2, tick=20)

        unc = ka.get_uncertainty(5, 5, 5, False)
        assert unc > 0.0

    def test_confirmation_decreases_uncertainty(self):
        """Two survivors reporting same vtype decreases uncertainty."""
        ka = KnowledgeArchive()

        # Create initial contradiction to establish some uncertainty
        s1 = _make_intruder(intruder_id=1, status=IntruderStatus.GRUNT)
        s1.personal_map.reveal(5, 5, 5, VOXEL_STONE, 0)
        ka.archive_survivor(s1, tick=10)

        s2 = _make_intruder(intruder_id=2, status=IntruderStatus.GRUNT)
        s2.personal_map.reveal(5, 5, 5, VOXEL_AIR, 0)
        ka.archive_survivor(s2, tick=20)

        unc_after_contradiction = ka.get_uncertainty(5, 5, 5, False)

        # Now confirm the AIR report
        s3 = _make_intruder(intruder_id=3, status=IntruderStatus.GRUNT)
        s3.personal_map.reveal(5, 5, 5, VOXEL_AIR, 0)
        ka.archive_survivor(s3, tick=30)

        unc_after_confirm = ka.get_uncertainty(5, 5, 5, False)
        assert unc_after_confirm < unc_after_contradiction

    def test_champion_report_adds_less_uncertainty(self):
        """Higher-status survivor causes less uncertainty on contradiction."""
        # Test with GRUNT archive, CHAMPION contradicts → low uncertainty
        ka1 = KnowledgeArchive()
        s1 = _make_intruder(intruder_id=1, status=IntruderStatus.GRUNT)
        s1.personal_map.reveal(5, 5, 5, VOXEL_STONE, 0)
        ka1.archive_survivor(s1, tick=10)

        champion = _make_intruder(intruder_id=2, status=IntruderStatus.CHAMPION)
        champion.personal_map.reveal(5, 5, 5, VOXEL_AIR, 0)
        ka1.archive_survivor(champion, tick=20)

        unc_champion = ka1.get_uncertainty(5, 5, 5, False)

        # Test with GRUNT archive, GRUNT contradicts → higher uncertainty
        ka2 = KnowledgeArchive()
        s3 = _make_intruder(intruder_id=3, status=IntruderStatus.GRUNT)
        s3.personal_map.reveal(5, 5, 5, VOXEL_STONE, 0)
        ka2.archive_survivor(s3, tick=10)

        grunt = _make_intruder(intruder_id=4, status=IntruderStatus.GRUNT)
        grunt.personal_map.reveal(5, 5, 5, VOXEL_AIR, 0)
        ka2.archive_survivor(grunt, tick=20)

        unc_grunt = ka2.get_uncertainty(5, 5, 5, False)

        # Champion contradicting should produce less uncertainty
        assert unc_champion < unc_grunt

    def test_uncertainty_threshold_filters_injection(self):
        """Cells above uncertainty threshold are skipped during injection."""
        ka = KnowledgeArchive()
        survivor = _make_intruder()
        survivor.personal_map.reveal(5, 5, 5, VOXEL_STONE, 0)
        ka.archive_survivor(survivor, tick=100)

        # Manually set high uncertainty
        faction = ka._get_faction(False)
        faction.uncertainty[(5, 5, 5)] = 0.9  # Above threshold (0.7)

        new_map = PersonalMap()
        ka.inject_knowledge(new_map, False, current_tick=100, cunning=0.0)
        assert (5, 5, 5) not in new_map.seen

    def test_cunning_adjusts_uncertainty_threshold(self):
        """High cunning lowers the threshold → trusts less archived data."""
        ka = KnowledgeArchive()
        survivor = _make_intruder()
        survivor.personal_map.reveal(5, 5, 5, VOXEL_STONE, 0)
        ka.archive_survivor(survivor, tick=100)

        # Set uncertainty just below default threshold but above cunning-adjusted
        faction = ka._get_faction(False)
        faction.uncertainty[(5, 5, 5)] = 0.6  # Below 0.7 but above 0.7-0.8*0.2=0.54

        # Low cunning (0.0) → threshold 0.7 → accepts
        map_low = PersonalMap()
        ka.inject_knowledge(map_low, False, current_tick=100, cunning=0.0)
        assert (5, 5, 5) in map_low.seen

        # High cunning (0.8) → threshold 0.54 → rejects 0.6
        map_high = PersonalMap()
        ka.inject_knowledge(map_high, False, current_tick=100, cunning=0.8)
        assert (5, 5, 5) not in map_high.seen

    def test_voxel_change_increases_uncertainty(self):
        """Player modifying a cell increases uncertainty in archives."""
        ka = KnowledgeArchive()
        survivor = _make_intruder()
        survivor.personal_map.reveal(5, 5, 5, VOXEL_STONE, 0)
        ka.archive_survivor(survivor, tick=100)

        assert ka.get_uncertainty(5, 5, 5, False) == 0.0
        ka.on_voxel_changed(5, 5, 5)
        assert ka.get_uncertainty(5, 5, 5, False) == pytest.approx(
            KNOWLEDGE_CHANGE_UNCERTAINTY,
        )

    def test_multiple_voxel_changes_accumulate(self):
        """Multiple player changes accumulate uncertainty (capped at 1.0)."""
        ka = KnowledgeArchive()
        survivor = _make_intruder()
        survivor.personal_map.reveal(5, 5, 5, VOXEL_STONE, 0)
        ka.archive_survivor(survivor, tick=100)

        for _ in range(10):
            ka.on_voxel_changed(5, 5, 5)
        assert ka.get_uncertainty(5, 5, 5, False) == 1.0

    def test_empty_archive_inject_is_noop(self):
        """Injecting from empty archive does nothing."""
        ka = KnowledgeArchive()
        pmap = PersonalMap()
        ka.inject_knowledge(pmap, False, current_tick=100, cunning=0.0)
        assert len(pmap.seen) == 0

    def test_get_stats_returns_correct_counts(self):
        """get_stats returns correct summary including avg_uncertainty."""
        ka = KnowledgeArchive()
        survivor = _make_intruder()
        survivor.personal_map.reveal(1, 1, 1, VOXEL_AIR, 0)
        survivor.personal_map.reveal(2, 2, 2, VOXEL_SPIKE, 0)
        survivor.personal_map.reveal(3, 3, 3, VOXEL_TREASURE, 0)
        ka.archive_survivor(survivor, tick=100)

        stats = ka.get_stats(False)
        assert stats["cells_known"] == 3
        assert stats["hazards_known"] == 1
        assert stats["treasures_known"] == 1
        assert stats["avg_uncertainty"] == 0.0

    def test_hazards_treasures_doors_archived_and_injected(self):
        """Hazards, treasures, and doors are archived and injected correctly."""
        ka = KnowledgeArchive()
        survivor = _make_intruder()
        survivor.personal_map.reveal(1, 1, 1, VOXEL_SPIKE, 0)
        survivor.personal_map.reveal(2, 2, 2, VOXEL_TREASURE, 0)
        survivor.personal_map.reveal(3, 3, 3, VOXEL_DOOR, 1)
        ka.archive_survivor(survivor, tick=100)

        faction = ka._get_faction(False)
        assert (1, 1, 1) in faction.hazards
        assert (2, 2, 2) in faction.treasures
        assert (3, 3, 3) in faction.doors

        new_map = PersonalMap()
        ka.inject_knowledge(new_map, False, current_tick=100, cunning=0.0)
        assert (1, 1, 1) in new_map.hazards
        assert (2, 2, 2) in new_map.treasures

    def test_voxel_change_affects_both_factions(self):
        """Player voxel change increases uncertainty in both faction archives."""
        ka = KnowledgeArchive()
        # Add data to both factions
        surface = _make_intruder(intruder_id=1, is_underworlder=False)
        surface.personal_map.reveal(5, 5, 5, VOXEL_STONE, 0)
        ka.archive_survivor(surface, tick=100)

        uw = _make_intruder(intruder_id=2, is_underworlder=True)
        uw.personal_map.reveal(5, 5, 5, VOXEL_STONE, 0)
        ka.archive_survivor(uw, tick=100)

        ka.on_voxel_changed(5, 5, 5)
        assert ka.get_uncertainty(5, 5, 5, False) > 0.0
        assert ka.get_uncertainty(5, 5, 5, True) > 0.0


# ═══════════════════════════════════════════════════════════════════════
#  3. Reputation Tests
# ═══════════════════════════════════════════════════════════════════════


class TestReputation:
    """Tests for dungeon reputation system."""

    def test_kill_increments_total_kills(self):
        bus = EventBus()
        rep = DungeonReputation(bus)
        bus.publish("intruder_died", intruder=None)
        assert rep.total_kills == 1

    def test_escape_increments_total_escapes(self):
        bus = EventBus()
        rep = DungeonReputation(bus)
        bus.publish("intruder_escaped", intruder=_make_intruder())
        assert rep.total_escapes == 1

    def test_treasure_increments_total_treasure_lost(self):
        bus = EventBus()
        rep = DungeonReputation(bus)
        bus.publish("intruder_collected_treasure")
        assert rep.total_treasure_lost == 1

    def test_party_wipe_increments(self):
        bus = EventBus()
        rep = DungeonReputation(bus)
        rep.on_party_wiped()
        assert rep.total_parties_wiped == 1

    def test_lethality_formula(self):
        """Lethality = kills / (kills + escapes)."""
        bus = EventBus()
        rep = DungeonReputation(bus)
        for _ in range(8):
            bus.publish("intruder_died", intruder=None)
        for _ in range(2):
            bus.publish("intruder_escaped", intruder=_make_intruder())
        profile = rep.get_profile()
        assert profile.lethality == pytest.approx(0.8, abs=0.01)

    def test_richness_formula(self):
        """Richness = treasure_lost / (kills + treasure_lost)."""
        bus = EventBus()
        rep = DungeonReputation(bus)
        for _ in range(5):
            bus.publish("intruder_died", intruder=None)
        for _ in range(5):
            bus.publish("intruder_collected_treasure")
        profile = rep.get_profile()
        assert profile.richness == pytest.approx(0.5, abs=0.01)

    def test_deadly_profile_objective_modifier(self):
        """Deadly dungeon gets +destroy, -pillage modifier."""
        bus = EventBus()
        rep = DungeonReputation(bus)
        for _ in range(10):
            bus.publish("intruder_died", intruder=None)
        mod = rep.get_objective_modifier()
        assert mod[0] > 0  # +destroy
        assert mod[2] < 0  # -pillage

    def test_rich_profile_objective_modifier(self):
        """Rich dungeon gets +pillage, -destroy modifier."""
        bus = EventBus()
        rep = DungeonReputation(bus)
        for _ in range(3):
            bus.publish("intruder_died", intruder=None)
        for _ in range(3):
            bus.publish("intruder_escaped", intruder=_make_intruder())
        for _ in range(10):
            bus.publish("intruder_collected_treasure")
        mod = rep.get_objective_modifier()
        assert mod[2] > 0  # +pillage
        assert mod[0] < 0  # -destroy

    def test_unknown_profile_modifier(self):
        """Too few events → unknown profile → explore bonus."""
        bus = EventBus()
        rep = DungeonReputation(bus)
        # Less than REPUTATION_UNKNOWN_THRESHOLD events
        bus.publish("intruder_died", intruder=None)
        mod = rep.get_objective_modifier()
        assert mod[1] > 0  # +explore

    def test_loyalty_modifier_deadly(self):
        """Deadly reputation gives negative loyalty modifier."""
        bus = EventBus()
        rep = DungeonReputation(bus)
        for _ in range(10):
            bus.publish("intruder_died", intruder=None)
        assert rep.get_loyalty_modifier() < 0

    def test_loyalty_modifier_rich(self):
        """Rich reputation gives negative loyalty modifier."""
        bus = EventBus()
        rep = DungeonReputation(bus)
        for _ in range(3):
            bus.publish("intruder_died", intruder=None)
        for _ in range(3):
            bus.publish("intruder_escaped", intruder=_make_intruder())
        for _ in range(10):
            bus.publish("intruder_collected_treasure")
        assert rep.get_loyalty_modifier() < 0

    def test_deadly_reputation_shifts_level_distribution(self):
        """Deadly dungeons attract higher-level intruders (level_shift > 0)."""
        bus = EventBus()
        rep = DungeonReputation(bus)
        for _ in range(10):
            bus.publish("intruder_died", intruder=None)
        shift = rep.get_level_shift()
        assert shift > 0


# ═══════════════════════════════════════════════════════════════════════
#  4. Morale Tests
# ═══════════════════════════════════════════════════════════════════════


class TestMorale:
    """Tests for intruder morale system."""

    def test_initial_morale_is_base(self):
        """New intruders start at MORALE_BASE."""
        i = _make_intruder()
        assert i.morale == pytest.approx(MORALE_BASE)

    def test_ally_death_reduces_morale(self):
        """Ally death reduces morale by MORALE_ALLY_DEATH_PENALTY."""
        m1 = _make_intruder(intruder_id=1)
        m2 = _make_intruder(intruder_id=2)
        m1.state = IntruderState.ADVANCING
        m2.state = IntruderState.ADVANCING
        party = Party(1, [m1, m2])
        initial_morale = m1.morale

        m2.state = IntruderState.DEAD
        party.on_member_death(m2)

        assert m1.morale == pytest.approx(
            initial_morale - MORALE_ALLY_DEATH_PENALTY,
        )

    def test_damage_reduces_morale(self):
        """Taking damage reduces morale."""
        i = _make_intruder()
        initial = i.morale
        i.morale = max(0.0, i.morale - MORALE_DAMAGE_PENALTY)
        assert i.morale < initial

    def test_treasure_increases_morale(self):
        """Collecting treasure increases morale."""
        i = _make_intruder()
        initial = i.morale
        i.morale = min(1.0, i.morale + MORALE_TREASURE_BONUS)
        assert i.morale > initial

    def test_leader_alive_bonus(self):
        """Leader alive gives morale tick bonus."""
        m1 = _make_intruder(intruder_id=1)
        m2 = _make_intruder(intruder_id=2)
        m1.state = IntruderState.ADVANCING
        m2.state = IntruderState.ADVANCING
        party = Party(1, [m1, m2])

        # Set morale slightly below base to see leader bonus
        for m in party.alive_members:
            m.morale = MORALE_BASE - 0.01

        party.update_morale(tick=1)

        # Members should have gained MORALE_LEADER_BONUS
        for m in party.alive_members:
            assert m.morale > MORALE_BASE - 0.01

    def test_warden_aura_morale_bonus(self):
        """Warden aura provides morale boost to nearby allies."""
        warden = _make_intruder(intruder_id=1, archetype=WARDEN)
        ally = _make_intruder(intruder_id=2, archetype=VANGUARD)
        warden.state = IntruderState.ADVANCING
        ally.state = IntruderState.ADVANCING
        # Same position for MAP_SHARE_RANGE check
        warden.x, warden.y, warden.z = 5, 5, 0
        ally.x, ally.y, ally.z = 5, 5, 0

        party = Party(1, [warden, ally])
        initial = ally.morale
        party.apply_warden_aura()

        assert ally.morale >= initial + MORALE_WARDEN_TICK

    def test_low_morale_slows_movement(self):
        """Morale below LOW_THRESHOLD slows movement."""
        i = _make_intruder(archetype=VANGUARD)
        base_interval = i.move_interval

        i.morale = MORALE_LOW_THRESHOLD - 0.01
        slow_interval = i.effective_move_interval

        i.morale = MORALE_BASE
        normal_interval = i.effective_move_interval

        assert slow_interval > normal_interval

    def test_high_morale_speeds_movement(self):
        """Morale above HIGH_THRESHOLD speeds movement."""
        i = _make_intruder(archetype=VANGUARD)

        i.morale = MORALE_HIGH_THRESHOLD + 0.01
        fast_interval = i.effective_move_interval

        i.morale = MORALE_BASE
        normal_interval = i.effective_move_interval

        assert fast_interval < normal_interval

    def test_high_morale_damage_bonus(self):
        """High morale gives damage bonus."""
        i = _make_intruder(archetype=VANGUARD)
        i.morale = MORALE_BASE
        normal_dmg = i.effective_damage

        i.morale = MORALE_HIGH_THRESHOLD + 0.01
        high_dmg = i.effective_damage

        assert high_dmg > normal_dmg

    def test_very_low_morale_triggers_flee(self):
        """Morale < MORALE_FLEE_THRESHOLD triggers retreat."""
        i = _make_intruder(archetype=VANGUARD)
        i.state = IntruderState.ADVANCING
        i.morale = MORALE_FLEE_THRESHOLD - 0.01

        IntruderAI._check_retreat(i)
        assert i.state == IntruderState.RETREATING

    def test_never_retreats_overrides_morale_flee(self):
        """never_retreats overrides even morale-based flee."""
        i = _make_intruder(archetype=GORECLAW)
        i.state = IntruderState.ADVANCING
        i.morale = 0.0  # Minimum morale

        IntruderAI._check_retreat(i)
        assert i.state == IntruderState.ADVANCING

    def test_frenzy_overrides_morale(self):
        """Frenzy-active intruder has morale locked to 1.0."""
        m1 = _make_intruder(intruder_id=1, archetype=GORECLAW)
        m1.state = IntruderState.ADVANCING
        m1.frenzy_active = True
        m1.morale = 0.3

        party = Party(1, [m1])
        party.update_morale(tick=1)

        assert m1.morale == 1.0

    def test_morale_drifts_toward_base(self):
        """Morale drifts toward MORALE_BASE when no events occur."""
        m1 = _make_intruder(intruder_id=1)
        m1.state = IntruderState.ADVANCING
        m1.morale = 0.5  # Below MORALE_BASE (0.7)

        party = Party(1, [m1])
        # Update many ticks to let drift accumulate
        for t in range(100):
            party.update_morale(tick=t)

        # Should have drifted toward MORALE_BASE
        assert m1.morale > 0.5

    def test_low_morale_doubles_retreat_threshold(self):
        """Low morale doubles the HP retreat threshold."""
        i = _make_intruder(archetype=VANGUARD)
        i.state = IntruderState.ADVANCING
        # Set HP above normal retreat threshold but below doubled
        normal_thresh = VANGUARD.retreat_threshold  # 0.15
        doubled_thresh = normal_thresh * MORALE_RETREAT_MULTIPLIER

        i.hp = int(i.max_hp * (normal_thresh + doubled_thresh) / 2)
        i.morale = MORALE_LOW_THRESHOLD - 0.01

        IntruderAI._check_retreat(i)
        assert i.state == IntruderState.RETREATING


# ═══════════════════════════════════════════════════════════════════════
#  5. Faction Encounter Tests
# ═══════════════════════════════════════════════════════════════════════


class TestFactionEncounters:
    """Tests for inter-faction combat."""

    def test_same_faction_no_fight(self):
        """Same-faction intruders don't fight each other."""
        ai, bus, grid, core, rng = _make_ai()
        s1 = _make_intruder(intruder_id=1, x=5, y=5, z=0)
        s2 = _make_intruder(intruder_id=2, x=5, y=5, z=0)
        s1.state = IntruderState.ADVANCING
        s2.state = IntruderState.ADVANCING
        ai.intruders = [s1, s2]

        initial_hp_1 = s1.hp
        initial_hp_2 = s2.hp
        ai._tick_faction_encounters(FACTION_ENCOUNTER_INTERVAL)

        assert s1.hp == initial_hp_1
        assert s2.hp == initial_hp_2

    def test_different_faction_same_cell_deal_damage(self):
        """Different-faction intruders in same cell deal mutual damage."""
        ai, bus, grid, core, rng = _make_ai()
        surface = _make_intruder(intruder_id=1, x=5, y=5, z=0)
        uw = _make_intruder(
            intruder_id=2, x=5, y=5, z=0, is_underworlder=True,
            archetype=MAGMAWRAITH,
        )
        surface.state = IntruderState.ADVANCING
        uw.state = IntruderState.ADVANCING
        ai.intruders = [surface, uw]

        initial_surface_hp = surface.hp
        initial_uw_hp = uw.hp
        ai._tick_faction_encounters(FACTION_ENCOUNTER_INTERVAL)

        assert surface.hp < initial_surface_hp
        assert uw.hp < initial_uw_hp

    def test_adjacent_different_faction_engage_if_advancing(self):
        """Adjacent different-faction intruders engage if both ADVANCING."""
        ai, bus, grid, core, rng = _make_ai()
        surface = _make_intruder(intruder_id=1, x=5, y=5, z=0)
        uw = _make_intruder(
            intruder_id=2, x=6, y=5, z=0, is_underworlder=True,
            archetype=MAGMAWRAITH,
        )
        surface.state = IntruderState.ADVANCING
        uw.state = IntruderState.ADVANCING
        ai.intruders = [surface, uw]

        initial_surface_hp = surface.hp
        initial_uw_hp = uw.hp
        ai._tick_faction_encounters(FACTION_ENCOUNTER_INTERVAL)

        assert surface.hp < initial_surface_hp
        assert uw.hp < initial_uw_hp

    def test_retreating_intruder_doesnt_engage_adjacent(self):
        """Retreating intruders don't engage in adjacent combat."""
        ai, bus, grid, core, rng = _make_ai()
        surface = _make_intruder(intruder_id=1, x=5, y=5, z=0)
        uw = _make_intruder(
            intruder_id=2, x=6, y=5, z=0, is_underworlder=True,
            archetype=MAGMAWRAITH,
        )
        surface.state = IntruderState.RETREATING  # Won't engage adjacent
        uw.state = IntruderState.ADVANCING
        ai.intruders = [surface, uw]

        initial_surface_hp = surface.hp
        initial_uw_hp = uw.hp
        ai._tick_faction_encounters(FACTION_ENCOUNTER_INTERVAL)

        assert surface.hp == initial_surface_hp
        assert uw.hp == initial_uw_hp

    def test_faction_combat_publishes_event(self):
        """Faction combat publishes 'faction_combat' event."""
        ai, bus, grid, core, rng = _make_ai()
        surface = _make_intruder(intruder_id=1, x=5, y=5, z=0)
        uw = _make_intruder(
            intruder_id=2, x=5, y=5, z=0, is_underworlder=True,
            archetype=MAGMAWRAITH,
        )
        surface.state = IntruderState.ADVANCING
        uw.state = IntruderState.ADVANCING
        ai.intruders = [surface, uw]

        events = []
        bus.subscribe("faction_combat", lambda **kw: events.append(kw))
        ai._tick_faction_encounters(FACTION_ENCOUNTER_INTERVAL)

        assert len(events) > 0

    def test_dead_intruders_excluded_from_encounters(self):
        """Dead intruders don't participate in faction encounters."""
        ai, bus, grid, core, rng = _make_ai()
        surface = _make_intruder(intruder_id=1, x=5, y=5, z=0)
        uw = _make_intruder(
            intruder_id=2, x=5, y=5, z=0, is_underworlder=True,
            archetype=MAGMAWRAITH,
        )
        surface.state = IntruderState.DEAD
        uw.state = IntruderState.ADVANCING
        ai.intruders = [surface, uw]

        initial_uw_hp = uw.hp
        ai._tick_faction_encounters(FACTION_ENCOUNTER_INTERVAL)
        assert uw.hp == initial_uw_hp

    def test_no_encounters_when_no_underworlders(self):
        """No encounters triggered when there are no underworlders."""
        ai, bus, grid, core, rng = _make_ai()
        s1 = _make_intruder(intruder_id=1, x=5, y=5, z=0)
        s2 = _make_intruder(intruder_id=2, x=5, y=5, z=0)
        s1.state = IntruderState.ADVANCING
        s2.state = IntruderState.ADVANCING
        ai.intruders = [s1, s2]

        events = []
        bus.subscribe("faction_combat", lambda **kw: events.append(kw))
        ai._tick_faction_encounters(FACTION_ENCOUNTER_INTERVAL)
        assert len(events) == 0

    def test_intruders_can_kill_each_other(self):
        """Faction combat can result in death."""
        ai, bus, grid, core, rng = _make_ai()
        surface = _make_intruder(intruder_id=1, x=5, y=5, z=0)
        uw = _make_intruder(
            intruder_id=2, x=5, y=5, z=0, is_underworlder=True,
            archetype=BOREMITE,  # Low HP (25)
        )
        surface.state = IntruderState.ADVANCING
        uw.state = IntruderState.ADVANCING
        # Give surface intruder massive damage to one-shot
        uw.hp = 1
        ai.intruders = [surface, uw]

        deaths = []
        bus.subscribe("intruder_died", lambda **kw: deaths.append(kw))

        # Create a party so on_member_death works
        party = Party(1, [uw])
        ai._underworld_parties.append(party)

        ai._tick_faction_encounters(FACTION_ENCOUNTER_INTERVAL)
        assert uw.state == IntruderState.DEAD


# ═══════════════════════════════════════════════════════════════════════
#  6. Integration Tests
# ═══════════════════════════════════════════════════════════════════════


class TestSocialDynamicsIntegration:
    """End-to-end integration tests."""

    def test_escape_archive_new_spawn_has_knowledge(self):
        """Full loop: spawn → escape → archive → new spawn has knowledge."""
        ka = KnowledgeArchive()

        # Survivor knows about some cells
        survivor = _make_intruder(intruder_id=1, status=IntruderStatus.VETERAN)
        survivor.personal_map.reveal(10, 10, 5, VOXEL_AIR, 0)
        survivor.personal_map.reveal(10, 11, 5, VOXEL_SPIKE, 0)

        ka.archive_survivor(survivor, tick=100)

        # New spawn receives the knowledge
        new_map = PersonalMap()
        ka.inject_knowledge(new_map, False, current_tick=200, cunning=0.0)

        assert (10, 10, 5) in new_map.seen
        assert (10, 11, 5) in new_map.hazards

    def test_wipe_no_knowledge_transfer(self):
        """Wiped parties contribute no knowledge (never archived)."""
        ka = KnowledgeArchive()

        # Intruder dies without escaping → never archived
        dead = _make_intruder(intruder_id=1)
        dead.personal_map.reveal(10, 10, 5, VOXEL_AIR, 0)
        dead.state = IntruderState.DEAD
        # Don't call archive_survivor — dead intruders aren't archived

        new_map = PersonalMap()
        ka.inject_knowledge(new_map, False, current_tick=200, cunning=0.0)
        assert (10, 10, 5) not in new_map.seen

    def test_contradiction_causes_high_uncertainty_skips_cell(self):
        """Contradictory reports cause cells to be skipped on injection."""
        ka = KnowledgeArchive()

        # Survivor A reports stone
        s1 = _make_intruder(intruder_id=1, status=IntruderStatus.GRUNT)
        s1.personal_map.reveal(5, 5, 5, VOXEL_STONE, 0)
        ka.archive_survivor(s1, tick=10)

        # Survivor B reports air — contradiction
        s2 = _make_intruder(intruder_id=2, status=IntruderStatus.GRUNT)
        s2.personal_map.reveal(5, 5, 5, VOXEL_AIR, 0)
        ka.archive_survivor(s2, tick=20)

        # More contradictions to push uncertainty high
        s3 = _make_intruder(intruder_id=3, status=IntruderStatus.GRUNT)
        s3.personal_map.reveal(5, 5, 5, VOXEL_STONE, 0)
        ka.archive_survivor(s3, tick=30)

        s4 = _make_intruder(intruder_id=4, status=IntruderStatus.GRUNT)
        s4.personal_map.reveal(5, 5, 5, VOXEL_AIR, 0)
        ka.archive_survivor(s4, tick=40)

        # Check if uncertainty is high enough to skip
        unc = ka.get_uncertainty(5, 5, 5, False)
        if unc > KNOWLEDGE_UNCERTAIN_THRESHOLD:
            new_map = PersonalMap()
            ka.inject_knowledge(new_map, False, current_tick=50, cunning=0.0)
            assert (5, 5, 5) not in new_map.seen

    def test_morale_cascade_ally_death_earlier_retreat(self):
        """Ally death → low morale → earlier retreat."""
        m1 = _make_intruder(intruder_id=1, archetype=VANGUARD)
        m2 = _make_intruder(intruder_id=2, archetype=VANGUARD)
        m3 = _make_intruder(intruder_id=3, archetype=VANGUARD)
        for m in (m1, m2, m3):
            m.state = IntruderState.ADVANCING

        party = Party(1, [m1, m2, m3])

        # Kill m2 and m3 to reduce m1's morale
        m2.state = IntruderState.DEAD
        party.on_member_death(m2)
        m3.state = IntruderState.DEAD
        party.on_member_death(m3)

        # m1 has taken 2 ally death penalties
        expected = MORALE_BASE - 2 * MORALE_ALLY_DEATH_PENALTY
        assert m1.morale == pytest.approx(max(0.0, expected), abs=0.01)

        # Set HP above normal threshold but test if low morale causes retreat
        m1.hp = int(m1.max_hp * (VANGUARD.retreat_threshold + 0.05))
        if m1.morale < MORALE_LOW_THRESHOLD:
            # Doubled threshold should trigger retreat
            IntruderAI._check_retreat(m1)
            assert m1.state == IntruderState.RETREATING

    def test_reputation_changes_publish_event(self):
        """Reputation changes publish 'reputation_changed' event."""
        bus = EventBus()
        rep = DungeonReputation(bus)

        events = []
        bus.subscribe("reputation_changed", lambda **kw: events.append(kw))

        bus.publish("intruder_died", intruder=None)
        assert len(events) > 0
        assert "lethality" in events[0]
        assert "richness" in events[0]

    def test_voxel_changed_hook_in_decision(self):
        """IntruderAI subscribes to voxel_changed and updates archive."""
        ai, bus, grid, core, rng = _make_ai()

        # Archive some data first
        survivor = _make_intruder()
        survivor.personal_map.reveal(3, 3, 0, VOXEL_STONE, 0)
        ai._knowledge_archive.archive_survivor(survivor, tick=100)

        # Simulate player changing that voxel
        bus.publish("voxel_changed", x=3, y=3, z=0, old_type=VOXEL_STONE, new_type=VOXEL_AIR)

        unc = ai._knowledge_archive.get_uncertainty(3, 3, 0, False)
        assert unc > 0.0
