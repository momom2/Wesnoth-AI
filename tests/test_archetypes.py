"""Tests for intruder archetype definitions and agent model."""

import pytest

from dungeon_builder.intruders.archetypes import (
    ArchetypeStats,
    IntruderObjective,
    ALL_ARCHETYPES,
    ARCHETYPE_BY_NAME,
    VANGUARD,
    SHADOWBLADE,
    TUNNELER,
    PYREMANCER,
    WINDCALLER,
    WARDEN,
    GORECLAW,
    GLOOMSEER,
)
from dungeon_builder.intruders.agent import Intruder, IntruderState
from dungeon_builder.intruders.personal_map import PersonalMap


# ── Archetype definition tests ──────────────────────────────────────


class TestArchetypeStats:
    """Verify all 8 archetypes are correctly defined."""

    def test_exactly_8_archetypes(self):
        assert len(ALL_ARCHETYPES) == 8

    def test_all_unique_names(self):
        names = [a.name for a in ALL_ARCHETYPES]
        assert len(names) == len(set(names))

    def test_archetype_by_name_lookup(self):
        for arch in ALL_ARCHETYPES:
            assert ARCHETYPE_BY_NAME[arch.name] is arch

    def test_archetypes_are_frozen(self):
        with pytest.raises(AttributeError):
            VANGUARD.hp = 999  # type: ignore[misc]

    @pytest.mark.parametrize("arch", ALL_ARCHETYPES, ids=lambda a: a.name)
    def test_hp_positive(self, arch):
        assert arch.hp > 0

    @pytest.mark.parametrize("arch", ALL_ARCHETYPES, ids=lambda a: a.name)
    def test_speed_in_range(self, arch):
        assert 1 <= arch.speed <= 4

    @pytest.mark.parametrize("arch", ALL_ARCHETYPES, ids=lambda a: a.name)
    def test_move_interval_positive(self, arch):
        assert arch.move_interval >= 1

    @pytest.mark.parametrize("arch", ALL_ARCHETYPES, ids=lambda a: a.name)
    def test_objective_weights_sum_positive(self, arch):
        total = sum(arch.objective_weights)
        assert total > 0

    @pytest.mark.parametrize("arch", ALL_ARCHETYPES, ids=lambda a: a.name)
    def test_greed_in_range(self, arch):
        assert 0.0 <= arch.greed <= 1.0

    @pytest.mark.parametrize("arch", ALL_ARCHETYPES, ids=lambda a: a.name)
    def test_loyalty_in_range(self, arch):
        assert 0.0 <= arch.loyalty <= 1.0

    @pytest.mark.parametrize("arch", ALL_ARCHETYPES, ids=lambda a: a.name)
    def test_retreat_threshold_in_range(self, arch):
        assert 0.0 <= arch.retreat_threshold <= 1.0


class TestArchetypeAbilities:
    """Verify unique archetype abilities are properly flagged."""

    def test_vanguard_bashes_doors(self):
        assert VANGUARD.can_bash_door is True
        assert VANGUARD.can_lockpick is False
        assert VANGUARD.can_fly is False
        assert VANGUARD.can_dig is False

    def test_shadowblade_lockpicks(self):
        assert SHADOWBLADE.can_lockpick is True
        assert SHADOWBLADE.can_bash_door is False
        assert SHADOWBLADE.spike_detect_range == 2

    def test_tunneler_digs(self):
        assert TUNNELER.can_dig is True
        assert TUNNELER.can_fly is False

    def test_pyremancer_fire_immune(self):
        assert PYREMANCER.fire_immune is True
        assert PYREMANCER.attack_range == 3

    def test_windcaller_flies(self):
        assert WINDCALLER.can_fly is True
        assert WINDCALLER.can_bash_door is False
        assert WINDCALLER.can_lockpick is False
        assert WINDCALLER.can_dig is False

    def test_warden_heals(self):
        assert WARDEN.healer is True
        assert WARDEN.damage == 2  # Low damage

    def test_goreclaw_frenzies(self):
        assert GORECLAW.frenzy_threshold == 0.5
        assert GORECLAW.never_retreats is True
        assert GORECLAW.can_bash_door is True

    def test_gloomseer_arcane_sight(self):
        assert GLOOMSEER.arcane_sight_range == 6
        assert GLOOMSEER.perception_range == 4

    def test_only_one_flyer(self):
        flyers = [a for a in ALL_ARCHETYPES if a.can_fly]
        assert len(flyers) == 1
        assert flyers[0] is WINDCALLER

    def test_only_one_digger(self):
        diggers = [a for a in ALL_ARCHETYPES if a.can_dig]
        assert len(diggers) == 1
        assert diggers[0] is TUNNELER

    def test_only_one_lockpicker(self):
        pickers = [a for a in ALL_ARCHETYPES if a.can_lockpick]
        assert len(pickers) == 1
        assert pickers[0] is SHADOWBLADE

    def test_only_one_fire_immune(self):
        immune = [a for a in ALL_ARCHETYPES if a.fire_immune]
        assert len(immune) == 1
        assert immune[0] is PYREMANCER


# ── IntruderObjective tests ─────────────────────────────────────────


class TestIntruderObjective:
    def test_three_objectives(self):
        assert len(IntruderObjective) == 3

    def test_objective_names(self):
        names = {o.name for o in IntruderObjective}
        assert names == {"DESTROY_CORE", "EXPLORE", "PILLAGE"}


# ── Intruder agent model tests ──────────────────────────────────────


class TestIntruderAgent:
    """Tests for the rewritten Intruder class."""

    def test_construction_from_archetype(self):
        i = Intruder(
            intruder_id=1, x=10, y=20, z=3,
            archetype=VANGUARD,
            objective=IntruderObjective.DESTROY_CORE,
            personal_map=PersonalMap(),
        )
        assert i.id == 1
        assert i.pos == (10, 20, 3)
        assert i.hp == 120  # VANGUARD HP
        assert i.max_hp == 120
        assert i.archetype is VANGUARD
        assert i.objective == IntruderObjective.DESTROY_CORE
        assert i.state == IntruderState.SPAWNING

    def test_different_archetypes_different_hp(self):
        v = Intruder(1, 0, 0, 0, VANGUARD, IntruderObjective.DESTROY_CORE, PersonalMap())
        s = Intruder(2, 0, 0, 0, SHADOWBLADE, IntruderObjective.PILLAGE, PersonalMap())
        assert v.hp == 120
        assert s.hp == 40

    def test_move_interval_from_archetype(self):
        v = Intruder(1, 0, 0, 0, VANGUARD, IntruderObjective.DESTROY_CORE, PersonalMap())
        w = Intruder(2, 0, 0, 0, WINDCALLER, IntruderObjective.EXPLORE, PersonalMap())
        assert v.move_interval == 10  # VANGUARD is slow
        assert w.move_interval == 2   # WINDCALLER is fast

    def test_party_id_default_none(self):
        i = Intruder(1, 0, 0, 0, VANGUARD, IntruderObjective.DESTROY_CORE, PersonalMap())
        assert i.party_id is None

    def test_party_id_set(self):
        i = Intruder(1, 0, 0, 0, VANGUARD, IntruderObjective.DESTROY_CORE,
                     PersonalMap(), party_id=42)
        assert i.party_id == 42

    def test_take_damage(self):
        i = Intruder(1, 0, 0, 0, VANGUARD, IntruderObjective.DESTROY_CORE, PersonalMap())
        i.state = IntruderState.ADVANCING
        i.take_damage(30)
        assert i.hp == 90
        assert i.state == IntruderState.ADVANCING

    def test_take_damage_kills(self):
        i = Intruder(1, 0, 0, 0, SHADOWBLADE, IntruderObjective.PILLAGE, PersonalMap())
        i.state = IntruderState.ADVANCING
        i.take_damage(999)
        assert i.hp == 0
        assert i.state == IntruderState.DEAD
        assert not i.alive

    def test_alive_property(self):
        i = Intruder(1, 0, 0, 0, VANGUARD, IntruderObjective.DESTROY_CORE, PersonalMap())
        assert i.alive is True
        i.state = IntruderState.DEAD
        assert i.alive is False
        i.state = IntruderState.ESCAPED
        assert i.alive is False

    def test_effective_loyalty_base(self):
        i = Intruder(1, 0, 0, 0, VANGUARD, IntruderObjective.DESTROY_CORE, PersonalMap())
        assert i.effective_loyalty == pytest.approx(0.9)

    def test_effective_loyalty_with_modifier(self):
        i = Intruder(1, 0, 0, 0, SHADOWBLADE, IntruderObjective.PILLAGE, PersonalMap())
        i.loyalty_modifier = 0.2
        assert i.effective_loyalty == pytest.approx(0.4)  # 0.2 + 0.2

    def test_effective_loyalty_clamped(self):
        i = Intruder(1, 0, 0, 0, WARDEN, IntruderObjective.DESTROY_CORE, PersonalMap())
        i.loyalty_modifier = 0.5
        # Warden loyalty=1.0, + 0.5 = 1.5, clamped to 1.0
        assert i.effective_loyalty == 1.0

    def test_frenzy_speed(self):
        i = Intruder(1, 0, 0, 0, GORECLAW, IntruderObjective.DESTROY_CORE, PersonalMap())
        assert i.effective_speed == 2
        i.frenzy_active = True
        assert i.effective_speed == 4

    def test_frenzy_damage(self):
        i = Intruder(1, 0, 0, 0, GORECLAW, IntruderObjective.DESTROY_CORE, PersonalMap())
        assert i.effective_damage == 15
        i.frenzy_active = True
        assert i.effective_damage == 22  # int(15 * 1.5) = 22

    def test_frenzy_move_interval(self):
        i = Intruder(1, 0, 0, 0, GORECLAW, IntruderObjective.DESTROY_CORE, PersonalMap())
        assert i.effective_move_interval == 5
        i.frenzy_active = True
        assert i.effective_move_interval == 2  # max(1, 5 // 2)

    def test_loot_starts_zero(self):
        i = Intruder(1, 0, 0, 0, SHADOWBLADE, IntruderObjective.PILLAGE, PersonalMap())
        assert i.loot_count == 0

    def test_dig_progress_starts_empty(self):
        i = Intruder(1, 0, 0, 0, TUNNELER, IntruderObjective.DESTROY_CORE, PersonalMap())
        assert i.dig_progress == {}

    def test_personal_map_attached(self):
        pm = PersonalMap()
        i = Intruder(1, 0, 0, 0, VANGUARD, IntruderObjective.DESTROY_CORE, pm)
        assert i.personal_map is pm

    def test_repr_includes_archetype(self):
        i = Intruder(1, 5, 10, 3, SHADOWBLADE, IntruderObjective.PILLAGE, PersonalMap())
        r = repr(i)
        assert "Shadowblade" in r
        assert "5,10,3" in r


# ── IntruderState tests ─────────────────────────────────────────────


class TestIntruderState:
    def test_has_all_states(self):
        expected = {
            "SPAWNING", "ADVANCING", "INTERACTING", "ATTACKING",
            "RETREATING", "PILLAGING", "DEAD", "ESCAPED",
        }
        actual = {s.name for s in IntruderState}
        assert actual == expected

    def test_new_states_exist(self):
        """INTERACTING and PILLAGING are new in the archetype rewrite."""
        assert IntruderState.INTERACTING
        assert IntruderState.PILLAGING
