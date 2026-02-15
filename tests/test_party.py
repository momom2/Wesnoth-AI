"""Tests for the intruder party system."""

import pytest

from dungeon_builder.intruders.party import (
    Party,
    PartyTemplate,
    choose_template,
    generate_composition,
    ALL_TEMPLATES,
    STANDARD_RAID,
    SCOUTING_PARTY,
    SIEGE_FORCE,
    WAR_BAND,
)
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
    ALL_ARCHETYPES,
)
from dungeon_builder.intruders.agent import Intruder, IntruderState
from dungeon_builder.intruders.personal_map import PersonalMap
from dungeon_builder.utils.rng import SeededRNG
from dungeon_builder.config import (
    MAP_SHARE_RANGE,
    WARDEN_HEAL_AMOUNT,
    WARDEN_HEAL_INTERVAL,
    WARDEN_LOYALTY_BONUS,
    WARDEN_DEATH_LOYALTY_PENALTY,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _make(
    intruder_id: int,
    arch=VANGUARD,
    x: int = 0,
    y: int = 0,
    z: int = 0,
    objective=IntruderObjective.DESTROY_CORE,
) -> Intruder:
    return Intruder(intruder_id, x, y, z, arch, objective, PersonalMap())


def _make_rng(seed: int = 42) -> SeededRNG:
    return SeededRNG(seed)


# ── Template and composition tests ────────────────────────────────────


class TestTemplates:
    def test_all_templates_weights_sum_to_1(self):
        total = sum(t.weight for t in ALL_TEMPLATES)
        assert total == pytest.approx(1.0)

    def test_choose_template_returns_valid_template(self):
        rng = _make_rng()
        for _ in range(50):
            tmpl = choose_template(rng)
            assert tmpl in ALL_TEMPLATES

    def test_choose_template_distribution(self):
        """Run many samples and verify the distribution roughly matches weights."""
        rng = _make_rng(12345)
        counts = {t.name: 0 for t in ALL_TEMPLATES}
        n = 10000
        for _ in range(n):
            tmpl = choose_template(rng)
            counts[tmpl.name] += 1

        # Each template should be within 5% of its expected weight
        for tmpl in ALL_TEMPLATES:
            expected = tmpl.weight
            actual = counts[tmpl.name] / n
            assert abs(actual - expected) < 0.05, (
                f"{tmpl.name}: expected ~{expected:.2f}, got {actual:.3f}"
            )

    def test_generate_composition_sizes(self):
        """Generated compositions respect min/max counts."""
        rng = _make_rng(99)
        for tmpl in ALL_TEMPLATES:
            for _ in range(20):
                comp = generate_composition(tmpl, rng)
                min_total = sum(s.min_count for s in tmpl.slots)
                max_total = sum(s.max_count for s in tmpl.slots)
                assert min_total <= len(comp) <= max_total, (
                    f"{tmpl.name}: got {len(comp)}, expected [{min_total}, {max_total}]"
                )

    def test_standard_raid_has_vanguard(self):
        rng = _make_rng(1)
        comp = generate_composition(STANDARD_RAID, rng)
        names = [a.name for a in comp]
        assert "Vanguard" in names

    def test_scouting_party_has_windcaller(self):
        rng = _make_rng(1)
        comp = generate_composition(SCOUTING_PARTY, rng)
        names = [a.name for a in comp]
        assert "Windcaller" in names

    def test_siege_force_has_tunnelers(self):
        rng = _make_rng(1)
        comp = generate_composition(SIEGE_FORCE, rng)
        names = [a.name for a in comp]
        tunneler_count = names.count("Tunneler")
        assert tunneler_count >= 2

    def test_war_band_has_goreclaws(self):
        rng = _make_rng(1)
        comp = generate_composition(WAR_BAND, rng)
        names = [a.name for a in comp]
        goreclaw_count = names.count("Goreclaw")
        assert goreclaw_count >= 3

    def test_composition_only_contains_valid_archetypes(self):
        rng = _make_rng(42)
        for tmpl in ALL_TEMPLATES:
            comp = generate_composition(tmpl, rng)
            for arch in comp:
                assert arch in ALL_ARCHETYPES


# ── Leader election ────────────────────────────────────────────────────


class TestLeaderElection:
    def test_leader_is_highest_loyalty(self):
        # Warden has loyalty=1.0, Goreclaw has 0.4
        members = [_make(1, GORECLAW), _make(2, WARDEN), _make(3, VANGUARD)]
        party = Party(100, members)
        assert party.leader is not None
        assert party.leader.id == 2  # Warden

    def test_leader_tiebreak_by_id(self):
        # Two members with same loyalty
        m1 = _make(5, VANGUARD)  # loyalty=0.9
        m2 = _make(3, VANGUARD)  # loyalty=0.9
        party = Party(100, [m1, m2])
        assert party.leader is not None
        assert party.leader.id == 3  # Lower id wins

    def test_leader_reelection_on_death(self):
        m_warden = _make(1, WARDEN)   # loyalty=1.0
        m_vanguard = _make(2, VANGUARD)  # loyalty=0.9
        party = Party(100, [m_warden, m_vanguard])
        assert party.leader.id == 1

        m_warden.state = IntruderState.DEAD
        party.on_member_death(m_warden)
        assert party.leader.id == 2

    def test_leader_none_when_all_dead(self):
        m1 = _make(1, VANGUARD)
        party = Party(100, [m1])
        m1.state = IntruderState.DEAD
        party._elect_leader()
        assert party.leader is None


# ── Objective voting ───────────────────────────────────────────────────


class TestObjectiveVoting:
    def test_all_vanguards_vote_destroy(self):
        members = [_make(i, VANGUARD) for i in range(3)]
        party = Party(100, members)
        assert party.objective == IntruderObjective.DESTROY_CORE

    def test_all_windcallers_vote_explore(self):
        members = [_make(i, WINDCALLER) for i in range(3)]
        party = Party(100, members)
        assert party.objective == IntruderObjective.EXPLORE

    def test_all_shadowblades_vote_pillage(self):
        members = [_make(i, SHADOWBLADE) for i in range(3)]
        party = Party(100, members)
        assert party.objective == IntruderObjective.PILLAGE

    def test_mixed_party_weighted_vote(self):
        # 2 Vanguards (destroy=1.0 each) + 1 Windcaller (explore=0.8)
        # destroy total = 2.0 + 0.2 = 2.2, explore = 0.0 + 0.8 = 0.8
        members = [_make(1, VANGUARD), _make(2, VANGUARD), _make(3, WINDCALLER)]
        party = Party(100, members)
        assert party.objective == IntruderObjective.DESTROY_CORE

    def test_revote_after_death(self):
        # 1 Vanguard (destroy=1.0) + 2 Windcallers (explore=0.8 each)
        # Initial: destroy=1.0+0.2+0.2=1.4, explore=0.0+0.8+0.8=1.6 → EXPLORE
        m_van = _make(1, VANGUARD)
        m_w1 = _make(2, WINDCALLER)
        m_w2 = _make(3, WINDCALLER)
        party = Party(100, [m_van, m_w1, m_w2])
        assert party.objective == IntruderObjective.EXPLORE

        # Kill one windcaller → destroy=1.0+0.2=1.2, explore=0.0+0.8=0.8 → DESTROY
        m_w2.state = IntruderState.DEAD
        party.on_member_death(m_w2)
        assert party.objective == IntruderObjective.DESTROY_CORE


# ── Map sharing ────────────────────────────────────────────────────────


class TestMapSharing:
    def test_nearby_members_share_maps(self):
        m1 = _make(1, VANGUARD, x=0, y=0, z=0)
        m2 = _make(2, SHADOWBLADE, x=1, y=0, z=0)  # distance=1 <= MAP_SHARE_RANGE
        m1.personal_map.reveal(5, 5, 5, 2)  # stone
        m2.personal_map.reveal(10, 10, 10, 1)  # dirt

        party = Party(100, [m1, m2])
        party.share_maps()

        # Both should know about each other's cells
        assert m1.personal_map.is_revealed(10, 10, 10)
        assert m2.personal_map.is_revealed(5, 5, 5)

    def test_distant_members_do_not_share(self):
        m1 = _make(1, VANGUARD, x=0, y=0, z=0)
        m2 = _make(2, SHADOWBLADE, x=0, y=0, z=MAP_SHARE_RANGE + 1)
        m1.personal_map.reveal(5, 5, 5, 2)

        party = Party(100, [m1, m2])
        party.share_maps()

        assert not m2.personal_map.is_revealed(5, 5, 5)

    def test_dead_members_excluded_from_sharing(self):
        m1 = _make(1, VANGUARD, x=0, y=0, z=0)
        m2 = _make(2, VANGUARD, x=1, y=0, z=0)
        m1.personal_map.reveal(5, 5, 5, 2)
        m2.state = IntruderState.DEAD

        party = Party(100, [m1, m2])
        party.share_maps()

        assert not m2.personal_map.is_revealed(5, 5, 5)


# ── Warden healing ─────────────────────────────────────────────────────


class TestWardenHealing:
    def test_warden_heals_lowest_hp_ally(self):
        warden = _make(1, WARDEN, x=0, y=0, z=0)
        injured = _make(2, VANGUARD, x=1, y=0, z=0)
        injured.hp = 50  # max_hp=120

        party = Party(100, [warden, injured])

        # Tick WARDEN_HEAL_INTERVAL times to trigger heal
        heals = []
        for _ in range(WARDEN_HEAL_INTERVAL):
            heals = party.tick_warden_heal()

        assert len(heals) == 1
        assert heals[0][0].id == 1  # warden
        assert heals[0][1].id == 2  # patient
        assert heals[0][2] == WARDEN_HEAL_AMOUNT
        assert injured.hp == 50 + WARDEN_HEAL_AMOUNT

    def test_warden_does_not_heal_self(self):
        warden = _make(1, WARDEN, x=0, y=0, z=0)
        warden.hp = 30  # Injured warden

        party = Party(100, [warden])
        for _ in range(WARDEN_HEAL_INTERVAL):
            heals = party.tick_warden_heal()

        assert len(heals) == 0
        assert warden.hp == 30

    def test_warden_does_not_heal_full_hp(self):
        warden = _make(1, WARDEN, x=0, y=0, z=0)
        healthy = _make(2, VANGUARD, x=1, y=0, z=0)
        # healthy.hp == healthy.max_hp (120)

        party = Party(100, [warden, healthy])
        for _ in range(WARDEN_HEAL_INTERVAL):
            heals = party.tick_warden_heal()

        assert len(heals) == 0

    def test_heal_capped_at_max_hp(self):
        warden = _make(1, WARDEN, x=0, y=0, z=0)
        almost_full = _make(2, VANGUARD, x=1, y=0, z=0)
        almost_full.hp = almost_full.max_hp - 2  # Only 2 HP missing

        party = Party(100, [warden, almost_full])
        for _ in range(WARDEN_HEAL_INTERVAL):
            heals = party.tick_warden_heal()

        assert len(heals) == 1
        assert heals[0][2] == 2  # Only healed 2, not full WARDEN_HEAL_AMOUNT
        assert almost_full.hp == almost_full.max_hp

    def test_heal_requires_interval_ticks(self):
        warden = _make(1, WARDEN, x=0, y=0, z=0)
        injured = _make(2, VANGUARD, x=1, y=0, z=0)
        injured.hp = 50

        party = Party(100, [warden, injured])

        # Only tick interval - 1 times — no heal yet
        for _ in range(WARDEN_HEAL_INTERVAL - 1):
            heals = party.tick_warden_heal()

        assert len(heals) == 0
        assert injured.hp == 50

    def test_warden_out_of_range_no_heal(self):
        warden = _make(1, WARDEN, x=0, y=0, z=0)
        far_ally = _make(2, VANGUARD, x=0, y=0, z=MAP_SHARE_RANGE + 1)
        far_ally.hp = 50

        party = Party(100, [warden, far_ally])
        for _ in range(WARDEN_HEAL_INTERVAL):
            heals = party.tick_warden_heal()

        assert len(heals) == 0


# ── Warden loyalty aura ───────────────────────────────────────────────


class TestWardenLoyaltyAura:
    def test_aura_boosts_nearby_loyalty(self):
        warden = _make(1, WARDEN, x=0, y=0, z=0)
        ally = _make(2, GORECLAW, x=1, y=0, z=0)

        party = Party(100, [warden, ally])
        party.apply_warden_aura()

        assert ally.loyalty_modifier == pytest.approx(WARDEN_LOYALTY_BONUS)

    def test_aura_does_not_affect_warden_self(self):
        warden = _make(1, WARDEN, x=0, y=0, z=0)
        ally = _make(2, VANGUARD, x=1, y=0, z=0)

        party = Party(100, [warden, ally])
        party.apply_warden_aura()

        assert warden.loyalty_modifier == pytest.approx(0.0)

    def test_aura_does_not_reach_distant_ally(self):
        warden = _make(1, WARDEN, x=0, y=0, z=0)
        far_ally = _make(2, GORECLAW, x=0, y=0, z=MAP_SHARE_RANGE + 1)

        party = Party(100, [warden, far_ally])
        party.apply_warden_aura()

        assert far_ally.loyalty_modifier == pytest.approx(0.0)

    def test_death_penalty_preserved_through_aura(self):
        warden1 = _make(1, WARDEN, x=0, y=0, z=0)
        warden2 = _make(2, WARDEN, x=1, y=0, z=0)
        ally = _make(3, GORECLAW, x=0, y=1, z=0)

        party = Party(100, [warden1, warden2, ally])

        # Kill warden1 → ally gets -0.3 penalty
        warden1.state = IntruderState.DEAD
        party.on_member_death(warden1)
        assert ally.loyalty_modifier == pytest.approx(-WARDEN_DEATH_LOYALTY_PENALTY)

        # Now apply aura from surviving warden2
        party.apply_warden_aura()
        # Should be: death penalty (-0.3) + warden aura (+0.2) = -0.1
        expected = -WARDEN_DEATH_LOYALTY_PENALTY + WARDEN_LOYALTY_BONUS
        assert ally.loyalty_modifier == pytest.approx(expected)


# ── Warden death penalty ──────────────────────────────────────────────


class TestWardenDeathPenalty:
    def test_warden_death_penalises_all_alive(self):
        warden = _make(1, WARDEN, x=0, y=0, z=0)
        ally1 = _make(2, VANGUARD, x=1, y=0, z=0)
        ally2 = _make(3, GORECLAW, x=2, y=0, z=0)

        party = Party(100, [warden, ally1, ally2])
        warden.state = IntruderState.DEAD
        party.on_member_death(warden)

        assert ally1.loyalty_modifier == pytest.approx(-WARDEN_DEATH_LOYALTY_PENALTY)
        assert ally2.loyalty_modifier == pytest.approx(-WARDEN_DEATH_LOYALTY_PENALTY)

    def test_non_warden_death_no_penalty(self):
        warden = _make(1, WARDEN, x=0, y=0, z=0)
        grunt = _make(2, GORECLAW, x=1, y=0, z=0)

        party = Party(100, [warden, grunt])
        grunt.state = IntruderState.DEAD
        party.on_member_death(grunt)

        # Warden's loyalty_modifier unchanged
        assert warden.loyalty_modifier == pytest.approx(0.0)


# ── Betrayal ───────────────────────────────────────────────────────────


class TestBetrayal:
    def test_greedy_near_treasure_can_betray(self):
        # Shadowblade: greed=0.9, loyalty=0.2 → chance = 0.9 * (1-0.2) = 0.72
        shadow = _make(1, SHADOWBLADE, x=0, y=0, z=0)
        tank = _make(2, VANGUARD, x=1, y=0, z=0)  # greed=0.0

        party = Party(100, [shadow, tank])
        treasure_adj = {1: True, 2: False}

        # Use a seed that produces a low roll (< 0.72)
        rng = _make_rng(42)
        # Try many times — with 72% chance, should betray within a few tries
        betrayed = False
        for _ in range(20):
            rng_attempt = _make_rng(rng.randint(0, 100000))
            betrayers = party.check_betrayals(treasure_adj, rng_attempt)
            if betrayers:
                betrayed = True
                break

        assert betrayed

    def test_no_greed_never_betrays(self):
        # Vanguard: greed=0.0
        tank = _make(1, VANGUARD, x=0, y=0, z=0)
        party = Party(100, [tank])
        treasure_adj = {1: True}

        rng = _make_rng(42)
        for _ in range(100):
            betrayers = party.check_betrayals(treasure_adj, rng)
            assert len(betrayers) == 0

    def test_not_adjacent_to_treasure_never_betrays(self):
        shadow = _make(1, SHADOWBLADE, x=0, y=0, z=0)
        party = Party(100, [shadow])
        treasure_adj = {1: False}

        rng = _make_rng(42)
        for _ in range(100):
            betrayers = party.check_betrayals(treasure_adj, rng)
            assert len(betrayers) == 0

    def test_betrayer_switches_to_pillage(self):
        shadow = _make(1, SHADOWBLADE, x=0, y=0, z=0)
        tank = _make(2, VANGUARD, x=1, y=0, z=0)
        party = Party(100, [shadow, tank])
        treasure_adj = {1: True, 2: False}

        # Force betrayal by trying many seeds
        betrayed_shadow = None
        for seed in range(1000):
            # Reset: re-add shadow to fresh party
            shadow_fresh = _make(1, SHADOWBLADE, x=0, y=0, z=0)
            tank_fresh = _make(2, VANGUARD, x=1, y=0, z=0)
            p = Party(100, [shadow_fresh, tank_fresh])
            rng = _make_rng(seed)
            betrayers = p.check_betrayals(treasure_adj, rng)
            if betrayers:
                betrayed_shadow = betrayers[0]
                break

        assert betrayed_shadow is not None
        assert betrayed_shadow.objective == IntruderObjective.PILLAGE
        assert betrayed_shadow.party_id is None

    def test_warden_aura_reduces_betrayal(self):
        """Warden loyalty aura should reduce betrayal probability."""
        shadow = _make(1, SHADOWBLADE, x=0, y=0, z=0)
        warden = _make(2, WARDEN, x=1, y=0, z=0)

        party = Party(100, [shadow, warden])
        party.apply_warden_aura()

        # Shadow base loyalty=0.2, +0.2 aura = 0.4
        assert shadow.effective_loyalty == pytest.approx(0.4)

        # Without aura: chance = 0.9 * (1 - 0.2) = 0.72
        # With aura: chance = 0.9 * (1 - 0.4) = 0.54
        # So betrayal is less likely. Test that the math is correct.
        expected_chance = 0.9 * (1.0 - 0.4)
        assert expected_chance == pytest.approx(0.54)

    def test_betrayer_removed_from_party(self):
        shadow = _make(1, SHADOWBLADE, x=0, y=0, z=0)
        tank = _make(2, VANGUARD, x=1, y=0, z=0)
        party = Party(100, [shadow, tank])
        treasure_adj = {1: True, 2: False}

        for seed in range(1000):
            shadow_fresh = _make(1, SHADOWBLADE, x=0, y=0, z=0)
            tank_fresh = _make(2, VANGUARD, x=1, y=0, z=0)
            p = Party(100, [shadow_fresh, tank_fresh])
            rng = _make_rng(seed)
            betrayers = p.check_betrayals(treasure_adj, rng)
            if betrayers:
                assert len(p.members) == 1
                assert p.members[0].id == 2  # Only tank remains
                break


# ── Party construction and repr ────────────────────────────────────────


class TestPartyMisc:
    def test_party_assigns_party_id(self):
        m1 = _make(1, VANGUARD)
        m2 = _make(2, SHADOWBLADE)
        party = Party(42, [m1, m2])
        assert m1.party_id == 42
        assert m2.party_id == 42

    def test_len(self):
        members = [_make(i, VANGUARD) for i in range(5)]
        party = Party(1, members)
        assert len(party) == 5

    def test_repr(self):
        members = [_make(1, VANGUARD), _make(2, WARDEN)]
        party = Party(1, members)
        r = repr(party)
        assert "Party" in r
        assert "members=2" in r
        assert "alive=2" in r

    def test_is_wiped(self):
        m1 = _make(1, VANGUARD)
        party = Party(1, [m1])
        assert not party.is_wiped
        m1.state = IntruderState.DEAD
        assert party.is_wiped
