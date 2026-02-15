"""Tests for per-archetype intruder rendering colors and config.

These tests verify the color mapping logic without requiring Panda3D.
"""

import pytest

from dungeon_builder.config import (
    ARCHETYPE_COLORS,
    ARCHETYPE_DEFAULT_COLOR,
    ARCHETYPE_FRENZY_COLOR,
)
from dungeon_builder.intruders.archetypes import (
    ALL_ARCHETYPES,
    VANGUARD,
    SHADOWBLADE,
    TUNNELER,
    PYREMANCER,
    WINDCALLER,
    WARDEN,
    GORECLAW,
    GLOOMSEER,
)
from dungeon_builder.intruders.agent import Intruder
from dungeon_builder.intruders.personal_map import PersonalMap
from dungeon_builder.rendering.intruder_renderer import _archetype_color


def _make(arch):
    from dungeon_builder.intruders.archetypes import IntruderObjective
    return Intruder(1, 0, 0, 0, arch, IntruderObjective.DESTROY_CORE, PersonalMap())


# ── Color mapping per archetype ──────────────────────────────────────


class TestArchetypeColors:
    @pytest.mark.parametrize("arch", ALL_ARCHETYPES, ids=lambda a: a.name)
    def test_every_archetype_has_a_color(self, arch):
        """Each archetype must have an entry in ARCHETYPE_COLORS."""
        assert arch.name in ARCHETYPE_COLORS

    @pytest.mark.parametrize("arch", ALL_ARCHETYPES, ids=lambda a: a.name)
    def test_colors_are_valid_rgb(self, arch):
        r, g, b = ARCHETYPE_COLORS[arch.name]
        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0

    def test_all_colors_are_distinct(self):
        """No two archetypes should share the same color."""
        colors = list(ARCHETYPE_COLORS.values())
        assert len(colors) == len(set(colors))

    def test_default_color_is_valid_rgb(self):
        r, g, b = ARCHETYPE_DEFAULT_COLOR
        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0

    def test_frenzy_color_is_valid_rgb(self):
        r, g, b = ARCHETYPE_FRENZY_COLOR
        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0


# ── _archetype_color helper ──────────────────────────────────────────


class TestArchetypeColorHelper:
    def test_vanguard_gets_steel_blue(self):
        color = _archetype_color(_make(VANGUARD))
        assert color == ARCHETYPE_COLORS["Vanguard"]

    def test_shadowblade_gets_purple(self):
        color = _archetype_color(_make(SHADOWBLADE))
        assert color == ARCHETYPE_COLORS["Shadowblade"]

    def test_tunneler_gets_brown(self):
        color = _archetype_color(_make(TUNNELER))
        assert color == ARCHETYPE_COLORS["Tunneler"]

    def test_pyremancer_gets_orange(self):
        color = _archetype_color(_make(PYREMANCER))
        assert color == ARCHETYPE_COLORS["Pyremancer"]

    def test_windcaller_gets_cyan(self):
        color = _archetype_color(_make(WINDCALLER))
        assert color == ARCHETYPE_COLORS["Windcaller"]

    def test_warden_gets_gold(self):
        color = _archetype_color(_make(WARDEN))
        assert color == ARCHETYPE_COLORS["Warden"]

    def test_goreclaw_gets_red(self):
        color = _archetype_color(_make(GORECLAW))
        assert color == ARCHETYPE_COLORS["Goreclaw"]

    def test_gloomseer_gets_indigo(self):
        color = _archetype_color(_make(GLOOMSEER))
        assert color == ARCHETYPE_COLORS["Gloomseer"]

    @pytest.mark.parametrize("arch", ALL_ARCHETYPES, ids=lambda a: a.name)
    def test_color_matches_config(self, arch):
        """_archetype_color returns the same color as ARCHETYPE_COLORS."""
        color = _archetype_color(_make(arch))
        assert color == ARCHETYPE_COLORS[arch.name]


# ── Color distinctness ────────────────────────────────────────────────


class TestColorDistinctness:
    def test_frenzy_color_differs_from_all_archetypes(self):
        """Frenzy color must be visually distinguishable from normal colors."""
        for name, color in ARCHETYPE_COLORS.items():
            assert color != ARCHETYPE_FRENZY_COLOR, (
                f"Frenzy color matches {name}'s normal color"
            )

    def test_goreclaw_normal_differs_from_frenzy(self):
        """Goreclaw's base color should differ from frenzy color."""
        assert ARCHETYPE_COLORS["Goreclaw"] != ARCHETYPE_FRENZY_COLOR
