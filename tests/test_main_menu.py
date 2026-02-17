"""Tests for the main menu system (no Panda3D window needed).

Covers:
- Menu constants registry validation
- Config modification via module references
- GameState.menu_open field
- MainMenu navigation logic (via source inspection)
- Event bus config_changed publishing
"""

from __future__ import annotations

import inspect

import pytest

import dungeon_builder.config as _cfg
from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.core.game_state import GameState
from dungeon_builder.ui.menu_constants import (
    DIFFICULTY_SETTINGS,
    VISIBILITY_SETTINGS,
    FOG_COLOR_SETTINGS,
    capture_defaults,
    capture_fog_defaults,
)


# ── Test: Menu Constants Registry ───────────────────────────────────────


class TestMenuConstants:
    """Validate the slider configuration registry in menu_constants.py."""

    def test_difficulty_attrs_exist_in_config(self):
        """Every difficulty attr must exist as a config module attribute."""
        for setting in DIFFICULTY_SETTINGS:
            assert hasattr(_cfg, setting["attr"]), (
                f"config.py missing attribute: {setting['attr']}"
            )

    def test_visibility_attrs_exist_in_config(self):
        """Every visibility attr must exist as a config module attribute."""
        for setting in VISIBILITY_SETTINGS:
            assert hasattr(_cfg, setting["attr"]), (
                f"config.py missing attribute: {setting['attr']}"
            )

    def test_min_less_than_max(self):
        """All settings must have min < max."""
        all_settings = DIFFICULTY_SETTINGS + VISIBILITY_SETTINGS + FOG_COLOR_SETTINGS
        for s in all_settings:
            label = s.get("label", s.get("attr", "?"))
            assert s["min"] < s["max"], f"{label}: min >= max"

    def test_step_positive(self):
        """All settings must have step > 0."""
        all_settings = DIFFICULTY_SETTINGS + VISIBILITY_SETTINGS + FOG_COLOR_SETTINGS
        for s in all_settings:
            label = s.get("label", s.get("attr", "?"))
            assert s["step"] > 0, f"{label}: step <= 0"

    def test_default_values_within_range(self):
        """Current config defaults must lie within [min, max]."""
        for s in DIFFICULTY_SETTINGS + VISIBILITY_SETTINGS:
            val = getattr(_cfg, s["attr"])
            assert s["min"] <= val <= s["max"], (
                f"{s['attr']}: default {val} outside [{s['min']}, {s['max']}]"
            )

    def test_fog_color_defaults_within_range(self):
        """FOG_COLOR components must lie within [min, max]."""
        fog = _cfg.FOG_COLOR
        for s in FOG_COLOR_SETTINGS:
            val = fog[s["index"]]
            assert s["min"] <= val <= s["max"], (
                f"FOG_COLOR[{s['index']}]: {val} outside [{s['min']}, {s['max']}]"
            )

    def test_fog_indices_cover_rgba(self):
        """FOG_COLOR_SETTINGS should have indices 0-3 (RGBA)."""
        indices = sorted(s["index"] for s in FOG_COLOR_SETTINGS)
        assert indices == [0, 1, 2, 3]

    def test_difficulty_has_expected_count(self):
        """Difficulty section should have 10 slider entries."""
        assert len(DIFFICULTY_SETTINGS) == 10

    def test_visibility_has_expected_count(self):
        """Visibility section should have 8 slider entries."""
        assert len(VISIBILITY_SETTINGS) == 8

    def test_setting_types_valid(self):
        """All settings with 'type' key must be int or float."""
        for s in DIFFICULTY_SETTINGS + VISIBILITY_SETTINGS:
            assert s["type"] in (int, float), (
                f"{s['attr']}: type must be int or float, got {s['type']}"
            )


# ── Test: capture_defaults ──────────────────────────────────────────────


class TestCaptureDefaults:
    """Test that capture_defaults / capture_fog_defaults snapshot correctly."""

    def test_capture_difficulty_defaults(self):
        """capture_defaults should return current values for all difficulty attrs."""
        defaults = capture_defaults(DIFFICULTY_SETTINGS)
        for s in DIFFICULTY_SETTINGS:
            assert s["attr"] in defaults
            assert defaults[s["attr"]] == getattr(_cfg, s["attr"])

    def test_capture_visibility_defaults(self):
        """capture_defaults should return current values for visibility attrs."""
        defaults = capture_defaults(VISIBILITY_SETTINGS)
        for s in VISIBILITY_SETTINGS:
            assert s["attr"] in defaults
            assert defaults[s["attr"]] == getattr(_cfg, s["attr"])

    def test_capture_fog_defaults_is_tuple(self):
        """capture_fog_defaults should return a tuple matching FOG_COLOR."""
        fog = capture_fog_defaults()
        assert isinstance(fog, tuple)
        assert fog == tuple(_cfg.FOG_COLOR)


# ── Test: Config Modification via Module Refs ───────────────────────────


class TestConfigModification:
    """Verify setattr(_cfg, ...) is visible through module references."""

    def test_setattr_changes_config_value(self):
        """Setting a config attr via setattr should be visible via _cfg.X."""
        original = _cfg.INTRUDER_DEFAULT_HP
        try:
            setattr(_cfg, "INTRUDER_DEFAULT_HP", 999)
            assert _cfg.INTRUDER_DEFAULT_HP == 999
        finally:
            setattr(_cfg, "INTRUDER_DEFAULT_HP", original)

    def test_fog_color_tuple_modification(self):
        """FOG_COLOR tuple can be replaced via setattr."""
        original = _cfg.FOG_COLOR
        try:
            new_fog = (0.1, 0.2, 0.3, 0.4)
            _cfg.FOG_COLOR = new_fog
            assert _cfg.FOG_COLOR == new_fog
        finally:
            _cfg.FOG_COLOR = original

    def test_module_ref_reflects_changes(self):
        """Reading _cfg.X after setattr should return the new value."""
        original = _cfg.CAMERA_PAN_SPEED
        try:
            _cfg.CAMERA_PAN_SPEED = 77.0
            # A separate read should see the change
            val = _cfg.CAMERA_PAN_SPEED
            assert val == 77.0
        finally:
            _cfg.CAMERA_PAN_SPEED = original

    def test_module_ref_not_copied_at_import_time(self):
        """Demonstrate that direct import copies don't see module ref changes."""
        # This test documents the *reason* we use module refs
        from dungeon_builder.config import INTRUDER_DEFAULT_HP as copied_val
        original = _cfg.INTRUDER_DEFAULT_HP
        try:
            _cfg.INTRUDER_DEFAULT_HP = 12345
            # The copied value should NOT have changed
            assert copied_val == original
            # But the module ref DOES see the change
            assert _cfg.INTRUDER_DEFAULT_HP == 12345
        finally:
            _cfg.INTRUDER_DEFAULT_HP = original


# ── Test: GameState.menu_open ───────────────────────────────────────────


class TestGameStateMenuOpen:
    """Verify the menu_open field on GameState."""

    def test_menu_open_defaults_true(self):
        """GameState should start with menu_open = True (main menu visible)."""
        gs = GameState(seed=42)
        assert gs.menu_open is True

    def test_menu_open_toggleable(self):
        """menu_open can be set to False and back."""
        gs = GameState(seed=42)
        gs.menu_open = False
        assert gs.menu_open is False
        gs.menu_open = True
        assert gs.menu_open is True


# ── Test: MainMenu Navigation Logic (source inspection) ────────────────


class TestMainMenuNavigation:
    """Verify MainMenu navigation structure via source inspection.

    Since MainMenu requires Panda3D (DirectGui widgets), we inspect the
    source code to verify the navigation state machine and frame builders.
    """

    @pytest.fixture(autouse=True)
    def _load_source(self):
        from dungeon_builder.ui.main_menu import MainMenu
        self.cls = MainMenu
        self.init_src = inspect.getsource(MainMenu.__init__)
        self.class_src = inspect.getsource(MainMenu)

    def test_states_defined(self):
        """All expected states should be listed in _STATES."""
        from dungeon_builder.ui.main_menu import _STATES
        expected = {"main", "options", "game_constants",
                    "difficulty", "visibility", "keybinding", "sound"}
        assert set(_STATES) == expected

    def test_build_methods_exist(self):
        """A _build_*_frame method should exist for each state."""
        for state in ("main", "options", "game_constants",
                      "difficulty", "visibility", "keybinding", "sound"):
            method_name = f"_build_{state}_frame"
            assert hasattr(self.cls, method_name), (
                f"Missing method: {method_name}"
            )

    def test_navigate_to_pushes_stack(self):
        """_navigate_to should append to _nav_stack."""
        src = inspect.getsource(self.cls._navigate_to)
        assert "_nav_stack.append" in src

    def test_navigate_back_pops_stack(self):
        """_navigate_back should pop from _nav_stack."""
        src = inspect.getsource(self.cls._navigate_back)
        assert "_nav_stack.pop" in src

    def test_escape_bound_in_init(self):
        """Escape key should be bound to toggle in __init__."""
        assert '"escape"' in self.init_src or "'escape'" in self.init_src
        assert "self.toggle" in self.init_src

    def test_play_sets_first_play_false(self):
        """_on_play should set _first_play = False."""
        src = inspect.getsource(self.cls._on_play)
        assert "_first_play = False" in src

    def test_show_sets_menu_open_true(self):
        """show() should set game_state.menu_open = True."""
        src = inspect.getsource(self.cls.show)
        assert "menu_open = True" in src

    def test_hide_sets_menu_open_false(self):
        """hide() should set game_state.menu_open = False."""
        src = inspect.getsource(self.cls.hide)
        assert "menu_open = False" in src

    def test_main_frame_has_play_options_quit(self):
        """Main frame builder should create Play, Options, and Quit buttons."""
        src = inspect.getsource(self.cls._build_main_frame)
        assert '"Play"' in src
        assert '"Options"' in src
        assert '"Quit"' in src

    def test_options_frame_navigates_to_submenus(self):
        """Options frame should navigate to keybinding, game_constants, sound."""
        src = inspect.getsource(self.cls._build_options_frame)
        assert '"keybinding"' in src
        assert '"game_constants"' in src
        assert '"sound"' in src

    def test_game_constants_navigates_to_difficulty_visibility(self):
        """Game constants should navigate to difficulty and visibility."""
        src = inspect.getsource(self.cls._build_game_constants_frame)
        assert '"difficulty"' in src
        assert '"visibility"' in src


# ── Test: Event Bus config_changed ──────────────────────────────────────


class TestConfigChangedEvent:
    """Test that config changes publish events correctly."""

    def test_event_bus_publishes_config_changed(self):
        """Publishing config_changed should deliver key/value to subscribers."""
        bus = EventBus()
        received = []
        bus.subscribe("config_changed", lambda **kw: received.append(kw))
        bus.publish("config_changed", key="INTRUDER_DEFAULT_HP", value=100)
        assert len(received) == 1
        assert received[0]["key"] == "INTRUDER_DEFAULT_HP"
        assert received[0]["value"] == 100

    def test_reset_defaults_restores_values(self):
        """capture_defaults + setattr + restore should round-trip config values."""
        defaults = capture_defaults(DIFFICULTY_SETTINGS)
        original_hp = _cfg.INTRUDER_DEFAULT_HP
        try:
            _cfg.INTRUDER_DEFAULT_HP = 999
            assert _cfg.INTRUDER_DEFAULT_HP == 999
            # Restore from defaults snapshot
            for attr, val in defaults.items():
                setattr(_cfg, attr, val)
            assert _cfg.INTRUDER_DEFAULT_HP == original_hp
        finally:
            _cfg.INTRUDER_DEFAULT_HP = original_hp


# ── Test: main.py wiring (source inspection) ───────────────────────────


class TestMainPyWiring:
    """Verify main.py correctly wires the MainMenu."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        from dungeon_builder.main import DungeonApp
        self.class_src = inspect.getsource(DungeonApp)

    def test_main_menu_imported(self):
        """main.py should import MainMenu."""
        import dungeon_builder.main as main_mod
        src = inspect.getsource(main_mod)
        assert "from dungeon_builder.ui.main_menu import MainMenu" in src

    def test_main_menu_instantiated(self):
        """MainMenu should be constructed in DungeonApp.__init__."""
        assert "MainMenu(" in self.class_src

    def test_escape_quit_removed(self):
        """The old escape→userExit binding should NOT be present."""
        assert 'self.accept("escape", self.userExit)' not in self.class_src
        assert "self.accept('escape', self.userExit)" not in self.class_src

    def test_starts_paused(self):
        """TimeManager should be set to speed 0 at startup."""
        assert "set_speed(0)" in self.class_src

    def test_layer_manager_gets_event_bus(self):
        """LayerSliceManager should receive event_bus argument."""
        assert "LayerSliceManager(self.render, event_bus)" in self.class_src

    def test_main_menu_in_subsystems(self):
        """main_menu should be stored in _subsystems."""
        assert '"main_menu"' in self.class_src or "'main_menu'" in self.class_src


# ── Test: Slider callback logic (source inspection) ────────────────────


class TestSliderCallbacks:
    """Verify slider callback logic in MainMenu."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        from dungeon_builder.ui.main_menu import MainMenu
        self.slider_src = inspect.getsource(MainMenu._build_slider_row)
        self.fog_src = inspect.getsource(MainMenu._build_fog_slider)
        self.reset_src = inspect.getsource(MainMenu._reset_defaults)

    def test_slider_snaps_to_step(self):
        """Slider callback should round to step size."""
        assert "round(raw / step) * step" in self.slider_src

    def test_slider_publishes_config_changed(self):
        """Slider callback should publish config_changed event."""
        assert 'config_changed' in self.slider_src

    def test_fog_slider_modifies_tuple(self):
        """Fog slider should decompose and recompose FOG_COLOR tuple."""
        assert "list(_cfg.FOG_COLOR)" in self.fog_src
        assert "tuple(fog)" in self.fog_src

    def test_reset_restores_via_setattr(self):
        """Reset defaults should use setattr to restore config values."""
        assert "setattr(_cfg" in self.reset_src
