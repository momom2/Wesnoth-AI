"""Slider configuration registry for the options menu.

Each entry maps a ``config.py`` attribute to a UI slider with display name,
range, step size, and type.  The ``MainMenu`` reads these lists to build
sliders dynamically — adding a new setting only requires a new dict entry.
"""

from __future__ import annotations

import dungeon_builder.config as _cfg

# ── Difficulty settings ──────────────────────────────────────────────────

DIFFICULTY_SETTINGS: list[dict] = [
    {"attr": "INTRUDER_DEFAULT_HP",           "label": "Intruder HP",          "min": 10,   "max": 200,  "step": 5,    "type": int},
    {"attr": "INTRUDER_DEFAULT_DAMAGE",       "label": "Intruder Damage",      "min": 1,    "max": 50,   "step": 1,    "type": int},
    {"attr": "INTRUDER_SPAWN_INTERVAL",       "label": "Spawn Interval",       "min": 50,   "max": 1000, "step": 10,   "type": int},
    {"attr": "INTRUDER_PARTY_SPAWN_INTERVAL", "label": "Party Spawn Interval", "min": 100,  "max": 2000, "step": 20,   "type": int},
    {"attr": "MAX_PARTIES",                   "label": "Max Parties",          "min": 1,    "max": 10,   "step": 1,    "type": int},
    {"attr": "MAX_INTRUDERS_TOTAL",           "label": "Max Intruders",        "min": 4,    "max": 64,   "step": 2,    "type": int},
    {"attr": "MORALE_BASE",                   "label": "Morale Base",          "min": 0.1,  "max": 1.0,  "step": 0.05, "type": float},
    {"attr": "MORALE_FLEE_THRESHOLD",         "label": "Morale Flee",          "min": 0.0,  "max": 0.5,  "step": 0.05, "type": float},
    {"attr": "LEVEL_HP_SCALE",                "label": "Level HP Scale",       "min": 0.0,  "max": 0.5,  "step": 0.05, "type": float},
    {"attr": "LEVEL_DAMAGE_SCALE",            "label": "Level Dmg Scale",      "min": 0.0,  "max": 0.5,  "step": 0.05, "type": float},
]

# ── Visibility settings ──────────────────────────────────────────────────

VISIBILITY_SETTINGS: list[dict] = [
    {"attr": "CAMERA_PAN_SPEED",         "label": "Camera Pan Speed",    "min": 5.0,  "max": 100.0, "step": 5.0,  "type": float},
    {"attr": "CAMERA_ROTATE_SPEED",      "label": "Camera Rotate Speed", "min": 10.0, "max": 180.0, "step": 10.0, "type": float},
    {"attr": "CAMERA_ZOOM_STEP",         "label": "Camera Zoom Step",    "min": 1.0,  "max": 20.0,  "step": 1.0,  "type": float},
    {"attr": "CAMERA_DEFAULT_DISTANCE",  "label": "Camera Distance",     "min": 10.0, "max": 100.0, "step": 5.0,  "type": float},
    {"attr": "CAMERA_MIN_DISTANCE",      "label": "Camera Min Distance", "min": 5.0,  "max": 50.0,  "step": 5.0,  "type": float},
    {"attr": "CAMERA_MAX_DISTANCE",      "label": "Camera Max Distance", "min": 50.0, "max": 200.0, "step": 10.0, "type": float},
    {"attr": "LAYER_MAX_VISIBLE_BELOW",  "label": "Visible Below",       "min": 1,    "max": 15,    "step": 1,    "type": int},
    {"attr": "LAYER_MAX_VISIBLE_ABOVE",  "label": "Visible Above",       "min": 0,    "max": 10,    "step": 1,    "type": int},
]

# ── Fog colour (special: tuple decomposition) ────────────────────────────

FOG_COLOR_SETTINGS: list[dict] = [
    {"index": 0, "label": "Fog Red",   "min": 0.0, "max": 1.0, "step": 0.01},
    {"index": 1, "label": "Fog Green", "min": 0.0, "max": 1.0, "step": 0.01},
    {"index": 2, "label": "Fog Blue",  "min": 0.0, "max": 1.0, "step": 0.01},
    {"index": 3, "label": "Fog Alpha", "min": 0.0, "max": 1.0, "step": 0.05},
]


def capture_defaults(settings: list[dict]) -> dict[str, object]:
    """Snapshot current config values for the given settings list."""
    return {s["attr"]: getattr(_cfg, s["attr"]) for s in settings}


def capture_fog_defaults() -> tuple[float, ...]:
    """Snapshot the current FOG_COLOR tuple."""
    return tuple(_cfg.FOG_COLOR)
