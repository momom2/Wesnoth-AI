"""Tests for PersonalMap handling of new block types.

Covers hazard classification, alarm/bait/floodgate tracking, merge
semantics for new sets, and hazard discard on type change.
"""

import pytest

from dungeon_builder.intruders.personal_map import PersonalMap
from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_PRESSURE_PLATE,
    VOXEL_STEAM_VENT,
    VOXEL_FRAGILE_FLOOR,
    VOXEL_ALARM_BELL,
    VOXEL_FLOODGATE,
    VOXEL_GOLD_BAIT,
)


# ── Hazard classification ────────────────────────────────────────────


def test_pressure_plate_is_hazard():
    """Revealing a pressure plate adds the position to the hazards set."""
    pm = PersonalMap()
    pm.reveal(5, 5, 3, VOXEL_PRESSURE_PLATE)

    assert (5, 5, 3) in pm.hazards


def test_steam_vent_is_hazard():
    """Revealing a steam vent adds the position to the hazards set."""
    pm = PersonalMap()
    pm.reveal(10, 2, 7, VOXEL_STEAM_VENT)

    assert (10, 2, 7) in pm.hazards


def test_fragile_floor_is_hazard():
    """Revealing a fragile floor adds the position to the hazards set."""
    pm = PersonalMap()
    pm.reveal(0, 0, 1, VOXEL_FRAGILE_FLOOR)

    assert (0, 0, 1) in pm.hazards


# ── Alarm bell tracking ──────────────────────────────────────────────


def test_alarm_bell_added_to_alarms():
    """Revealing an alarm bell adds it to the alarms set."""
    pm = PersonalMap()
    pm.reveal(3, 4, 2, VOXEL_ALARM_BELL)

    assert (3, 4, 2) in pm.alarms


def test_alarm_bell_removed_on_type_change():
    """Revealing the same position as a different type removes it from alarms."""
    pm = PersonalMap()
    pm.reveal(3, 4, 2, VOXEL_ALARM_BELL)
    assert (3, 4, 2) in pm.alarms

    # The bell is destroyed / replaced with air
    pm.reveal(3, 4, 2, VOXEL_AIR)
    assert (3, 4, 2) not in pm.alarms


# ── Floodgate as door ────────────────────────────────────────────────


def test_floodgate_tracked_as_door():
    """Revealing a floodgate adds it to the doors dict with its block state."""
    pm = PersonalMap()
    pm.reveal(7, 7, 5, VOXEL_FLOODGATE, block_state=1)

    assert (7, 7, 5) in pm.doors
    assert pm.doors[(7, 7, 5)] == 1


# ── Mark bait / alarm zone ───────────────────────────────────────────


def test_mark_bait_adds_to_baits():
    """mark_bait() adds the position to the baits set."""
    pm = PersonalMap()
    pm.mark_bait(12, 8, 4)

    assert (12, 8, 4) in pm.baits


def test_mark_alarm_zone_adds_to_alarms():
    """mark_alarm_zone() adds the position to the alarms set."""
    pm = PersonalMap()
    pm.mark_alarm_zone(6, 6, 0)

    assert (6, 6, 0) in pm.alarms


# ── Merge semantics for new sets ─────────────────────────────────────


def test_merge_includes_baits():
    """Merging two maps combines their baits sets."""
    pm_a = PersonalMap()
    pm_a.mark_bait(1, 2, 3)

    pm_b = PersonalMap()
    pm_b.mark_bait(4, 5, 6)
    # Give pm_b a reveal so its generation > -1 (merge requires change)
    pm_b.reveal(4, 5, 6, VOXEL_GOLD_BAIT)

    pm_a.merge(pm_b)

    assert (1, 2, 3) in pm_a.baits
    assert (4, 5, 6) in pm_a.baits


def test_merge_includes_alarms():
    """Merging two maps combines their alarms sets."""
    pm_a = PersonalMap()
    pm_a.mark_alarm_zone(10, 10, 0)

    pm_b = PersonalMap()
    pm_b.reveal(20, 20, 1, VOXEL_ALARM_BELL)

    pm_a.merge(pm_b)

    assert (10, 10, 0) in pm_a.alarms
    assert (20, 20, 1) in pm_a.alarms


# ── Hazard discard on type change ────────────────────────────────────


def test_hazard_discarded_on_type_change_to_air():
    """Revealing a hazard position as AIR removes it from hazards."""
    pm = PersonalMap()
    pm.reveal(5, 5, 3, VOXEL_PRESSURE_PLATE)
    assert (5, 5, 3) in pm.hazards

    pm.reveal(5, 5, 3, VOXEL_AIR)
    assert (5, 5, 3) not in pm.hazards
