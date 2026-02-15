"""Tests for the dungeon core."""

from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.dungeon_core.core import DungeonCore


def test_initial_hp():
    bus = EventBus()
    core = DungeonCore(bus, 5, 5, 5, hp=100)
    assert core.hp == 100
    assert core.max_hp == 100
    assert core.alive is True


def test_take_damage():
    bus = EventBus()
    core = DungeonCore(bus, 5, 5, 5, hp=100)

    events = []
    bus.subscribe("core_damaged", lambda **kw: events.append(kw))

    core.take_damage(30)
    assert core.hp == 70
    assert len(events) == 1
    assert events[0]["hp"] == 70
    assert events[0]["max_hp"] == 100


def test_game_over_at_zero():
    bus = EventBus()
    core = DungeonCore(bus, 5, 5, 5, hp=50)

    game_over_events = []
    bus.subscribe("game_over", lambda **kw: game_over_events.append(kw))

    core.take_damage(50)
    assert core.hp == 0
    assert core.alive is False
    assert len(game_over_events) == 1
    assert game_over_events[0]["reason"] == "core_destroyed"


def test_damage_clamped_at_zero():
    bus = EventBus()
    core = DungeonCore(bus, 5, 5, 5, hp=20)
    core.take_damage(100)
    assert core.hp == 0


def test_no_damage_after_dead():
    bus = EventBus()
    core = DungeonCore(bus, 5, 5, 5, hp=10)
    core.take_damage(10)  # Kill
    assert core.alive is False

    damage_events = []
    bus.subscribe("core_damaged", lambda **kw: damage_events.append(kw))
    core.take_damage(10)  # Should be ignored
    assert len(damage_events) == 0
