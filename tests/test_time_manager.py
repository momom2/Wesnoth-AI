"""Tests for the TimeManager tick system."""

from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.core.time_manager import TimeManager


def test_paused_produces_no_ticks():
    bus = EventBus()
    tm = TimeManager(bus)
    tm.set_speed(0)

    ticks = []
    bus.subscribe("tick", lambda tick: ticks.append(tick))
    tm.update(1.0)  # 1 second
    assert ticks == []


def test_normal_speed_20_ticks_per_second():
    bus = EventBus()
    tm = TimeManager(bus)
    tm.set_speed(1)

    ticks = []
    bus.subscribe("tick", lambda tick: ticks.append(tick))
    # Use slightly more than 1s to avoid float precision edge case
    tm.update(1.001)
    assert len(ticks) == 20


def test_fast_speed():
    bus = EventBus()
    tm = TimeManager(bus)
    tm.set_speed(2)

    ticks = []
    bus.subscribe("tick", lambda tick: ticks.append(tick))
    tm.update(1.0)  # 1 second at 3x speed -> 40 ticks (capped by spiral-of-death)
    # At 3x speed, should get ~60 ticks, but capped at 40 (TICKS_PER_SECOND * 2)
    assert len(ticks) == 40


def test_tick_count_increments():
    bus = EventBus()
    tm = TimeManager(bus)
    tm.set_speed(1)

    tm.update(0.151)  # ~0.15 seconds -> 3 ticks
    assert tm.tick_count == 3


def test_speed_change_event():
    bus = EventBus()
    tm = TimeManager(bus)

    events = []
    bus.subscribe("speed_changed", lambda **kw: events.append(kw))
    tm.set_speed(1)
    assert events == [{"speed": 1, "old_speed": 0}]


def test_paused_property():
    bus = EventBus()
    tm = TimeManager(bus)
    assert tm.paused is True
    tm.set_speed(1)
    assert tm.paused is False
