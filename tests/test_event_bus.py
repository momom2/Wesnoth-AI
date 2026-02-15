"""Tests for the EventBus pub/sub system."""

from dungeon_builder.core.event_bus import EventBus


def test_subscribe_and_publish():
    bus = EventBus()
    received = []
    bus.subscribe("test", lambda **kw: received.append(kw))
    bus.publish("test", x=1, y=2)
    assert received == [{"x": 1, "y": 2}]


def test_multiple_subscribers():
    bus = EventBus()
    results = []
    bus.subscribe("evt", lambda **kw: results.append("a"))
    bus.subscribe("evt", lambda **kw: results.append("b"))
    bus.publish("evt")
    assert results == ["a", "b"]


def test_unsubscribe():
    bus = EventBus()
    results = []
    cb = lambda **kw: results.append(1)
    bus.subscribe("evt", cb)
    bus.unsubscribe("evt", cb)
    bus.publish("evt")
    assert results == []


def test_no_subscribers():
    bus = EventBus()
    # Should not raise
    bus.publish("nonexistent", data=42)


def test_error_in_handler_does_not_crash():
    bus = EventBus()
    results = []

    def bad_handler(**kw):
        raise ValueError("oops")

    def good_handler(**kw):
        results.append("ok")

    bus.subscribe("evt", bad_handler)
    bus.subscribe("evt", good_handler)
    bus.publish("evt")
    # Good handler still fires despite bad handler error
    assert results == ["ok"]


def test_clear():
    bus = EventBus()
    results = []
    bus.subscribe("evt", lambda **kw: results.append(1))
    bus.clear()
    bus.publish("evt")
    assert results == []
