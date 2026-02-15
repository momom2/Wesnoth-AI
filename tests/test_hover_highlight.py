"""Tests for voxel hover highlight event flow."""

from dungeon_builder.core.event_bus import EventBus


class TestHoverEvents:
    """Test that hover events propagate correctly through the event bus."""

    def test_voxel_hover_event_published(self):
        """voxel_hover event carries x, y, z coordinates."""
        bus = EventBus()
        received = []
        bus.subscribe("voxel_hover", lambda x, y, z: received.append((x, y, z)))

        bus.publish("voxel_hover", x=10, y=20, z=3)
        assert received == [(10, 20, 3)]

    def test_voxel_hover_clear_event_published(self):
        """voxel_hover_clear event fires with no coordinates."""
        bus = EventBus()
        received = []
        bus.subscribe("voxel_hover_clear", lambda **kw: received.append(True))

        bus.publish("voxel_hover_clear")
        assert received == [True]

    def test_hover_then_clear_sequence(self):
        """A hover followed by a clear produces the correct event sequence."""
        bus = EventBus()
        events = []
        bus.subscribe("voxel_hover", lambda x, y, z: events.append(("hover", x, y, z)))
        bus.subscribe("voxel_hover_clear", lambda **kw: events.append(("clear",)))

        bus.publish("voxel_hover", x=5, y=5, z=1)
        bus.publish("voxel_hover_clear")

        assert events == [("hover", 5, 5, 1), ("clear",)]

    def test_hover_changes_voxel(self):
        """Moving hover from one voxel to another publishes two hover events."""
        bus = EventBus()
        received = []
        bus.subscribe("voxel_hover", lambda x, y, z: received.append((x, y, z)))

        bus.publish("voxel_hover", x=1, y=2, z=3)
        bus.publish("voxel_hover", x=4, y=5, z=6)

        assert received == [(1, 2, 3), (4, 5, 6)]
