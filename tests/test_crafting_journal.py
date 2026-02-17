"""Tests for the CraftingJournal discovery tracking system."""

from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.building.crafting_book import CraftingBook
from dungeon_builder.building.crafting_journal import CraftingJournal


def _setup():
    bus = EventBus()
    book = CraftingBook()
    journal = CraftingJournal(bus, book)
    return bus, book, journal


class TestInitialState:
    def test_starts_empty(self):
        """Journal starts with no discovered recipes."""
        _, book, journal = _setup()
        assert journal.discovered_count == 0
        assert journal.total_recipes == len(book.recipes)
        assert len(journal.discovered_names) == 0

    def test_total_matches_book(self):
        """Total recipes matches the number in the crafting book."""
        _, book, journal = _setup()
        assert journal.total_recipes == len(book.recipes)
        assert journal.total_recipes > 0  # sanity check


class TestDiscovery:
    def test_craft_success_discovers_recipe(self):
        """Publishing craft_success marks the recipe as discovered."""
        bus, _, journal = _setup()
        bus.publish("craft_success", recipe="Marble Wall", x=0, y=0, z=0)
        assert journal.is_discovered("Marble Wall")
        assert journal.discovered_count == 1

    def test_duplicate_craft_no_double_count(self):
        """Crafting the same recipe twice does not increase count."""
        bus, _, journal = _setup()
        bus.publish("craft_success", recipe="Marble Wall", x=0, y=0, z=0)
        bus.publish("craft_success", recipe="Marble Wall", x=1, y=1, z=1)
        assert journal.discovered_count == 1

    def test_multiple_distinct_recipes(self):
        """Discovering different recipes increases count."""
        bus, _, journal = _setup()
        bus.publish("craft_success", recipe="Marble Wall", x=0, y=0, z=0)
        bus.publish("craft_success", recipe="Ore Smelting", x=0, y=0, z=0)
        bus.publish("craft_success", recipe="Door", x=0, y=0, z=0)
        assert journal.discovered_count == 3
        assert journal.is_discovered("Marble Wall")
        assert journal.is_discovered("Ore Smelting")
        assert journal.is_discovered("Door")

    def test_unknown_name_not_discovered(self):
        """Querying an unknown recipe name returns False."""
        _, _, journal = _setup()
        assert journal.is_discovered("Nonexistent Recipe") is False


class TestRecipeDiscoveredEvent:
    def test_event_published_on_first_discovery(self):
        """First discovery publishes recipe_discovered event."""
        bus, _, journal = _setup()
        events = []
        bus.subscribe("recipe_discovered", lambda **kw: events.append(kw))

        bus.publish("craft_success", recipe="Glass", x=0, y=0, z=0)
        assert len(events) == 1
        assert events[0]["recipe"] == "Glass"
        assert events[0]["total"] == 1

    def test_event_not_published_on_duplicate(self):
        """Second craft of same recipe does NOT publish recipe_discovered."""
        bus, _, journal = _setup()
        events = []
        bus.subscribe("recipe_discovered", lambda **kw: events.append(kw))

        bus.publish("craft_success", recipe="Glass", x=0, y=0, z=0)
        bus.publish("craft_success", recipe="Glass", x=1, y=1, z=1)
        assert len(events) == 1  # only first time

    def test_event_total_increments(self):
        """Each new discovery increments the total in the event."""
        bus, _, journal = _setup()
        events = []
        bus.subscribe("recipe_discovered", lambda **kw: events.append(kw))

        bus.publish("craft_success", recipe="Glass", x=0, y=0, z=0)
        bus.publish("craft_success", recipe="Door", x=0, y=0, z=0)
        assert events[0]["total"] == 1
        assert events[1]["total"] == 2


class TestDisplayData:
    def test_all_undiscovered_shows_placeholder(self):
        """Undiscovered recipes show as '???' with placeholder description."""
        _, book, journal = _setup()
        display = journal.get_all_recipes_display()

        assert len(display) == len(book.recipes)
        for entry in display:
            assert entry["name"] == "???"
            assert entry["discovered"] is False
            assert "reveal" in entry["description"].lower()

    def test_mixed_discovered_undiscovered(self):
        """Mix of discovered and undiscovered recipes."""
        bus, book, journal = _setup()
        first_name = book.recipes[0].name
        first_desc = book.recipes[0].description
        bus.publish("craft_success", recipe=first_name, x=0, y=0, z=0)

        display = journal.get_all_recipes_display()

        # First recipe is discovered
        assert display[0]["name"] == first_name
        assert display[0]["discovered"] is True
        assert display[0]["description"] == first_desc

        # Rest are undiscovered
        for entry in display[1:]:
            assert entry["name"] == "???"
            assert entry["discovered"] is False

    def test_display_preserves_recipe_order(self):
        """Display list matches crafting book recipe order."""
        bus, book, journal = _setup()
        # Discover last recipe
        last_name = book.recipes[-1].name
        bus.publish("craft_success", recipe=last_name, x=0, y=0, z=0)

        display = journal.get_all_recipes_display()

        # Last entry should be discovered
        assert display[-1]["name"] == last_name
        assert display[-1]["discovered"] is True
        # First entry should still be undiscovered
        assert display[0]["name"] == "???"
        assert display[0]["discovered"] is False


class TestDiscoverAll:
    def test_discover_all(self):
        """discover_all() marks every recipe as discovered."""
        _, book, journal = _setup()
        journal.discover_all()
        assert journal.discovered_count == journal.total_recipes

        display = journal.get_all_recipes_display()
        for entry in display:
            assert entry["discovered"] is True
            assert entry["name"] != "???"


class TestDiscoveredNames:
    def test_returns_frozenset(self):
        """discovered_names returns a frozenset (immutable copy)."""
        _, _, journal = _setup()
        names = journal.discovered_names
        assert isinstance(names, frozenset)

    def test_frozenset_contains_discovered(self):
        """frozenset contains all discovered recipe names."""
        bus, _, journal = _setup()
        bus.publish("craft_success", recipe="Marble Wall", x=0, y=0, z=0)
        bus.publish("craft_success", recipe="Door", x=0, y=0, z=0)
        names = journal.discovered_names
        assert "Marble Wall" in names
        assert "Door" in names
        assert len(names) == 2
