"""Per-intruder fog-of-war map.

Each intruder maintains a sparse representation of the dungeon cells it has
personally observed.  Only "seen" cells are available for pathfinding.
"""

from __future__ import annotations

from dungeon_builder.config import (
    VOXEL_SPIKE,
    VOXEL_LAVA,
    VOXEL_TARP,
    VOXEL_TREASURE,
    VOXEL_DOOR,
    VOXEL_GOLD_BAIT,
    VOXEL_PRESSURE_PLATE,
    VOXEL_ALARM_BELL,
    VOXEL_FLOODGATE,
    VOXEL_STEAM_VENT,
    VOXEL_FRAGILE_FLOOR,
)

# Voxel types that are automatically flagged as hazards when revealed
_HAZARD_TYPES = frozenset({
    VOXEL_SPIKE, VOXEL_LAVA, VOXEL_TARP,
    VOXEL_PRESSURE_PLATE, VOXEL_STEAM_VENT, VOXEL_FRAGILE_FLOOR,
})


class PersonalMap:
    """Sparse fog-of-war map for a single intruder.

    Only cells the intruder has "seen" (via LOS or arcane sight) are stored.
    This is the *only* map data the intruder's pathfinder is allowed to use.
    """

    __slots__ = ("seen", "hazards", "treasures", "doors", "baits", "alarms",
                 "_generation", "_last_merge_gen")

    def __init__(self) -> None:
        # pos → last-observed voxel type (int)
        self.seen: dict[tuple[int, int, int], int] = {}
        # positions known to be dangerous
        self.hazards: set[tuple[int, int, int]] = set()
        # positions known to contain treasure
        self.treasures: set[tuple[int, int, int]] = set()
        # positions known to be doors → last-known block_state
        self.doors: dict[tuple[int, int, int], int] = {}
        # positions known to be gold baits (seen through arcane sight)
        self.baits: set[tuple[int, int, int]] = set()
        # positions known to be alarm bells
        self.alarms: set[tuple[int, int, int]] = set()
        # Monotonically increasing counter; bumped on every reveal/merge
        self._generation: int = 0
        # Tracks last-seen generation of *other* maps during merge
        # Maps id(other_map) → generation at last merge
        self._last_merge_gen: dict[int, int] = {}

    # ── Revelation ──────────────────────────────────────────────────

    def reveal(
        self,
        x: int,
        y: int,
        z: int,
        voxel_type: int,
        block_state: int = 0,
    ) -> None:
        """Record or update a single cell."""
        pos = (x, y, z)
        old = self.seen.get(pos)
        self.seen[pos] = voxel_type
        if old != voxel_type:
            self._generation += 1

        # Track special types
        if voxel_type in _HAZARD_TYPES:
            self.hazards.add(pos)
        else:
            self.hazards.discard(pos)

        if voxel_type == VOXEL_TREASURE:
            self.treasures.add(pos)
        else:
            self.treasures.discard(pos)

        if voxel_type == VOXEL_DOOR:
            self.doors[pos] = block_state
        else:
            self.doors.pop(pos, None)

        # Floodgate acts like a door (state-dependent transparency)
        if voxel_type == VOXEL_FLOODGATE:
            self.doors[pos] = block_state

        if voxel_type == VOXEL_ALARM_BELL:
            self.alarms.add(pos)
        else:
            self.alarms.discard(pos)

    # ── Queries ─────────────────────────────────────────────────────

    def is_revealed(self, x: int, y: int, z: int) -> bool:
        return (x, y, z) in self.seen

    def get_type(self, x: int, y: int, z: int) -> int | None:
        """Return the last-observed voxel type, or *None* if unseen."""
        return self.seen.get((x, y, z))

    def get_door_state(self, x: int, y: int, z: int) -> int | None:
        """Return last-known door state, or *None* if not a known door."""
        return self.doors.get((x, y, z))

    def get_frontier(self) -> list[tuple[int, int, int]]:
        """Return revealed cells that have at least one unrevealed orthogonal neighbor.

        These are the "exploration frontier" — targets for EXPLORE-objective
        intruders who want to uncover more of the map.
        """
        frontier: list[tuple[int, int, int]] = []
        for (x, y, z) in self.seen:
            for dx, dy, dz in ((1, 0, 0), (-1, 0, 0),
                                (0, 1, 0), (0, -1, 0),
                                (0, 0, 1), (0, 0, -1)):
                if (x + dx, y + dy, z + dz) not in self.seen:
                    frontier.append((x, y, z))
                    break
        return frontier

    # ── Cooperation ─────────────────────────────────────────────────

    def merge(self, other: PersonalMap) -> None:
        """Merge *other*'s knowledge into this map (ally sharing).

        For cells seen by both, the *other* map's data wins (it may be more
        recent).  This is a deliberate simplification — we don't track
        timestamps, so last-merge-wins is acceptable.

        Uses generation-based skip: if *other* hasn't changed since the last
        merge, the merge is a no-op.
        """
        other_id = id(other)
        last_gen = self._last_merge_gen.get(other_id, -1)
        if other._generation == last_gen:
            return  # Nothing new to merge

        self._last_merge_gen[other_id] = other._generation

        old_size = len(self.seen)
        self.seen.update(other.seen)
        self.hazards |= other.hazards
        self.treasures |= other.treasures
        self.doors.update(other.doors)
        self.baits |= other.baits
        self.alarms |= other.alarms

        # If anything was actually added, bump our generation
        if len(self.seen) != old_size:
            self._generation += 1

    def mark_hazard(self, x: int, y: int, z: int) -> None:
        """Manually flag a position as hazardous (e.g. after stepping on a tarp)."""
        self.hazards.add((x, y, z))

    def mark_bait(self, x: int, y: int, z: int) -> None:
        """Mark a gold bait position (seen through arcane sight)."""
        self.baits.add((x, y, z))

    def mark_alarm_zone(self, x: int, y: int, z: int) -> None:
        """Mark an alarm bell position."""
        self.alarms.add((x, y, z))

    def remove_treasure(self, x: int, y: int, z: int) -> None:
        """Remove a treasure position (after collection or destruction)."""
        self.treasures.discard((x, y, z))

    # ── Dunder ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.seen)

    def __repr__(self) -> str:
        return (
            f"PersonalMap(seen={len(self.seen)}, hazards={len(self.hazards)}, "
            f"treasures={len(self.treasures)}, doors={len(self.doors)}, "
            f"baits={len(self.baits)}, alarms={len(self.alarms)})"
        )
