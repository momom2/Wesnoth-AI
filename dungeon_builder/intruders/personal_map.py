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
)

# Voxel types that are automatically flagged as hazards when revealed
_HAZARD_TYPES = frozenset({VOXEL_SPIKE, VOXEL_LAVA, VOXEL_TARP})


class PersonalMap:
    """Sparse fog-of-war map for a single intruder.

    Only cells the intruder has "seen" (via LOS or arcane sight) are stored.
    This is the *only* map data the intruder's pathfinder is allowed to use.
    """

    __slots__ = ("seen", "hazards", "treasures", "doors")

    def __init__(self) -> None:
        # pos → last-observed voxel type (int)
        self.seen: dict[tuple[int, int, int], int] = {}
        # positions known to be dangerous
        self.hazards: set[tuple[int, int, int]] = set()
        # positions known to contain treasure
        self.treasures: set[tuple[int, int, int]] = set()
        # positions known to be doors → last-known block_state
        self.doors: dict[tuple[int, int, int], int] = {}

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
        self.seen[pos] = voxel_type

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
        """
        self.seen.update(other.seen)
        self.hazards |= other.hazards
        self.treasures |= other.treasures
        self.doors.update(other.doors)

    def mark_hazard(self, x: int, y: int, z: int) -> None:
        """Manually flag a position as hazardous (e.g. after stepping on a tarp)."""
        self.hazards.add((x, y, z))

    def remove_treasure(self, x: int, y: int, z: int) -> None:
        """Remove a treasure position (after collection or destruction)."""
        self.treasures.discard((x, y, z))

    # ── Dunder ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.seen)

    def __repr__(self) -> str:
        return (
            f"PersonalMap(seen={len(self.seen)}, hazards={len(self.hazards)}, "
            f"treasures={len(self.treasures)}, doors={len(self.doors)})"
        )
