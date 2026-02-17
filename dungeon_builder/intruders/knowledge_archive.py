"""Persistent per-faction knowledge archive with uncertainty tracking.

Escaped intruders archive their personal maps into a faction-level knowledge pool.
New parties receive this archived knowledge at spawn time (filtered by staleness
and uncertainty). Contradictory reports increase uncertainty; confirmations decrease
it. Player modifications to the dungeon also increase uncertainty in archived cells.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dungeon_builder.config import (
    KNOWLEDGE_STALE_TICKS,
    KNOWLEDGE_UNCERTAIN_THRESHOLD,
    KNOWLEDGE_CONTRADICTION_BASE,
    KNOWLEDGE_CONFIRM_DECAY,
    KNOWLEDGE_CHANGE_UNCERTAINTY,
    VOXEL_SPIKE,
    VOXEL_LAVA,
    VOXEL_TARP,
    VOXEL_TREASURE,
    VOXEL_DOOR,
)

if TYPE_CHECKING:
    from dungeon_builder.intruders.agent import Intruder
    from dungeon_builder.intruders.personal_map import PersonalMap

_HAZARD_TYPES = frozenset((VOXEL_SPIKE, VOXEL_LAVA, VOXEL_TARP))


class FactionMap:
    """Faction-level knowledge of the dungeon with per-cell uncertainty.

    Stores what cells have been seen, what their types were, when they were
    last reported, and how much the faction trusts that data. Uncertainty is
    a float from 0.0 (certain) to 1.0 (completely unknown).
    """

    __slots__ = ("seen", "hazards", "treasures", "doors", "uncertainty")

    def __init__(self) -> None:
        # (x,y,z) → (vtype, tick_archived, trust_weight)
        self.seen: dict[tuple[int, int, int], tuple[int, int, float]] = {}
        self.hazards: set[tuple[int, int, int]] = set()
        self.treasures: set[tuple[int, int, int]] = set()
        self.doors: dict[tuple[int, int, int], int] = {}
        # (x,y,z) → uncertainty float 0.0-1.0
        self.uncertainty: dict[tuple[int, int, int], float] = {}


class KnowledgeArchive:
    """Persistent knowledge archive for both surface and underworld factions.

    Pure Python, no Panda3D dependency — fully testable without a window.
    """

    __slots__ = ("_surface_data", "_underworld_data")

    def __init__(self) -> None:
        self._surface_data = FactionMap()
        self._underworld_data = FactionMap()

    def _get_faction(self, is_underworlder: bool) -> FactionMap:
        return self._underworld_data if is_underworlder else self._surface_data

    def archive_survivor(self, intruder: Intruder, tick: int) -> None:
        """Archive an escaped intruder's personal map into the faction archive.

        Contradictory reports (different vtype for same cell) increase
        uncertainty, weighted by relative trust. Confirmations (same vtype)
        decrease uncertainty.

        Parameters
        ----------
        intruder : Intruder
            The escaped intruder whose map to archive.
        tick : int
            The current game tick (for timestamps).
        """
        from dungeon_builder.intruders.archetypes import STATUS_TRUST

        faction = self._get_faction(intruder.is_underworlder)
        pmap = intruder.personal_map
        survivor_trust = STATUS_TRUST.get(intruder.status, 0.5)

        for pos, vtype in pmap.seen.items():
            if pos in faction.seen:
                existing_vtype, _existing_tick, existing_trust = faction.seen[pos]
                if existing_vtype != vtype:
                    # Contradiction: increase uncertainty weighted by relative trust
                    # Higher survivor trust → less uncertainty added (they're more credible)
                    weight = existing_trust / (existing_trust + survivor_trust)
                    unc_delta = KNOWLEDGE_CONTRADICTION_BASE * weight
                    old_unc = faction.uncertainty.get(pos, 0.0)
                    faction.uncertainty[pos] = min(1.0, old_unc + unc_delta)
                else:
                    # Confirmation: decrease uncertainty
                    old_unc = faction.uncertainty.get(pos, 0.0)
                    if old_unc > 0.0:
                        new_unc = old_unc * KNOWLEDGE_CONFIRM_DECAY
                        if new_unc < 0.01:
                            faction.uncertainty.pop(pos, None)
                        else:
                            faction.uncertainty[pos] = new_unc

            # Update seen entry with survivor's data
            faction.seen[pos] = (vtype, tick, survivor_trust)

            # Update hazards
            if vtype in _HAZARD_TYPES:
                faction.hazards.add(pos)
            else:
                faction.hazards.discard(pos)

            # Update treasures
            if vtype == VOXEL_TREASURE:
                faction.treasures.add(pos)
            else:
                faction.treasures.discard(pos)

            # Update doors
            if vtype == VOXEL_DOOR:
                door_state = pmap.doors.get(pos, 0)
                faction.doors[pos] = door_state
            else:
                faction.doors.pop(pos, None)

    def inject_knowledge(
        self,
        personal_map: PersonalMap,
        is_underworlder: bool,
        current_tick: int,
        cunning: float,
    ) -> None:
        """Inject faction knowledge into a new intruder's personal map.

        Filters by staleness and uncertainty (adjusted by intruder's cunning).
        Risk-averse intruders (high cunning) trust less data and prefer to
        explore uncertain areas themselves.

        Parameters
        ----------
        personal_map : PersonalMap
            The new intruder's (empty) personal map to populate.
        is_underworlder : bool
            Which faction's archive to read from.
        current_tick : int
            Current game tick for staleness check.
        cunning : float
            The intruder's cunning stat (0.0-1.0). Higher cunning = lower
            uncertainty threshold = trusts less archived data.
        """
        faction = self._get_faction(is_underworlder)

        if not faction.seen:
            return

        # Cunning adjusts threshold: high cunning → lower threshold → trusts less
        threshold = KNOWLEDGE_UNCERTAIN_THRESHOLD - cunning * 0.2

        for pos, (vtype, archived_tick, _trust) in faction.seen.items():
            # Staleness filter
            if current_tick - archived_tick > KNOWLEDGE_STALE_TICKS:
                continue

            # Uncertainty filter
            unc = faction.uncertainty.get(pos, 0.0)
            if unc > threshold:
                continue

            # Inject into personal map
            x, y, z = pos
            door_state = faction.doors.get(pos, 0)
            personal_map.reveal(x, y, z, vtype, door_state)

    def on_voxel_changed(self, x: int, y: int, z: int) -> None:
        """Increase uncertainty when the player modifies a cell.

        Called when any voxel changes type (dig, build, craft).
        Affects both faction archives since player actions are visible
        to any faction that has seen that cell.
        """
        pos = (x, y, z)
        for faction in (self._surface_data, self._underworld_data):
            if pos in faction.seen:
                old_unc = faction.uncertainty.get(pos, 0.0)
                faction.uncertainty[pos] = min(1.0, old_unc + KNOWLEDGE_CHANGE_UNCERTAINTY)

    def get_uncertainty(self, x: int, y: int, z: int, is_underworlder: bool) -> float:
        """Return uncertainty for a cell (0.0 if not tracked)."""
        faction = self._get_faction(is_underworlder)
        return faction.uncertainty.get((x, y, z), 0.0)

    def get_stats(self, is_underworlder: bool) -> dict:
        """Return summary stats for a faction's knowledge archive."""
        faction = self._get_faction(is_underworlder)
        unc_values = list(faction.uncertainty.values())
        avg_unc = sum(unc_values) / len(unc_values) if unc_values else 0.0
        return {
            "cells_known": len(faction.seen),
            "hazards_known": len(faction.hazards),
            "treasures_known": len(faction.treasures),
            "avg_uncertainty": avg_unc,
        }
