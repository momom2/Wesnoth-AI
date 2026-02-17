"""Dungeon reputation system — tracks cumulative outcomes to influence future raids.

The dungeon develops a reputation based on how the player handles intruders:
- **Deadly**: many kills, few escapes → future intruders focus on destruction
- **Rich**: treasure is being lost → future intruders are greedier
- **Unknown**: too few data points → scout-heavy parties

Reputation affects party composition, objective voting, loyalty, and intruder levels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dungeon_builder.config import (
    REPUTATION_DEADLY_LETHALITY,
    REPUTATION_RICH_RICHNESS,
    REPUTATION_UNKNOWN_THRESHOLD,
    REPUTATION_DEADLY_MODIFIER,
    REPUTATION_RICH_MODIFIER,
    REPUTATION_UNKNOWN_MODIFIER,
    REPUTATION_DEADLY_LOYALTY,
    REPUTATION_RICH_LOYALTY,
    LEVEL_DEADLY_SHIFT,
)

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus


@dataclass(frozen=True)
class ReputationProfile:
    """Snapshot of the dungeon's reputation."""

    lethality: float  # 0.0 (harmless) to 1.0 (death trap)
    richness: float   # 0.0 (barren) to 1.0 (treasure hoard)


class DungeonReputation:
    """Tracks cumulative intruder outcomes to compute a dungeon profile.

    Subscribes to intruder events and updates counters. Provides spawn
    modifiers (objective weights, template weights, loyalty, level shift)
    that decision.py applies when spawning new parties.

    Pure Python, no Panda3D dependency — fully testable.
    """

    __slots__ = (
        "total_kills",
        "total_escapes",
        "total_treasure_lost",
        "total_parties_wiped",
        "parties_with_loot_escaped",
        "event_bus",
        "_last_profile",
    )

    def __init__(self, event_bus: EventBus) -> None:
        self.total_kills: int = 0
        self.total_escapes: int = 0
        self.total_treasure_lost: int = 0
        self.total_parties_wiped: int = 0
        self.parties_with_loot_escaped: int = 0
        self.event_bus = event_bus
        self._last_profile: ReputationProfile | None = None

        event_bus.subscribe("intruder_died", self._on_intruder_died)
        event_bus.subscribe("intruder_escaped", self._on_intruder_escaped)
        event_bus.subscribe("intruder_collected_treasure", self._on_treasure_collected)

    def _on_intruder_died(self, intruder=None, **kwargs) -> None:
        self.total_kills += 1
        self._check_profile_changed()

    def _on_intruder_escaped(self, intruder=None, **kwargs) -> None:
        self.total_escapes += 1
        if intruder is not None and getattr(intruder, "loot_count", 0) > 0:
            self.parties_with_loot_escaped += 1
        self._check_profile_changed()

    def _on_treasure_collected(self, **kwargs) -> None:
        self.total_treasure_lost += 1
        self._check_profile_changed()

    def on_party_wiped(self) -> None:
        """Called by decision.py when a party is fully wiped (all dead, none escaped)."""
        self.total_parties_wiped += 1

    def _check_profile_changed(self) -> None:
        """Publish event if the reputation profile has shifted."""
        profile = self.get_profile()
        if self._last_profile is None or (
            profile.lethality != self._last_profile.lethality
            or profile.richness != self._last_profile.richness
        ):
            self._last_profile = profile
            self.event_bus.publish(
                "reputation_changed",
                lethality=profile.lethality,
                richness=profile.richness,
            )

    def get_profile(self) -> ReputationProfile:
        """Compute current reputation profile."""
        total = self.total_kills + self.total_escapes
        if total > 0:
            lethality = self.total_kills / total
        else:
            lethality = 0.5  # Default: moderate danger

        denom = self.total_kills + self.total_treasure_lost
        if denom > 0:
            richness = self.total_treasure_lost / denom
        else:
            richness = 0.0

        return ReputationProfile(
            lethality=round(lethality, 4),
            richness=round(richness, 4),
        )

    def get_objective_modifier(self) -> tuple[float, float, float]:
        """Return additive (destroy, explore, pillage) offsets for objective voting."""
        total = self.total_kills + self.total_escapes
        if total < REPUTATION_UNKNOWN_THRESHOLD:
            return REPUTATION_UNKNOWN_MODIFIER

        profile = self.get_profile()
        if profile.lethality > REPUTATION_DEADLY_LETHALITY and profile.richness < REPUTATION_RICH_RICHNESS:
            return REPUTATION_DEADLY_MODIFIER
        if profile.richness > REPUTATION_RICH_RICHNESS:
            return REPUTATION_RICH_MODIFIER
        return (0.0, 0.0, 0.0)

    def get_template_weight_modifier(self) -> dict[str, float]:
        """Return template weight adjustments based on reputation."""
        total = self.total_kills + self.total_escapes
        if total < REPUTATION_UNKNOWN_THRESHOLD:
            return {}

        profile = self.get_profile()
        if profile.lethality > REPUTATION_DEADLY_LETHALITY and profile.richness < REPUTATION_RICH_RICHNESS:
            return {
                "Siege Force": 0.15,
                "War Band": 0.10,
                "Scouting Party": -0.10,
            }
        if profile.richness > REPUTATION_RICH_RICHNESS:
            return {
                "Scouting Party": 0.15,
                "Standard Raid": -0.10,
            }
        return {}

    def get_loyalty_modifier(self) -> float:
        """Return loyalty modifier applied to spawned intruders."""
        total = self.total_kills + self.total_escapes
        if total < REPUTATION_UNKNOWN_THRESHOLD:
            return 0.0

        profile = self.get_profile()
        if profile.lethality > REPUTATION_DEADLY_LETHALITY:
            return REPUTATION_DEADLY_LOYALTY
        if profile.richness > REPUTATION_RICH_RICHNESS:
            return REPUTATION_RICH_LOYALTY
        return 0.0

    def get_level_shift(self) -> float:
        """Return level distribution shift based on lethality.

        Deadly dungeons attract stronger (higher-level) intruders.
        Returns weight to shift from level 1 to higher levels.
        """
        profile = self.get_profile()
        if profile.lethality > 0.5:
            return (profile.lethality - 0.5) * LEVEL_DEADLY_SHIFT / 0.1
        return 0.0
