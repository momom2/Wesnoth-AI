"""Per-block-type micro-interaction system for intruders.

When an intruder is about to step onto or interact with a functional block,
the interaction handler determines what happens: damage, timed interaction,
repath, collection, or pass-through.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_DOOR,
    VOXEL_SPIKE,
    VOXEL_TREASURE,
    VOXEL_TARP,
    VOXEL_ROLLING_STONE,
    VOXEL_REINFORCED_WALL,
    VOXEL_LAVA,
    VOXEL_WATER,
    VOXEL_SLOPE,
    VOXEL_STAIRS,
    SPIKE_DAMAGE,
    ROLLING_STONE_DAMAGE,
    DOOR_BASH_TICKS,
    DOOR_BASH_TICKS_GORECLAW,
    DOOR_LOCKPICK_TICKS,
    TREASURE_GRAB_TICKS,
    TARP_DETECT_CUNNING,
)

if TYPE_CHECKING:
    from dungeon_builder.intruders.agent import Intruder


class InteractionResult(Enum):
    """Outcome of an intruder stepping onto a functional block."""

    CONTINUE = auto()      # No interaction needed, proceed
    INTERACT = auto()      # Start a timed interaction (door bash, dig, treasure grab)
    DAMAGE = auto()        # Intruder takes damage but continues
    REPATH = auto()        # Cannot proceed, needs new path
    COLLECT = auto()       # Collect item (treasure)
    FALL = auto()          # Fall through (tarp collapse)
    DEATH = auto()         # Instant death (lava for non-immune)
    DESTROY_BLOCK = auto() # Block is destroyed (spike smash by Goreclaw)


class InteractionInfo:
    """Details about an interaction result.

    Wraps the result enum plus any additional data (damage amount,
    interaction duration, block to destroy, etc.).
    """

    __slots__ = ("result", "damage", "ticks", "interaction_type")

    def __init__(
        self,
        result: InteractionResult,
        damage: int = 0,
        ticks: int = 0,
        interaction_type: str = "",
    ) -> None:
        self.result = result
        self.damage = damage
        self.ticks = ticks
        self.interaction_type = interaction_type

    def __repr__(self) -> str:
        return (
            f"InteractionInfo({self.result.name}, damage={self.damage}, "
            f"ticks={self.ticks}, type={self.interaction_type!r})"
        )


def handle_block(
    intruder: Intruder,
    voxel_type: int,
    block_state: int,
) -> InteractionInfo:
    """Determine the interaction when *intruder* encounters *voxel_type*.

    This is a pure function — it does NOT modify intruder state or the grid.
    The caller (decision engine) applies the result.

    Parameters
    ----------
    intruder : Intruder
        The intruder encountering the block.
    voxel_type : int
        The voxel type of the block being entered.
    block_state : int
        The block_state value (door open/closed, spike extended/retracted).

    Returns
    -------
    InteractionInfo
        What should happen.
    """
    arch = intruder.archetype

    # ── Air / Slope / Stairs — pass through ─────────────────────
    if voxel_type in (VOXEL_AIR, VOXEL_SLOPE, VOXEL_STAIRS):
        return InteractionInfo(InteractionResult.CONTINUE)

    # ── Door ────────────────────────────────────────────────────
    if voxel_type == VOXEL_DOOR:
        if block_state == 0:  # Open
            return InteractionInfo(InteractionResult.CONTINUE)
        # Closed door
        if arch.can_lockpick:
            return InteractionInfo(
                InteractionResult.INTERACT,
                ticks=DOOR_LOCKPICK_TICKS,
                interaction_type="lockpick",
            )
        if arch.can_bash_door:
            bash_ticks = DOOR_BASH_TICKS
            if arch.name == "Goreclaw":
                bash_ticks = DOOR_BASH_TICKS_GORECLAW
            return InteractionInfo(
                InteractionResult.INTERACT,
                ticks=bash_ticks,
                interaction_type="bash_door",
            )
        return InteractionInfo(InteractionResult.REPATH)

    # ── Spike ───────────────────────────────────────────────────
    if voxel_type == VOXEL_SPIKE:
        if block_state == 0:  # Retracted
            return InteractionInfo(InteractionResult.CONTINUE)
        # Extended spike
        # Windcaller flies over
        if arch.can_fly:
            return InteractionInfo(InteractionResult.CONTINUE)
        # Goreclaw smashes spike (takes 10 damage, destroys it)
        if arch.frenzy_threshold > 0 and arch.name == "Goreclaw":
            return InteractionInfo(
                InteractionResult.DESTROY_BLOCK,
                damage=SPIKE_DAMAGE // 2,
            )
        # Vanguard takes half damage
        if arch.name == "Vanguard":
            return InteractionInfo(
                InteractionResult.DAMAGE,
                damage=SPIKE_DAMAGE // 2,
            )
        # Shadowblade detects and avoids (if spike_detect_range > 0)
        if arch.spike_detect_range > 0:
            return InteractionInfo(InteractionResult.REPATH)
        # Everyone else takes full damage
        return InteractionInfo(
            InteractionResult.DAMAGE,
            damage=SPIKE_DAMAGE,
        )

    # ── Treasure ────────────────────────────────────────────────
    if voxel_type == VOXEL_TREASURE:
        if arch.greed > 0:
            return InteractionInfo(
                InteractionResult.COLLECT,
                ticks=TREASURE_GRAB_TICKS,
                interaction_type="grab_treasure",
            )
        return InteractionInfo(InteractionResult.CONTINUE)

    # ── Tarp ────────────────────────────────────────────────────
    if voxel_type == VOXEL_TARP:
        # Windcaller flies over
        if arch.can_fly:
            return InteractionInfo(InteractionResult.CONTINUE)
        # Gloomseer detects via arcane sight
        if arch.arcane_sight_range > 0:
            return InteractionInfo(InteractionResult.REPATH)
        # High cunning detects
        if arch.cunning >= TARP_DETECT_CUNNING:
            return InteractionInfo(InteractionResult.REPATH)
        # Everyone else falls through
        return InteractionInfo(InteractionResult.FALL)

    # ── Rolling stone ───────────────────────────────────────────
    if voxel_type == VOXEL_ROLLING_STONE:
        # Windcaller flies over
        if arch.can_fly:
            return InteractionInfo(InteractionResult.CONTINUE)
        # Fast intruders dodge (speed >= 3)
        if arch.speed >= 3:
            return InteractionInfo(InteractionResult.CONTINUE)
        # Everyone else takes damage
        return InteractionInfo(
            InteractionResult.DAMAGE,
            damage=ROLLING_STONE_DAMAGE,
        )

    # ── Reinforced wall ─────────────────────────────────────────
    if voxel_type == VOXEL_REINFORCED_WALL:
        return InteractionInfo(InteractionResult.REPATH)

    # ── Lava ────────────────────────────────────────────────────
    if voxel_type == VOXEL_LAVA:
        if arch.fire_immune:
            return InteractionInfo(InteractionResult.CONTINUE)
        return InteractionInfo(InteractionResult.DEATH)

    # ── Water ───────────────────────────────────────────────────
    if voxel_type == VOXEL_WATER:
        # Pyremancer dies in water
        if arch.fire_immune:
            return InteractionInfo(InteractionResult.DEATH)
        return InteractionInfo(InteractionResult.REPATH)

    # ── Other solid blocks — impassable ─────────────────────────
    # Digger handled by pathfinder (not here, since dig is a timed action)
    return InteractionInfo(InteractionResult.REPATH)
