"""Party system for intruder groups.

Intruders travel in parties of 3-8 members.  Each party has a composition
template, a leader, a collective objective (voted from member weights), and
mechanics for map sharing, warden healing, and betrayal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from dungeon_builder.intruders.archetypes import (
    IntruderObjective,
    VANGUARD,
    SHADOWBLADE,
    TUNNELER,
    PYREMANCER,
    WINDCALLER,
    WARDEN,
    GORECLAW,
    GLOOMSEER,
    ArchetypeStats,
)
from dungeon_builder.config import (
    MAP_SHARE_RANGE,
    WARDEN_HEAL_AMOUNT,
    WARDEN_HEAL_INTERVAL,
    WARDEN_LOYALTY_BONUS,
    WARDEN_DEATH_LOYALTY_PENALTY,
    PARTY_WEIGHT_STANDARD_RAID,
    PARTY_WEIGHT_SCOUTING_PARTY,
    PARTY_WEIGHT_SIEGE_FORCE,
    PARTY_WEIGHT_WAR_BAND,
)

if TYPE_CHECKING:
    from dungeon_builder.intruders.agent import Intruder
    from dungeon_builder.utils.rng import SeededRNG


# ── Composition templates ──────────────────────────────────────────────

@dataclass(frozen=True)
class _MemberSlot:
    """One or more archetypes to choose from for a party slot."""
    choices: tuple[ArchetypeStats, ...]
    min_count: int
    max_count: int


@dataclass(frozen=True)
class PartyTemplate:
    """A weighted template describing a possible party composition."""
    name: str
    weight: float
    slots: tuple[_MemberSlot, ...]


STANDARD_RAID = PartyTemplate(
    name="Standard Raid",
    weight=PARTY_WEIGHT_STANDARD_RAID,
    slots=(
        _MemberSlot((VANGUARD,), 1, 2),
        _MemberSlot((SHADOWBLADE,), 1, 1),
        _MemberSlot((TUNNELER, PYREMANCER), 1, 1),
        _MemberSlot((GORECLAW,), 1, 2),
        _MemberSlot((WARDEN,), 0, 1),
    ),
)

SCOUTING_PARTY = PartyTemplate(
    name="Scouting Party",
    weight=PARTY_WEIGHT_SCOUTING_PARTY,
    slots=(
        _MemberSlot((WINDCALLER,), 1, 2),
        _MemberSlot((GLOOMSEER,), 1, 1),
        _MemberSlot((SHADOWBLADE,), 1, 2),
        _MemberSlot((WARDEN,), 0, 1),
    ),
)

SIEGE_FORCE = PartyTemplate(
    name="Siege Force",
    weight=PARTY_WEIGHT_SIEGE_FORCE,
    slots=(
        _MemberSlot((TUNNELER,), 2, 3),
        _MemberSlot((VANGUARD,), 1, 2),
        _MemberSlot((PYREMANCER,), 1, 1),
        _MemberSlot((WARDEN,), 0, 1),
    ),
)

WAR_BAND = PartyTemplate(
    name="War Band",
    weight=PARTY_WEIGHT_WAR_BAND,
    slots=(
        _MemberSlot((GORECLAW,), 3, 4),
        _MemberSlot((VANGUARD,), 1, 1),
        _MemberSlot((WARDEN,), 0, 1),
    ),
)

ALL_TEMPLATES: tuple[PartyTemplate, ...] = (
    STANDARD_RAID, SCOUTING_PARTY, SIEGE_FORCE, WAR_BAND,
)


# ── Composition generation ─────────────────────────────────────────────

def choose_template(rng: SeededRNG) -> PartyTemplate:
    """Weighted random selection of a party template."""
    roll = rng.random()
    cumulative = 0.0
    for tmpl in ALL_TEMPLATES:
        cumulative += tmpl.weight
        if roll < cumulative:
            return tmpl
    # Fallback (floating-point edge case)
    return ALL_TEMPLATES[-1]


def generate_composition(
    template: PartyTemplate, rng: SeededRNG,
) -> list[ArchetypeStats]:
    """Generate a concrete list of archetypes from a party template.

    For each slot, a random count between min_count and max_count is chosen.
    For each member in that slot, a random archetype from the slot choices
    is picked.
    """
    result: list[ArchetypeStats] = []
    for slot in template.slots:
        count = rng.randint(slot.min_count, slot.max_count)
        for _ in range(count):
            result.append(rng.choice(slot.choices))
    return result


# ── Party class ────────────────────────────────────────────────────────

class Party:
    """A group of intruders traveling together.

    The party manages shared objectives, map merging, warden healing,
    betrayal checks, and leader election.
    """

    __slots__ = (
        "id",
        "members",
        "_leader_id",
        "_objective",
        "_heal_tick_counter",
    )

    def __init__(self, party_id: int, members: list[Intruder]) -> None:
        self.id = party_id
        self.members: list[Intruder] = list(members)
        self._leader_id: int | None = None
        self._objective: IntruderObjective | None = None
        self._heal_tick_counter: int = 0

        # Assign party_id to all members
        for m in self.members:
            m.party_id = party_id

        # Initial elections
        self._elect_leader()
        self._vote_objective()

    # ── Leader election ────────────────────────────────────────────

    def _elect_leader(self) -> None:
        """Elect the alive member with highest effective_loyalty as leader.

        Ties are broken by intruder id (lowest wins for determinism).
        """
        alive = [m for m in self.members if m.alive]
        if not alive:
            self._leader_id = None
            return
        # Sort by (-loyalty, id) → highest loyalty first, then lowest id
        best = min(alive, key=lambda m: (-m.effective_loyalty, m.id))
        self._leader_id = best.id

    @property
    def leader(self) -> Intruder | None:
        """Return the current leader, or None if everyone is dead."""
        if self._leader_id is None:
            return None
        for m in self.members:
            if m.id == self._leader_id and m.alive:
                return m
        # Leader died since last election — re-elect
        self._elect_leader()
        if self._leader_id is None:
            return None
        for m in self.members:
            if m.id == self._leader_id:
                return m
        return None

    # ── Objective voting ───────────────────────────────────────────

    def _vote_objective(self) -> None:
        """Set party objective by weighted vote from alive members.

        Each member contributes its archetype's objective_weights.
        The objective with the highest total weight wins.
        Leader breaks ties.
        """
        alive = [m for m in self.members if m.alive]
        if not alive:
            self._objective = IntruderObjective.DESTROY_CORE
            return

        totals = [0.0, 0.0, 0.0]  # destroy, explore, pillage
        for m in alive:
            w = m.archetype.objective_weights
            totals[0] += w[0]
            totals[1] += w[1]
            totals[2] += w[2]

        objectives = [
            IntruderObjective.DESTROY_CORE,
            IntruderObjective.EXPLORE,
            IntruderObjective.PILLAGE,
        ]

        max_val = max(totals)
        # Collect all objectives tied at max
        tied = [objectives[i] for i, v in enumerate(totals) if v == max_val]

        if len(tied) == 1:
            self._objective = tied[0]
        else:
            # Leader breaks the tie: pick whichever the leader prefers most
            leader = self.leader
            if leader is not None:
                lw = leader.archetype.objective_weights
                # Among tied objectives, pick the one with highest leader weight
                best = max(tied, key=lambda o: lw[objectives.index(o)])
                self._objective = best
            else:
                self._objective = tied[0]

    @property
    def objective(self) -> IntruderObjective:
        """The party's collective objective."""
        if self._objective is None:
            return IntruderObjective.DESTROY_CORE
        return self._objective

    # ── Alive helpers ──────────────────────────────────────────────

    @property
    def alive_members(self) -> list[Intruder]:
        return [m for m in self.members if m.alive]

    @property
    def alive_count(self) -> int:
        return sum(1 for m in self.members if m.alive)

    @property
    def is_wiped(self) -> bool:
        return self.alive_count == 0

    # ── Map sharing ────────────────────────────────────────────────

    def share_maps(self) -> None:
        """Merge personal maps of alive members within MAP_SHARE_RANGE cells.

        Distance is Chebyshev (max of |dx|, |dy|, |dz|) so members in
        adjacent cells share instantly.
        """
        alive = self.alive_members
        n = len(alive)
        if n < 2:
            return

        for i in range(n):
            for j in range(i + 1, n):
                a, b = alive[i], alive[j]
                dist = max(
                    abs(a.x - b.x), abs(a.y - b.y), abs(a.z - b.z),
                )
                if dist <= MAP_SHARE_RANGE:
                    a.personal_map.merge(b.personal_map)
                    b.personal_map.merge(a.personal_map)

    # ── Warden healing ─────────────────────────────────────────────

    def tick_warden_heal(self) -> list[tuple[Intruder, Intruder, int]]:
        """Perform warden healing if the tick counter is ready.

        Each alive warden heals the lowest-HP ally (not self) within
        MAP_SHARE_RANGE cells by WARDEN_HEAL_AMOUNT.

        Returns a list of (warden, patient, amount_healed) for event reporting.
        """
        self._heal_tick_counter += 1
        if self._heal_tick_counter < WARDEN_HEAL_INTERVAL:
            return []
        self._heal_tick_counter = 0

        heals: list[tuple[Intruder, Intruder, int]] = []
        alive = self.alive_members
        wardens = [m for m in alive if m.archetype.healer]
        if not wardens:
            return []

        for warden in wardens:
            # Find lowest-HP ally within range (not self)
            candidates = []
            for ally in alive:
                if ally.id == warden.id:
                    continue
                if ally.hp >= ally.max_hp:
                    continue  # Already full
                dist = max(
                    abs(warden.x - ally.x),
                    abs(warden.y - ally.y),
                    abs(warden.z - ally.z),
                )
                if dist <= MAP_SHARE_RANGE:
                    candidates.append(ally)

            if not candidates:
                continue

            # Heal lowest HP ally (break ties by id for determinism)
            patient = min(candidates, key=lambda m: (m.hp, m.id))
            heal_amount = min(WARDEN_HEAL_AMOUNT, patient.max_hp - patient.hp)
            patient.hp += heal_amount
            heals.append((warden, patient, heal_amount))

        return heals

    # ── Warden loyalty aura ────────────────────────────────────────

    def apply_warden_aura(self) -> None:
        """Apply the loyalty bonus from alive wardens to nearby allies.

        This resets loyalty_modifier for all members, then adds
        WARDEN_LOYALTY_BONUS for each warden within MAP_SHARE_RANGE.
        Loyalty modifiers from warden death penalty are preserved
        by applying them *after* the aura (handled externally).
        """
        alive = self.alive_members
        wardens = [m for m in alive if m.archetype.healer]

        for m in alive:
            # Reset aura (we recalculate each call)
            base_penalty = min(0.0, m.loyalty_modifier)  # preserve death penalty
            m.loyalty_modifier = base_penalty

        for warden in wardens:
            for ally in alive:
                if ally.id == warden.id:
                    continue
                dist = max(
                    abs(warden.x - ally.x),
                    abs(warden.y - ally.y),
                    abs(warden.z - ally.z),
                )
                if dist <= MAP_SHARE_RANGE:
                    ally.loyalty_modifier += WARDEN_LOYALTY_BONUS

    # ── Warden death penalty ───────────────────────────────────────

    def on_member_death(self, dead: Intruder) -> None:
        """Handle a member's death.  If it's a warden, penalise loyalty.

        Re-elects leader and re-votes objective after any death.
        """
        if dead.archetype.healer:
            for m in self.alive_members:
                m.loyalty_modifier -= WARDEN_DEATH_LOYALTY_PENALTY
        self._elect_leader()
        self._vote_objective()

    # ── Betrayal ───────────────────────────────────────────────────

    def check_betrayals(
        self,
        treasure_adjacent: dict[int, bool],
        rng: SeededRNG,
    ) -> list[Intruder]:
        """Check each member for betrayal.

        *treasure_adjacent* maps intruder id → whether that intruder is
        adjacent to a treasure cell.

        Betrayal chance per tick:
            greed × (1 - effective_loyalty) × treasure_factor
        where treasure_factor is 1.0 if adjacent to treasure, else 0.0.

        On betrayal: the intruder leaves the party, switches objective to
        PILLAGE, and its party_id is cleared.

        Returns a list of intruders who betrayed this tick.
        """
        betrayers: list[Intruder] = []
        alive = self.alive_members

        for m in alive:
            if m.archetype.greed <= 0.0:
                continue
            if not treasure_adjacent.get(m.id, False):
                continue

            chance = m.archetype.greed * (1.0 - m.effective_loyalty)
            if rng.random() < chance:
                betrayers.append(m)

        # Process betrayals
        for m in betrayers:
            m.objective = IntruderObjective.PILLAGE
            m.party_id = None
            self.members.remove(m)

        if betrayers:
            self._elect_leader()
            self._vote_objective()

        return betrayers

    # ── Dunder ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.members)

    def __repr__(self) -> str:
        alive = self.alive_count
        return (
            f"Party(id={self.id}, members={len(self.members)}, "
            f"alive={alive}, objective={self.objective.name})"
        )
