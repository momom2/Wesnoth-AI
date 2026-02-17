"""Intruder AI: party spawning, per-intruder state machine, vision, pathfinding.

This is the main integration module that ties together archetypes, parties,
personal maps, vision, personal pathfinding, and block interactions into a
tick-driven simulation.  Each tick:

1. Parties are spawned at surface edges (if the timer is ready).
2. Each party shares maps, heals via wardens, checks betrayals.
3. Each alive intruder updates its state machine (vision -> decision -> move).
4. Dead/escaped intruders and wiped parties are cleaned up.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dungeon_builder.intruders.agent import Intruder, IntruderState
from dungeon_builder.intruders.archetypes import (
    IntruderObjective,
    IntruderStatus,
    VANGUARD,
)
from dungeon_builder.intruders.personal_map import PersonalMap
from dungeon_builder.intruders.personal_pathfinder import PersonalPathfinder
from dungeon_builder.intruders.vision import (
    compute_los,
    compute_arcane_sight,
    compute_thermal_vision,
)
from dungeon_builder.intruders.interactions import (
    handle_block,
    InteractionResult,
)
from dungeon_builder.intruders.party import (
    Party,
    choose_template,
    choose_underworld_template,
    generate_composition,
)
from dungeon_builder.intruders.knowledge_archive import KnowledgeArchive
from dungeon_builder.intruders.reputation import DungeonReputation
import dungeon_builder.config as _cfg
from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_STONE,
    VOXEL_TREASURE,
    VOXEL_DOOR,
    VOXEL_SPIKE,
    VOXEL_LAVA,
    VOXEL_TARP,
    VOXEL_GOLD_BAIT,
    VOXEL_HEAT_BEACON,
    VOXEL_PRESSURE_PLATE,
    VOXEL_IRON_BARS,
    VOXEL_FLOODGATE,
    VOXEL_ALARM_BELL,
    VOXEL_FRAGILE_FLOOR,
    VOXEL_PIPE,
    VOXEL_PUMP,
    VOXEL_STEAM_VENT,
    SURFACE_Z,
    DIG_DURATION,
    NON_DIGGABLE,
    PYREMANCER_HEAT_AMOUNT,
    PYREMANCER_HEAT_INTERVAL,
    GORECLAW_FRENZY_RANDOM_CHANCE,
    UNDERWORLD_SPAWN_INTERVAL,
    MAX_UNDERWORLD_PARTIES,
    MAX_UNDERWORLDERS_TOTAL,
    UNDERWORLD_SPAWN_Z_MIN,
    UNDERWORLD_SPAWN_Z_MAX,
    MAGMAWRAITH_HEAT_AMOUNT,
    MAGMAWRAITH_HEAT_INTERVAL,
    BOREMITE_DIG_DIVISOR,
    CORROSIVE_DAMAGE_FACTOR,
    LEVEL_WEIGHTS,
    MORALE_DAMAGE_PENALTY,
    MORALE_HAZARD_PENALTY,
    MORALE_TREASURE_BONUS,
    MORALE_LOW_THRESHOLD,
    MORALE_RETREAT_MULTIPLIER,
    FACTION_ENCOUNTER_INTERVAL,
    PRESSURE_PLATE_TRIGGER_RANGE,
    ALARM_BELL_DETECTION_RANGE,
    ALARM_BELL_COOLDOWN,
    FRAGILE_FLOOR_WEIGHT_THRESHOLD,
)

_HAZARD_TYPES = frozenset((
    VOXEL_SPIKE, VOXEL_LAVA, VOXEL_TARP,
    VOXEL_PRESSURE_PLATE, VOXEL_STEAM_VENT, VOXEL_FRAGILE_FLOOR,
))

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus
    from dungeon_builder.world.voxel_grid import VoxelGrid
    from dungeon_builder.world.pathfinding import AStarPathfinder
    from dungeon_builder.dungeon_core.core import DungeonCore
    from dungeon_builder.utils.rng import SeededRNG

logger = logging.getLogger("dungeon_builder.intruders")


class IntruderAI:
    """Manages intruder parties, spawning, movement, and decision-making.

    Public interface (kept compatible with main.py):
    - Constructor takes (event_bus, voxel_grid, pathfinder, core, rng)
    - ``intruders`` list contains all intruder agents
    - ``spawning_enabled`` flag controls whether parties are spawned
    - Subscribes to ``tick``, ``intruder_needs_repath``, ``game_over``
    """

    def __init__(
        self,
        event_bus: EventBus,
        voxel_grid: VoxelGrid,
        pathfinder: AStarPathfinder,
        core: DungeonCore,
        rng: SeededRNG,
    ) -> None:
        self.event_bus = event_bus
        self.voxel_grid = voxel_grid
        self.pathfinder = pathfinder  # Global pathfinder (fallback)
        self.core = core
        self.rng = rng

        self.intruders: list[Intruder] = []
        self.parties: list[Party] = []
        self._underworld_parties: list[Party] = []
        self._next_id = 1
        self._next_party_id = 1
        self._spawn_timer = 0
        self._underworld_spawn_timer = 0

        # Social dynamics systems
        self._knowledge_archive = KnowledgeArchive()
        self._reputation = DungeonReputation(event_bus)

        # Alarm bell cooldown tracking: pos → remaining ticks
        self._alarm_cooldowns: dict[tuple[int, int, int], int] = {}

        event_bus.subscribe("tick", self._on_tick)
        event_bus.subscribe("intruder_needs_repath", self._on_needs_repath)
        event_bus.subscribe("game_over", self._on_game_over)
        event_bus.subscribe("voxel_changed", self._on_voxel_changed)
        event_bus.subscribe("debug_spawn_party", self._on_debug_spawn_party)
        event_bus.subscribe("debug_spawn_underworld_party", self._on_debug_spawn_uw)

        self._game_over = False
        self.spawning_enabled = False

    # ── Event handlers ─────────────────────────────────────────────

    def _on_game_over(self, **kwargs) -> None:
        self._game_over = True

    def _on_tick(self, tick: int) -> None:
        if self._game_over:
            return

        if self.spawning_enabled:
            self._tick_spawning()
            self._tick_underworld_spawning()

        # Party-level updates (surface + underworld)
        for party in list(self.parties) + list(self._underworld_parties):
            if party.is_wiped:
                continue
            party.share_maps()
            party.apply_warden_aura()
            party.update_morale(tick)
            heals = party.tick_warden_heal()
            for warden, patient, amount in heals:
                logger.debug(
                    "Warden #%d heals #%d for %d HP",
                    warden.id, patient.id, amount,
                )
            self._tick_betrayals(party)

        # Alarm bell cooldown decrement
        if self._alarm_cooldowns:
            self._tick_alarm_cooldowns()

        # Per-intruder update
        for intruder in self.intruders:
            if not intruder.alive:
                continue
            self._update_intruder(intruder, tick)

        # TODO: Overhaul faction dynamics — current implementation is
        # hostile-only. Future work should add alliance, avoidance, and
        # negotiation mechanics.
        self._tick_faction_encounters(tick)

        # Periodic cleanup
        if tick % 100 == 0:
            self._cleanup()

    def _on_needs_repath(self, intruder: Intruder, **kwargs) -> None:
        self._repath_intruder(intruder)

    def _on_voxel_changed(self, x: int = 0, y: int = 0, z: int = 0, **kwargs) -> None:
        """When the player modifies a cell, increase uncertainty in faction archives."""
        self._knowledge_archive.on_voxel_changed(x, y, z)

    def _on_debug_spawn_party(self, **kwargs) -> None:
        """Debug: immediately spawn a surface intruder party."""
        self.spawning_enabled = True
        self._spawn_party()

    def _on_debug_spawn_uw(self, **kwargs) -> None:
        """Debug: immediately spawn an underworld party."""
        self.spawning_enabled = True
        self._spawn_underworld_party()

    # ── Spawning ───────────────────────────────────────────────────

    def _tick_spawning(self) -> None:
        self._spawn_timer += 1
        if self._spawn_timer < _cfg.INTRUDER_PARTY_SPAWN_INTERVAL:
            return
        self._spawn_timer = 0

        alive_count = sum(1 for i in self.intruders if i.alive)
        active_parties = sum(1 for p in self.parties if not p.is_wiped)

        if alive_count >= _cfg.MAX_INTRUDERS_TOTAL:
            return
        if active_parties >= _cfg.MAX_PARTIES:
            return

        self._spawn_party()

    def _spawn_party(self) -> None:
        """Spawn a new party at a random surface edge."""
        party_rng = self.rng.fork(f"party_{self._next_party_id}")
        template = choose_template(party_rng)
        composition = generate_composition(template, party_rng)

        # Find spawn position
        spawn_pos = self._find_spawn_position()
        if spawn_pos is None:
            logger.debug("No valid spawn position found for party")
            return

        sx, sy, sz = spawn_pos
        members: list[Intruder] = []
        loyalty_mod = self._reputation.get_loyalty_modifier()
        level_shift = self._reputation.get_level_shift()

        for arch in composition:
            obj = self._pick_objective(arch, party_rng)
            level = self._assign_level(party_rng, level_shift)
            status = self._level_to_status(level)
            pmap = PersonalMap()
            self._knowledge_archive.inject_knowledge(
                pmap, False, self._spawn_timer, arch.cunning,
            )
            intruder = Intruder(
                intruder_id=self._next_id,
                x=sx, y=sy, z=sz,
                archetype=arch,
                objective=obj,
                personal_map=pmap,
                level=level,
                status=status,
            )
            intruder.loyalty_modifier += loyalty_mod
            self._next_id += 1
            intruder.state = IntruderState.ADVANCING
            members.append(intruder)

        if not members:
            return

        rep_modifier = self._reputation.get_objective_modifier()
        party = Party(self._next_party_id, members)
        party._vote_objective(reputation_modifier=rep_modifier)
        self._next_party_id += 1
        self.parties.append(party)

        for m in members:
            self.intruders.append(m)
            # Initial vision scan
            self._update_vision(m)
            # Initial path
            self._repath_intruder(m)
            self.event_bus.publish("intruder_spawned", intruder=m)

        logger.info(
            "Spawned party #%d (%s) with %d members at (%d,%d,%d)",
            party.id, template.name, len(members), sx, sy, sz,
        )

    def _spawn_intruder(self) -> None:
        """Spawn a single intruder (legacy compatibility for tests)."""
        spawn_pos = self._find_spawn_position()
        if spawn_pos is None:
            logger.debug("No valid spawn position found")
            return

        sx, sy, sz = spawn_pos
        intruder = Intruder(
            intruder_id=self._next_id,
            x=sx, y=sy, z=sz,
            archetype=VANGUARD,
            objective=IntruderObjective.DESTROY_CORE,
            personal_map=PersonalMap(),
        )
        self._next_id += 1

        # Find path via global pathfinder
        path = self.pathfinder.find_path(
            (intruder.x, intruder.y, intruder.z),
            (self.core.x, self.core.y, self.core.z),
        )
        if path is None:
            logger.debug("No path from spawn to core for intruder #%d", intruder.id)
            return

        intruder.path = path
        intruder.path_index = 1
        intruder.state = IntruderState.ADVANCING
        self.intruders.append(intruder)
        self.event_bus.publish("intruder_spawned", intruder=intruder)
        logger.info("Spawned intruder #%d at (%d, %d, %d)", intruder.id, sx, sy, sz)

    @staticmethod
    def _pick_objective(arch, rng) -> IntruderObjective:
        """Pick an objective weighted by archetype's objective_weights."""
        w = arch.objective_weights
        roll = rng.random()
        if roll < w[0]:
            return IntruderObjective.DESTROY_CORE
        if roll < w[0] + w[1]:
            return IntruderObjective.EXPLORE
        return IntruderObjective.PILLAGE

    def _find_spawn_position(self) -> tuple[int, int, int] | None:
        """Find an air cell on Z=0 at or near the map edge."""
        grid = self.voxel_grid
        edges: list[tuple[int, int, int]] = []
        for x in range(grid.width):
            for y in [0, grid.depth - 1]:
                if grid.get(x, y, SURFACE_Z) == VOXEL_AIR:
                    edges.append((x, y, SURFACE_Z))
        for y in range(grid.depth):
            for x in [0, grid.width - 1]:
                if grid.get(x, y, SURFACE_Z) == VOXEL_AIR:
                    edges.append((x, y, SURFACE_Z))

        if not edges:
            for x in range(grid.width):
                for y in range(grid.depth):
                    if grid.get(x, y, SURFACE_Z) == VOXEL_AIR:
                        edges.append((x, y, SURFACE_Z))

        if not edges:
            return None
        return self.rng.choice(edges)

    # ── Underworld spawning ──────────────────────────────────────

    def _tick_underworld_spawning(self) -> None:
        self._underworld_spawn_timer += 1
        if self._underworld_spawn_timer < UNDERWORLD_SPAWN_INTERVAL:
            return
        self._underworld_spawn_timer = 0

        alive_uw = sum(1 for i in self.intruders if i.alive and i.is_underworlder)
        active_uw = sum(1 for p in self._underworld_parties if not p.is_wiped)

        if alive_uw >= MAX_UNDERWORLDERS_TOTAL:
            return
        if active_uw >= MAX_UNDERWORLD_PARTIES:
            return

        self._spawn_underworld_party()

    def _spawn_underworld_party(self) -> None:
        """Spawn a new underworld party at a deep map edge."""
        party_rng = self.rng.fork(f"uw_party_{self._next_party_id}")
        template = choose_underworld_template(party_rng)
        composition = generate_composition(template, party_rng)

        spawn_pos = self._find_underworld_spawn_position()
        if spawn_pos is None:
            logger.debug("No valid underworld spawn position found")
            return

        sx, sy, sz = spawn_pos
        members: list[Intruder] = []
        loyalty_mod = self._reputation.get_loyalty_modifier()
        level_shift = self._reputation.get_level_shift()

        for arch in composition:
            obj = self._pick_objective(arch, party_rng)
            level = self._assign_level(party_rng, level_shift)
            status = self._level_to_status(level)
            pmap = PersonalMap()
            self._knowledge_archive.inject_knowledge(
                pmap, True, self._underworld_spawn_timer, arch.cunning,
            )
            intruder = Intruder(
                intruder_id=self._next_id,
                x=sx, y=sy, z=sz,
                archetype=arch,
                objective=obj,
                personal_map=pmap,
                is_underworlder=True,
                level=level,
                status=status,
            )
            intruder.loyalty_modifier += loyalty_mod
            self._next_id += 1
            intruder.state = IntruderState.ADVANCING
            members.append(intruder)

        if not members:
            return

        rep_modifier = self._reputation.get_objective_modifier()
        party = Party(self._next_party_id, members)
        party._vote_objective(reputation_modifier=rep_modifier)
        self._next_party_id += 1
        self._underworld_parties.append(party)

        for m in members:
            self.intruders.append(m)
            self._update_vision(m)
            self._repath_intruder(m)
            self.event_bus.publish("intruder_spawned", intruder=m)

        logger.info(
            "Spawned underworld party #%d (%s) with %d members at (%d,%d,%d)",
            party.id, template.name, len(members), sx, sy, sz,
        )

    def _find_underworld_spawn_position(self) -> tuple[int, int, int] | None:
        """Find an air cell on a map edge at a deep z-level."""
        grid = self.voxel_grid
        target_z = self.rng.randint(UNDERWORLD_SPAWN_Z_MIN, UNDERWORLD_SPAWN_Z_MAX)

        # Try target z, then ±1, ±2
        for dz in (0, 1, -1, 2, -2):
            z = target_z + dz
            if z < UNDERWORLD_SPAWN_Z_MIN or z > UNDERWORLD_SPAWN_Z_MAX:
                continue
            edges: list[tuple[int, int, int]] = []
            for x in range(grid.width):
                for y_edge in (0, grid.depth - 1):
                    if grid.get(x, y_edge, z) == VOXEL_AIR:
                        edges.append((x, y_edge, z))
            for y in range(grid.depth):
                for x_edge in (0, grid.width - 1):
                    if grid.get(x_edge, y, z) == VOXEL_AIR:
                        edges.append((x_edge, y, z))
            if edges:
                return self.rng.choice(edges)

        # Fallback: carve into a diggable solid cell at the edge
        z = target_z
        solid_edges: list[tuple[int, int, int]] = []
        for x in range(grid.width):
            for y_edge in (0, grid.depth - 1):
                vtype = grid.get(x, y_edge, z)
                if vtype != VOXEL_AIR and vtype not in NON_DIGGABLE:
                    solid_edges.append((x, y_edge, z))
        for y in range(grid.depth):
            for x_edge in (0, grid.width - 1):
                vtype = grid.get(x_edge, y, z)
                if vtype != VOXEL_AIR and vtype not in NON_DIGGABLE:
                    solid_edges.append((x_edge, y, z))
        if solid_edges:
            pos = self.rng.choice(solid_edges)
            grid.set(pos[0], pos[1], pos[2], VOXEL_AIR)
            return pos

        return None

    # ── Per-intruder state machine ─────────────────────────────────

    def _update_intruder(self, intruder: Intruder, tick: int = 0) -> None:
        """Main per-tick update for a single intruder."""
        # 1. Vision update
        self._update_vision(intruder)

        # 2. Check frenzy activation
        self._check_frenzy(intruder)

        # 3. State-specific behavior
        state = intruder.state
        if state == IntruderState.SPAWNING:
            intruder.state = IntruderState.ADVANCING
            self._repath_intruder(intruder)
        elif state == IntruderState.ADVANCING:
            self._update_advancing(intruder, tick)
        elif state == IntruderState.INTERACTING:
            self._update_interacting(intruder)
        elif state == IntruderState.ATTACKING:
            self._update_attacking(intruder)
        elif state == IntruderState.RETREATING:
            self._update_retreating(intruder, tick)
        elif state == IntruderState.PILLAGING:
            self._update_pillaging(intruder)

    # ── Vision ─────────────────────────────────────────────────────

    def _update_vision(self, intruder: Intruder) -> None:
        """Update the intruder's personal map from current LOS + special vision.

        Only recomputes when the intruder has moved since the last vision update
        (_vision_dirty flag), avoiding expensive ray-casting on stationary ticks.
        """
        if not intruder._vision_dirty:
            return
        intruder._vision_dirty = False

        grid = self.voxel_grid
        arch = intruder.archetype
        x, y, z = intruder.x, intruder.y, intruder.z
        pmap = intruder.personal_map

        # Track hazards known before vision update (for morale penalty)
        hazards_before = len(pmap.hazards)

        # Standard LOS (with vision deception for certain blocks)
        visible = compute_los(grid, x, y, z, arch.perception_range)
        for vx, vy, vz in visible:
            vtype = grid.get(vx, vy, vz)
            bstate = int(grid.block_state[vx, vy, vz])
            # Vision deception: Gold Bait looks like Treasure to normal sight
            if vtype == VOXEL_GOLD_BAIT:
                pmap.reveal(vx, vy, vz, VOXEL_TREASURE, bstate)
            # Vision deception: Fragile Floor looks like Stone to normal sight
            elif vtype == VOXEL_FRAGILE_FLOOR:
                pmap.reveal(vx, vy, vz, VOXEL_STONE, bstate)
            else:
                pmap.reveal(vx, vy, vz, vtype, bstate)

        # Arcane sight (Gloomseer) — sees true types through walls
        if arch.arcane_sight_range > 0:
            arcane = compute_arcane_sight(grid, x, y, z, arch.arcane_sight_range)
            for vx, vy, vz in arcane:
                vtype = grid.get(vx, vy, vz)
                bstate = int(grid.block_state[vx, vy, vz])
                # Arcane sight sees true types (no deception)
                pmap.reveal(vx, vy, vz, vtype, bstate)
                # Additionally mark baits and hazards for true types
                if vtype == VOXEL_GOLD_BAIT:
                    pmap.mark_bait(vx, vy, vz)
                elif vtype == VOXEL_FRAGILE_FLOOR:
                    pmap.mark_hazard(vx, vy, vz)

        # Thermal vision (Pyremancer)
        if arch.fire_immune and arch.perception_range >= 4:
            thermal = compute_thermal_vision(grid, x, y, z, 4)
            for vx, vy, vz in thermal:
                vtype = grid.get(vx, vy, vz)
                bstate = int(grid.block_state[vx, vy, vz])
                pmap.reveal(vx, vy, vz, vtype, bstate)

        # Morale penalty for newly revealed hazards
        new_hazards = len(pmap.hazards) - hazards_before
        if new_hazards > 0:
            intruder.morale = max(
                0.0, intruder.morale - MORALE_HAZARD_PENALTY * new_hazards,
            )

    # ── Frenzy ─────────────────────────────────────────────────────

    @staticmethod
    def _check_frenzy(intruder: Intruder) -> None:
        arch = intruder.archetype
        if arch.frenzy_threshold <= 0:
            return
        hp_ratio = intruder.hp / intruder.max_hp
        if hp_ratio <= arch.frenzy_threshold and not intruder.frenzy_active:
            intruder.frenzy_active = True
            logger.info("Intruder #%d enters frenzy!", intruder.id)

    # ── ADVANCING state ────────────────────────────────────────────

    def _update_advancing(self, intruder: Intruder, tick: int = 0) -> None:
        # Check retreat condition
        self._check_retreat(intruder)
        if intruder.state != IntruderState.ADVANCING:
            return

        # Heat adjacent blocks periodically (Pyremancer / Magmawraith)
        self._tick_pyremancer_heat(intruder, tick)
        self._tick_magmawraith_heat(intruder, tick)

        # Movement tick
        intruder.ticks_since_move += 1
        if intruder.ticks_since_move < intruder.effective_move_interval:
            return
        intruder.ticks_since_move = 0

        # Frenzy random movement
        if intruder.frenzy_active:
            if self.rng.random() < GORECLAW_FRENZY_RANDOM_CHANCE:
                self._move_random(intruder)
                return

        self._advance_along_path(intruder)

        # Check if we reached the core (within attack range)
        core_pos = (self.core.x, self.core.y, self.core.z)
        dist = (
            abs(intruder.x - core_pos[0])
            + abs(intruder.y - core_pos[1])
            + abs(intruder.z - core_pos[2])
        )
        if dist <= intruder.archetype.attack_range:
            intruder.state = IntruderState.ATTACKING
            logger.info("Intruder #%d reached attack range of core!", intruder.id)

    # ── INTERACTING state ──────────────────────────────────────────

    def _update_interacting(self, intruder: Intruder) -> None:
        """Count down the interaction timer; complete when done."""
        intruder.interaction_ticks -= 1
        if intruder.interaction_ticks > 0:
            return

        itype = intruder.interaction_type
        target = intruder.interaction_target

        if target is None:
            intruder.state = IntruderState.ADVANCING
            return

        tx, ty, tz = target
        grid = self.voxel_grid

        if itype in ("bash_door", "lockpick"):
            # Open the door (set block_state to 0)
            if grid.get(tx, ty, tz) == VOXEL_DOOR:
                grid.set_block_state(tx, ty, tz, 0)
                intruder.personal_map.reveal(tx, ty, tz, VOXEL_DOOR, 0)
                logger.debug(
                    "Intruder #%d opened door at (%d,%d,%d)",
                    intruder.id, tx, ty, tz,
                )
                self.event_bus.publish(
                    "intruder_opened_door",
                    intruder=intruder, x=tx, y=ty, z=tz,
                )
        elif itype == "grab_treasure":
            if grid.get(tx, ty, tz) == VOXEL_TREASURE:
                grid.set(tx, ty, tz, VOXEL_AIR)
                intruder.loot_count += 1
                intruder.personal_map.remove_treasure(tx, ty, tz)
                intruder.morale = min(1.0, intruder.morale + MORALE_TREASURE_BONUS)
                logger.debug(
                    "Intruder #%d collected treasure at (%d,%d,%d)",
                    intruder.id, tx, ty, tz,
                )
                self.event_bus.publish(
                    "intruder_collected_treasure",
                    intruder=intruder, x=tx, y=ty, z=tz,
                )
        elif itype == "grab_bait":
            if grid.get(tx, ty, tz) == VOXEL_GOLD_BAIT:
                # Bait consumed
                grid.set(tx, ty, tz, VOXEL_AIR)
                # Intruder realizes they were tricked — morale hit
                intruder.morale = max(0.0, intruder.morale - 0.1)
                intruder.personal_map.mark_bait(tx, ty, tz)
                logger.debug(
                    "Intruder #%d grabbed gold bait at (%d,%d,%d)",
                    intruder.id, tx, ty, tz,
                )
                self.event_bus.publish(
                    "intruder_grabbed_bait",
                    intruder=intruder, x=tx, y=ty, z=tz,
                )
                # Bait triggers adjacent traps (same as pressure plate)
                self._activate_pressure_plate(tx, ty, tz, intruder)

        elif itype == "dig":
            vtype = grid.get(tx, ty, tz)
            if vtype != VOXEL_AIR and vtype not in NON_DIGGABLE:
                grid.set(tx, ty, tz, VOXEL_AIR)
                intruder.personal_map.reveal(tx, ty, tz, VOXEL_AIR, 0)
                # Corrosive Crawler: weaken adjacent blocks
                if intruder.archetype.name == "Corrosive Crawler":
                    self._apply_corrosive_damage(tx, ty, tz)
                logger.debug(
                    "Intruder #%d dug through (%d,%d,%d)",
                    intruder.id, tx, ty, tz,
                )
                self.event_bus.publish(
                    "intruder_digging",
                    intruder=intruder, x=tx, y=ty, z=tz,
                )

        # Clear interaction state and resume
        intruder.interaction_type = None
        intruder.interaction_target = None
        intruder.interaction_ticks = 0

        if intruder.objective == IntruderObjective.PILLAGE:
            intruder.state = IntruderState.PILLAGING
        else:
            intruder.state = IntruderState.ADVANCING
        self._repath_intruder(intruder)

    # ── ATTACKING state ────────────────────────────────────────────

    def _update_attacking(self, intruder: Intruder) -> None:
        intruder.ticks_since_attack += 1
        if intruder.ticks_since_attack >= intruder.attack_interval:
            intruder.ticks_since_attack = 0
            damage = intruder.effective_damage
            self.core.take_damage(damage)
            logger.debug(
                "Intruder #%d attacks core for %d damage",
                intruder.id, damage,
            )

    # ── RETREATING state ───────────────────────────────────────────

    def _update_retreating(self, intruder: Intruder, tick: int = 0) -> None:
        intruder.ticks_since_move += 1
        if intruder.ticks_since_move < intruder.effective_move_interval:
            return
        intruder.ticks_since_move = 0

        self._advance_along_path(intruder)

        if intruder.is_underworlder:
            if self._is_at_deep_edge(intruder):
                intruder.state = IntruderState.ESCAPED
                self._knowledge_archive.archive_survivor(intruder, tick)
                self.event_bus.publish("intruder_escaped", intruder=intruder)
                logger.info("Underworlder #%d escaped underground!", intruder.id)
        else:
            if intruder.z == SURFACE_Z:
                intruder.state = IntruderState.ESCAPED
                self._knowledge_archive.archive_survivor(intruder, tick)
                self.event_bus.publish("intruder_escaped", intruder=intruder)
                logger.info("Intruder #%d escaped to the surface!", intruder.id)

    # ── PILLAGING state ────────────────────────────────────────────

    def _update_pillaging(self, intruder: Intruder) -> None:
        """Head toward nearest known treasure, or retreat if none left."""
        self._check_retreat(intruder)
        if intruder.state != IntruderState.PILLAGING:
            return

        intruder.ticks_since_move += 1
        if intruder.ticks_since_move < intruder.effective_move_interval:
            return
        intruder.ticks_since_move = 0

        # If no path or path exhausted, find nearest treasure
        if intruder.path is None or intruder.path_index >= len(intruder.path):
            treasures = list(intruder.personal_map.treasures)
            if not treasures:
                self._start_retreat(intruder)
                return
            nearest = min(
                treasures,
                key=lambda t: (
                    abs(t[0] - intruder.x)
                    + abs(t[1] - intruder.y)
                    + abs(t[2] - intruder.z)
                ),
            )
            path = PersonalPathfinder.find_path(
                intruder.personal_map, intruder.pos, nearest, intruder.archetype,
            )
            if path and len(path) > 1:
                intruder.path = path
                intruder.path_index = 1
            else:
                self._start_retreat(intruder)
                return

        self._advance_along_path(intruder)

    # ── Path following & block interactions ─────────────────────────

    def _advance_along_path(self, intruder: Intruder) -> None:
        """Move the intruder one step along its path, handling interactions."""
        if intruder.path is None:
            return
        if intruder.path_index >= len(intruder.path):
            if intruder.state == IntruderState.ADVANCING:
                self._repath_intruder(intruder)
            return

        next_pos = intruder.path[intruder.path_index]
        nx, ny, nz = next_pos
        grid = self.voxel_grid

        if not grid.in_bounds(nx, ny, nz):
            self._repath_intruder(intruder)
            return

        vtype = grid.get(nx, ny, nz)
        bstate = int(grid.block_state[nx, ny, nz])

        # Handle block interaction
        info = handle_block(intruder, vtype, bstate)

        if info.result == InteractionResult.CONTINUE:
            self._move_to(intruder, next_pos)

        elif info.result == InteractionResult.INTERACT:
            intruder.state = IntruderState.INTERACTING
            intruder.interaction_type = info.interaction_type
            intruder.interaction_target = next_pos
            intruder.interaction_ticks = info.ticks

        elif info.result == InteractionResult.DAMAGE:
            intruder.take_damage(info.damage)
            intruder.morale = max(0.0, intruder.morale - MORALE_DAMAGE_PENALTY)
            if intruder.alive:
                self._move_to(intruder, next_pos)
            else:
                self._on_intruder_death(intruder)

        elif info.result == InteractionResult.REPATH:
            if (
                intruder.archetype.can_dig
                and vtype not in NON_DIGGABLE
                and vtype != VOXEL_AIR
            ):
                self._start_digging(intruder, next_pos, vtype)
            else:
                self._repath_intruder(intruder)

        elif info.result == InteractionResult.COLLECT:
            intruder.state = IntruderState.INTERACTING
            intruder.interaction_type = info.interaction_type
            intruder.interaction_target = next_pos
            intruder.interaction_ticks = info.ticks

        elif info.result == InteractionResult.FALL:
            intruder.personal_map.mark_hazard(nx, ny, nz)
            grid.set(nx, ny, nz, VOXEL_AIR)
            self._move_to(intruder, next_pos)
            self.event_bus.publish(
                "intruder_fell", intruder=intruder, x=nx, y=ny, z=nz,
            )

        elif info.result == InteractionResult.DEATH:
            intruder.take_damage(intruder.hp)
            self._on_intruder_death(intruder)

        elif info.result == InteractionResult.DESTROY_BLOCK:
            intruder.take_damage(info.damage)
            if grid.get(nx, ny, nz) != VOXEL_AIR:
                grid.set(nx, ny, nz, VOXEL_AIR)
            if intruder.alive:
                self._move_to(intruder, next_pos)
            else:
                self._on_intruder_death(intruder)

    def _move_to(self, intruder: Intruder, pos: tuple[int, int, int]) -> None:
        """Move intruder to a new position and advance path index."""
        intruder.x, intruder.y, intruder.z = pos
        intruder.path_index += 1
        intruder._vision_dirty = True
        self.event_bus.publish("intruder_moved", intruder=intruder)

        # Post-move triggers on the cell we just stepped into
        grid = self.voxel_grid
        nx, ny, nz = pos
        if grid.in_bounds(nx, ny, nz):
            vtype = grid.get(nx, ny, nz)

            # Pressure plate activation
            if vtype == VOXEL_PRESSURE_PLATE:
                bstate = int(grid.block_state[nx, ny, nz])
                if bstate == 0:  # Not yet triggered
                    grid.set_block_state(nx, ny, nz, 1)
                    self._activate_pressure_plate(nx, ny, nz, intruder)

            # Fragile floor collapse check (flyers don't trigger)
            if vtype == VOXEL_FRAGILE_FLOOR and not intruder.archetype.can_fly:
                collapsed = self._check_fragile_floor(intruder, nx, ny, nz)
                if collapsed:
                    # Intruder falls — check if there's air below
                    below_z = nz + 1  # z+1 = deeper
                    if (
                        grid.in_bounds(nx, ny, below_z)
                        and grid.get(nx, ny, below_z) == VOXEL_AIR
                    ):
                        intruder.z = below_z
                        intruder._vision_dirty = True
                        self.event_bus.publish(
                            "intruder_fell",
                            intruder=intruder, x=nx, y=ny, z=nz,
                        )

            # Alarm bell proximity check
            self._check_alarm_bells(intruder)

    def _move_random(self, intruder: Intruder) -> None:
        """Move the intruder to a random adjacent air cell (frenzy)."""
        grid = self.voxel_grid
        candidates = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = intruder.x + dx, intruder.y + dy
            nz = intruder.z
            if grid.in_bounds(nx, ny, nz) and grid.get(nx, ny, nz) == VOXEL_AIR:
                candidates.append((nx, ny, nz))
        if candidates:
            pos = self.rng.choice(candidates)
            intruder.x, intruder.y, intruder.z = pos
            intruder._vision_dirty = True
            self.event_bus.publish("intruder_moved", intruder=intruder)

    # ── Digging ────────────────────────────────────────────────────

    def _start_digging(
        self,
        intruder: Intruder,
        target: tuple[int, int, int],
        vtype: int,
    ) -> None:
        """Start digging through a solid block.

        Boremite digs at 1/3 base duration, others at 1/2.
        """
        base_ticks = DIG_DURATION.get(vtype, 40)
        if intruder.archetype.name == "Boremite":
            dig_ticks = max(1, base_ticks // BOREMITE_DIG_DIVISOR)
        else:
            dig_ticks = max(1, base_ticks // 2)

        intruder.state = IntruderState.INTERACTING
        intruder.interaction_type = "dig"
        intruder.interaction_target = target
        intruder.interaction_ticks = dig_ticks
        self.event_bus.publish(
            "intruder_digging",
            intruder=intruder, x=target[0], y=target[1], z=target[2],
        )

    # ── Retreat ────────────────────────────────────────────────────

    @staticmethod
    def _check_retreat(intruder: Intruder) -> None:
        arch = intruder.archetype
        if arch.never_retreats:
            return

        # Morale-based flee: very low morale → abandon party
        if intruder.morale < _cfg.MORALE_FLEE_THRESHOLD:
            intruder.state = IntruderState.RETREATING
            return

        if arch.retreat_threshold <= 0:
            return

        hp_ratio = intruder.hp / intruder.max_hp
        # Low morale doubles the retreat threshold (flee at higher HP)
        threshold = arch.retreat_threshold
        if intruder.morale < MORALE_LOW_THRESHOLD:
            threshold *= MORALE_RETREAT_MULTIPLIER

        if hp_ratio < threshold:
            intruder.state = IntruderState.RETREATING

    def _start_retreat(self, intruder: Intruder) -> None:
        if intruder.is_underworlder:
            self._start_underworld_retreat(intruder)
            return
        # Already at surface → escape immediately
        if intruder.z == SURFACE_Z:
            intruder.state = IntruderState.RETREATING
            return
        intruder.state = IntruderState.RETREATING
        path = PersonalPathfinder.find_path(
            intruder.personal_map,
            intruder.pos,
            (intruder.x, intruder.y, SURFACE_Z),
            intruder.archetype,
        )
        if path and len(path) > 1:
            intruder.path = path
            intruder.path_index = 1
        else:
            # Fallback to global pathfinder
            path = self.pathfinder.find_path(
                intruder.pos, (intruder.x, intruder.y, SURFACE_Z),
            )
            if path and len(path) > 1:
                intruder.path = path
                intruder.path_index = 1
            else:
                intruder.state = IntruderState.ADVANCING
        logger.info("Intruder #%d retreating (HP: %d)", intruder.id, intruder.hp)

    def _start_underworld_retreat(self, intruder: Intruder) -> None:
        """Underworlders retreat toward the nearest deep map edge."""
        # Already at deep edge → escape immediately
        if self._is_at_deep_edge(intruder):
            intruder.state = IntruderState.RETREATING
            return

        goal = self._find_underworld_retreat_goal(intruder)
        if goal is None:
            # No escape route — fight to the death
            intruder.state = IntruderState.ADVANCING
            return

        intruder.state = IntruderState.RETREATING
        path = PersonalPathfinder.find_path(
            intruder.personal_map, intruder.pos, goal, intruder.archetype,
        )
        if path and len(path) > 1:
            intruder.path = path
            intruder.path_index = 1
        else:
            path = self.pathfinder.find_path(intruder.pos, goal)
            if path and len(path) > 1:
                intruder.path = path
                intruder.path_index = 1
            else:
                intruder.state = IntruderState.ADVANCING
        logger.info(
            "Underworlder #%d retreating underground (HP: %d)",
            intruder.id, intruder.hp,
        )

    def _is_at_deep_edge(self, intruder: Intruder) -> bool:
        """Return True if the intruder is at a map edge at depth."""
        x, y, z = intruder.x, intruder.y, intruder.z
        if z < UNDERWORLD_SPAWN_Z_MIN:
            return False
        grid = self.voxel_grid
        return x == 0 or x == grid.width - 1 or y == 0 or y == grid.depth - 1

    def _find_underworld_retreat_goal(
        self, intruder: Intruder,
    ) -> tuple[int, int, int] | None:
        """Find the nearest air cell on a deep map edge for retreat."""
        x, y, z = intruder.x, intruder.y, intruder.z
        grid = self.voxel_grid

        edge_goals: list[tuple[int, int, int]] = []
        for test_z in range(
            max(UNDERWORLD_SPAWN_Z_MIN, z),
            min(UNDERWORLD_SPAWN_Z_MAX + 1, z + 5),
        ):
            for ex in range(grid.width):
                for ey_edge in (0, grid.depth - 1):
                    if grid.get(ex, ey_edge, test_z) == VOXEL_AIR:
                        edge_goals.append((ex, ey_edge, test_z))
            for ey in range(grid.depth):
                for ex_edge in (0, grid.width - 1):
                    if grid.get(ex_edge, ey, test_z) == VOXEL_AIR:
                        edge_goals.append((ex_edge, ey, test_z))

        if not edge_goals:
            return None

        # Pick nearest edge by Manhattan distance
        return min(
            edge_goals,
            key=lambda g: abs(g[0] - x) + abs(g[1] - y) + abs(g[2] - z),
        )

    # ── Pathing ────────────────────────────────────────────────────

    def _repath_intruder(self, intruder: Intruder) -> None:
        """Find a new path for the intruder based on its objective/state."""
        if intruder.state == IntruderState.RETREATING:
            if intruder.is_underworlder:
                goal = self._find_underworld_retreat_goal(intruder)
                if goal is None:
                    intruder.state = IntruderState.ADVANCING
                    return
            else:
                goal = (intruder.x, intruder.y, SURFACE_Z)
        elif intruder.state == IntruderState.PILLAGING:
            treasures = list(intruder.personal_map.treasures)
            if treasures:
                goal = min(
                    treasures,
                    key=lambda t: (
                        abs(t[0] - intruder.x)
                        + abs(t[1] - intruder.y)
                        + abs(t[2] - intruder.z)
                    ),
                )
            else:
                self._start_retreat(intruder)
                return
        elif intruder.state in (IntruderState.ADVANCING, IntruderState.SPAWNING):
            goal = (self.core.x, self.core.y, self.core.z)
        else:
            return

        # Skip repathing if position, goal, and map haven't changed
        cache_key = (intruder.pos, goal, intruder.personal_map._generation)
        if (
            intruder._path_cache_key == cache_key
            and intruder.path is not None
            and intruder.path_index < len(intruder.path)
        ):
            return

        # Try personal pathfinder first
        path = PersonalPathfinder.find_path(
            intruder.personal_map, intruder.pos, goal, intruder.archetype,
        )
        if path and len(path) > 1:
            intruder.path = path
            intruder.path_index = 1
            intruder._path_cache_key = cache_key
        else:
            # Fallback to global pathfinder
            path = self.pathfinder.find_path(intruder.pos, goal)
            if path and len(path) > 1:
                intruder.path = path
                intruder.path_index = 1
                intruder._path_cache_key = cache_key
            else:
                logger.debug(
                    "Intruder #%d cannot find path from %s to %s",
                    intruder.id, intruder.pos, goal,
                )
                intruder.path = None
                intruder._path_cache_key = None

    # ── Pyremancer heat ────────────────────────────────────────────

    def _tick_pyremancer_heat(self, intruder: Intruder, tick: int) -> None:
        """Pyremancers heat adjacent blocks periodically."""
        if not intruder.archetype.fire_immune:
            return
        if intruder.archetype.name != "Pyremancer":
            return
        if tick % PYREMANCER_HEAT_INTERVAL != 0:
            return

        grid = self.voxel_grid
        x, y, z = intruder.x, intruder.y, intruder.z
        for dx, dy, dz in (
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ):
            nx, ny, nz = x + dx, y + dy, z + dz
            if grid.in_bounds(nx, ny, nz):
                grid.temperature[nx, ny, nz] += PYREMANCER_HEAT_AMOUNT

    # ── Magmawraith heat ──────────────────────────────────────────

    def _tick_magmawraith_heat(self, intruder: Intruder, tick: int) -> None:
        """Magmawraiths heat adjacent blocks periodically."""
        if intruder.archetype.name != "Magmawraith":
            return
        if tick % MAGMAWRAITH_HEAT_INTERVAL != 0:
            return

        grid = self.voxel_grid
        x, y, z = intruder.x, intruder.y, intruder.z
        for dx, dy, dz in (
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ):
            nx, ny, nz = x + dx, y + dy, z + dz
            if grid.in_bounds(nx, ny, nz):
                grid.temperature[nx, ny, nz] += MAGMAWRAITH_HEAT_AMOUNT

    # ── Corrosive damage ────────────────────────────────────────

    def _apply_corrosive_damage(self, x: int, y: int, z: int) -> None:
        """Corrosive Crawler weakens adjacent blocks after digging."""
        grid = self.voxel_grid
        for dx, dy, dz in (
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ):
            nx, ny, nz = x + dx, y + dy, z + dz
            if (
                grid.in_bounds(nx, ny, nz)
                and grid.get(nx, ny, nz) != VOXEL_AIR
            ):
                grid.stress_ratio[nx, ny, nz] += CORROSIVE_DAMAGE_FACTOR

    # ── Pressure plate activation ─────────────────────────────────

    def _activate_pressure_plate(
        self, x: int, y: int, z: int, intruder: Intruder,
    ) -> None:
        """Activate a pressure plate and trigger adjacent traps.

        Within PRESSURE_PLATE_TRIGGER_RANGE, activates:
        - Spikes: set block_state = 1 (extended)
        - Doors: set block_state = 1 (closed)
        - Floodgates: toggle block_state (open↔closed)
        """
        grid = self.voxel_grid
        r = PRESSURE_PLATE_TRIGGER_RANGE
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if not grid.in_bounds(nx, ny, nz):
                        continue
                    vtype = grid.get(nx, ny, nz)
                    if vtype == VOXEL_SPIKE:
                        grid.set_block_state(nx, ny, nz, 1)  # Extend
                    elif vtype == VOXEL_DOOR:
                        grid.set_block_state(nx, ny, nz, 1)  # Close
                    elif vtype == VOXEL_FLOODGATE:
                        # Toggle: open→closed, closed→open
                        old = int(grid.block_state[nx, ny, nz])
                        grid.set_block_state(nx, ny, nz, 1 - old)
        self.event_bus.publish(
            "pressure_plate_activated",
            intruder=intruder, x=x, y=y, z=z,
        )

    # ── Fragile floor collapse ────────────────────────────────────

    def _check_fragile_floor(
        self, intruder: Intruder, x: int, y: int, z: int,
    ) -> bool:
        """Check and handle fragile floor collapse.

        Increments block_state each step. If >= FRAGILE_FLOOR_WEIGHT_THRESHOLD,
        the floor collapses to air and the intruder falls.

        Returns True if the floor collapsed (caller should handle the fall).
        """
        grid = self.voxel_grid
        if grid.get(x, y, z) != VOXEL_FRAGILE_FLOOR:
            return False

        current = int(grid.block_state[x, y, z])
        new_state = current + 1
        if new_state >= FRAGILE_FLOOR_WEIGHT_THRESHOLD:
            # Collapse!
            grid.set(x, y, z, VOXEL_AIR)
            intruder.personal_map.mark_hazard(x, y, z)
            self.event_bus.publish(
                "fragile_floor_collapsed",
                intruder=intruder, x=x, y=y, z=z,
            )
            return True
        else:
            grid.set_block_state(x, y, z, new_state)
            return False

    # ── Alarm bell detection ──────────────────────────────────────

    def _check_alarm_bells(self, intruder: Intruder) -> None:
        """Check if intruder is within range of any alarm bell.

        Alarm bells detect intruders within ALARM_BELL_DETECTION_RANGE
        (Manhattan distance) and publish an alarm event. Each bell has a
        cooldown (tracked in ``_alarm_cooldowns``) to prevent spam.
        """
        grid = self.voxel_grid
        ix, iy, iz = intruder.x, intruder.y, intruder.z
        r = ALARM_BELL_DETECTION_RANGE

        # Scan the area around the intruder for alarm bells
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if abs(dx) + abs(dy) + abs(dz) > r:
                        continue
                    bx, by, bz = ix + dx, iy + dy, iz + dz
                    if not grid.in_bounds(bx, by, bz):
                        continue
                    if grid.get(bx, by, bz) != VOXEL_ALARM_BELL:
                        continue
                    bell_pos = (bx, by, bz)
                    # Check cooldown
                    if self._alarm_cooldowns.get(bell_pos, 0) > 0:
                        continue
                    # Trigger alarm!
                    self._alarm_cooldowns[bell_pos] = ALARM_BELL_COOLDOWN
                    # Share alarm zone with intruder's personal map
                    intruder.personal_map.mark_alarm_zone(bx, by, bz)
                    self.event_bus.publish(
                        "alarm_bell_triggered",
                        intruder=intruder,
                        bell_x=bx, bell_y=by, bell_z=bz,
                    )

    def _tick_alarm_cooldowns(self) -> None:
        """Decrement all alarm bell cooldowns each tick."""
        expired: list[tuple[int, int, int]] = []
        for pos, cd in self._alarm_cooldowns.items():
            if cd <= 1:
                expired.append(pos)
            else:
                self._alarm_cooldowns[pos] = cd - 1
        for pos in expired:
            del self._alarm_cooldowns[pos]

    # ── Betrayal ───────────────────────────────────────────────────

    def _tick_betrayals(self, party: Party) -> None:
        """Check for treasure betrayals in a party."""
        grid = self.voxel_grid
        treasure_adj: dict[int, bool] = {}

        for m in party.alive_members:
            found = False
            for dx, dy, dz in (
                (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                (0, 0, 1), (0, 0, -1),
            ):
                nx, ny, nz = m.x + dx, m.y + dy, m.z + dz
                if (
                    grid.in_bounds(nx, ny, nz)
                    and grid.get(nx, ny, nz) == VOXEL_TREASURE
                ):
                    found = True
                    break
            treasure_adj[m.id] = found

        betrayers = party.check_betrayals(treasure_adj, self.rng)
        for b in betrayers:
            b.state = IntruderState.PILLAGING
            self._repath_intruder(b)
            logger.info("Intruder #%d betrayed their party for treasure!", b.id)
            self.event_bus.publish("intruder_betrayed", intruder=b)

    # ── Death handling ─────────────────────────────────────────────

    def _on_intruder_death(self, intruder: Intruder) -> None:
        """Handle intruder death: notify party, publish event."""
        intruder.state = IntruderState.DEAD
        self.event_bus.publish("intruder_died", intruder=intruder)
        logger.info(
            "Intruder #%d (%s) died", intruder.id, intruder.archetype.name,
        )

        # Notify party
        for party in list(self.parties) + list(self._underworld_parties):
            if any(m.id == intruder.id for m in party.members):
                party.on_member_death(intruder)
                break

    # ── Cleanup ────────────────────────────────────────────────────

    def _cleanup(self) -> None:
        """Remove dead/escaped intruders and wiped parties."""
        self.intruders = [i for i in self.intruders if i.alive]

        # Detect party wipes before removing them
        for party in self.parties:
            if party.is_wiped and all(
                m.state == IntruderState.DEAD for m in party.members
            ):
                self._reputation.on_party_wiped()
        for party in self._underworld_parties:
            if party.is_wiped and all(
                m.state == IntruderState.DEAD for m in party.members
            ):
                self._reputation.on_party_wiped()

        self.parties = [p for p in self.parties if not p.is_wiped]
        self._underworld_parties = [
            p for p in self._underworld_parties if not p.is_wiped
        ]

    # ── Level & status assignment ────────────────────────────────────

    @staticmethod
    def _assign_level(rng, level_shift: float = 0.0) -> int:
        """Assign a level 1-5 using weighted random, shifted by reputation.

        *level_shift* moves probability from level 1 toward higher levels.
        """
        weights = list(LEVEL_WEIGHTS)
        if level_shift > 0.0:
            shift = min(level_shift, weights[0] * 0.8)
            weights[0] -= shift
            # Distribute shift evenly across levels 2-5
            per_level = shift / 4
            for i in range(1, 5):
                weights[i] += per_level
        roll = rng.random()
        cumulative = 0.0
        for i, w in enumerate(weights):
            cumulative += w
            if roll < cumulative:
                return i + 1
        return 5

    @staticmethod
    def _level_to_status(level: int) -> IntruderStatus:
        """Convert intruder level to IntruderStatus."""
        if level <= 2:
            return IntruderStatus.GRUNT
        if level == 3:
            return IntruderStatus.VETERAN
        if level == 4:
            return IntruderStatus.ELITE
        return IntruderStatus.CHAMPION

    # ── Faction encounters ───────────────────────────────────────────

    def _tick_faction_encounters(self, tick: int) -> None:
        """Handle inter-faction combat between surface and underworld intruders.

        TODO: Overhaul faction dynamics — current implementation is
        hostile-only. Future work should add alliance, avoidance, and
        negotiation mechanics.
        """
        if tick % FACTION_ENCOUNTER_INTERVAL != 0:
            return

        # Performance guard: skip if no underworlders active
        has_uw = any(
            i.alive and i.is_underworlder for i in self.intruders
        )
        if not has_uw:
            return

        has_surface = any(
            i.alive and not i.is_underworlder for i in self.intruders
        )
        if not has_surface:
            return

        # Build spatial index: position → list of alive intruders
        pos_map: dict[tuple[int, int, int], list[Intruder]] = {}
        for i in self.intruders:
            if not i.alive:
                continue
            pos = i.pos
            if pos not in pos_map:
                pos_map[pos] = []
            pos_map[pos].append(i)

        # Check same-cell encounters
        engaged: set[int] = set()
        for pos, occupants in pos_map.items():
            if len(occupants) < 2:
                continue
            surface = [o for o in occupants if not o.is_underworlder]
            underworld = [o for o in occupants if o.is_underworlder]
            if not surface or not underworld:
                continue
            # All surface and underworld intruders in same cell fight
            for s in surface:
                for u in underworld:
                    if s.id in engaged or u.id in engaged:
                        continue
                    s.take_damage(u.effective_damage)
                    u.take_damage(s.effective_damage)
                    engaged.add(s.id)
                    engaged.add(u.id)
                    self.event_bus.publish(
                        "faction_combat",
                        surface=s, underworld=u,
                        x=pos[0], y=pos[1], z=pos[2],
                    )
                    if not s.alive:
                        self._on_intruder_death(s)
                    if not u.alive:
                        self._on_intruder_death(u)

        # Check adjacent encounters (6-connected)
        for pos, occupants in pos_map.items():
            for dx, dy, dz in (
                (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                (0, 0, 1), (0, 0, -1),
            ):
                adj = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
                if adj not in pos_map:
                    continue
                for a in occupants:
                    if not a.alive or a.id in engaged:
                        continue
                    if a.state not in (
                        IntruderState.ADVANCING, IntruderState.ATTACKING,
                    ):
                        continue
                    for b in pos_map[adj]:
                        if not b.alive or b.id in engaged:
                            continue
                        if b.state not in (
                            IntruderState.ADVANCING, IntruderState.ATTACKING,
                        ):
                            continue
                        if a.is_underworlder == b.is_underworlder:
                            continue
                        # Different factions, both advancing/attacking
                        a.take_damage(b.effective_damage)
                        b.take_damage(a.effective_damage)
                        engaged.add(a.id)
                        engaged.add(b.id)
                        self.event_bus.publish(
                            "faction_combat",
                            surface=b if not b.is_underworlder else a,
                            underworld=a if a.is_underworlder else b,
                            x=pos[0], y=pos[1], z=pos[2],
                        )
                        if not a.alive:
                            self._on_intruder_death(a)
                        if not b.alive:
                            self._on_intruder_death(b)
