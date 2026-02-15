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
    generate_composition,
)
from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_TREASURE,
    VOXEL_DOOR,
    SURFACE_Z,
    DIG_DURATION,
    NON_DIGGABLE,
    INTRUDER_PARTY_SPAWN_INTERVAL,
    MAX_PARTIES,
    MAX_INTRUDERS_TOTAL,
    PYREMANCER_HEAT_AMOUNT,
    PYREMANCER_HEAT_INTERVAL,
    GORECLAW_FRENZY_RANDOM_CHANCE,
)

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
        self._next_id = 1
        self._next_party_id = 1
        self._spawn_timer = 0

        event_bus.subscribe("tick", self._on_tick)
        event_bus.subscribe("intruder_needs_repath", self._on_needs_repath)
        event_bus.subscribe("game_over", self._on_game_over)

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

        # Party-level updates
        for party in self.parties:
            if party.is_wiped:
                continue
            party.share_maps()
            party.apply_warden_aura()
            heals = party.tick_warden_heal()
            for warden, patient, amount in heals:
                logger.debug(
                    "Warden #%d heals #%d for %d HP",
                    warden.id, patient.id, amount,
                )
            self._tick_betrayals(party)

        # Per-intruder update
        for intruder in self.intruders:
            if not intruder.alive:
                continue
            self._update_intruder(intruder, tick)

        # Periodic cleanup
        if tick % 100 == 0:
            self._cleanup()

    def _on_needs_repath(self, intruder: Intruder, **kwargs) -> None:
        self._repath_intruder(intruder)

    # ── Spawning ───────────────────────────────────────────────────

    def _tick_spawning(self) -> None:
        self._spawn_timer += 1
        if self._spawn_timer < INTRUDER_PARTY_SPAWN_INTERVAL:
            return
        self._spawn_timer = 0

        alive_count = sum(1 for i in self.intruders if i.alive)
        active_parties = sum(1 for p in self.parties if not p.is_wiped)

        if alive_count >= MAX_INTRUDERS_TOTAL:
            return
        if active_parties >= MAX_PARTIES:
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

        for arch in composition:
            obj = self._pick_objective(arch, party_rng)
            intruder = Intruder(
                intruder_id=self._next_id,
                x=sx, y=sy, z=sz,
                archetype=arch,
                objective=obj,
                personal_map=PersonalMap(),
            )
            self._next_id += 1
            intruder.state = IntruderState.ADVANCING
            members.append(intruder)

        if not members:
            return

        party = Party(self._next_party_id, members)
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
            self._update_retreating(intruder)
        elif state == IntruderState.PILLAGING:
            self._update_pillaging(intruder)

    # ── Vision ─────────────────────────────────────────────────────

    def _update_vision(self, intruder: Intruder) -> None:
        """Update the intruder's personal map from current LOS + special vision."""
        grid = self.voxel_grid
        arch = intruder.archetype
        x, y, z = intruder.x, intruder.y, intruder.z
        pmap = intruder.personal_map

        # Standard LOS
        visible = compute_los(grid, x, y, z, arch.perception_range)
        for vx, vy, vz in visible:
            vtype = grid.get(vx, vy, vz)
            bstate = int(grid.block_state[vx, vy, vz])
            pmap.reveal(vx, vy, vz, vtype, bstate)

        # Arcane sight (Gloomseer)
        if arch.arcane_sight_range > 0:
            arcane = compute_arcane_sight(grid, x, y, z, arch.arcane_sight_range)
            for vx, vy, vz in arcane:
                vtype = grid.get(vx, vy, vz)
                bstate = int(grid.block_state[vx, vy, vz])
                pmap.reveal(vx, vy, vz, vtype, bstate)

        # Thermal vision (Pyremancer)
        if arch.fire_immune and arch.perception_range >= 4:
            thermal = compute_thermal_vision(grid, x, y, z, 4)
            for vx, vy, vz in thermal:
                vtype = grid.get(vx, vy, vz)
                bstate = int(grid.block_state[vx, vy, vz])
                pmap.reveal(vx, vy, vz, vtype, bstate)

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

        # Pyremancer: heat adjacent blocks periodically
        self._tick_pyremancer_heat(intruder, tick)

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
                logger.debug(
                    "Intruder #%d collected treasure at (%d,%d,%d)",
                    intruder.id, tx, ty, tz,
                )
                self.event_bus.publish(
                    "intruder_collected_treasure",
                    intruder=intruder, x=tx, y=ty, z=tz,
                )
        elif itype == "dig":
            vtype = grid.get(tx, ty, tz)
            if vtype != VOXEL_AIR and vtype not in NON_DIGGABLE:
                grid.set(tx, ty, tz, VOXEL_AIR)
                intruder.personal_map.reveal(tx, ty, tz, VOXEL_AIR, 0)
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

    def _update_retreating(self, intruder: Intruder) -> None:
        intruder.ticks_since_move += 1
        if intruder.ticks_since_move < intruder.effective_move_interval:
            return
        intruder.ticks_since_move = 0

        self._advance_along_path(intruder)

        if intruder.z == SURFACE_Z:
            intruder.state = IntruderState.ESCAPED
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
        self.event_bus.publish("intruder_moved", intruder=intruder)

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
            self.event_bus.publish("intruder_moved", intruder=intruder)

    # ── Digging ────────────────────────────────────────────────────

    def _start_digging(
        self,
        intruder: Intruder,
        target: tuple[int, int, int],
        vtype: int,
    ) -> None:
        """Start digging through a solid block (Tunneler half-duration)."""
        base_ticks = DIG_DURATION.get(vtype, 40)
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
        if arch.retreat_threshold <= 0:
            return
        hp_ratio = intruder.hp / intruder.max_hp
        if hp_ratio < arch.retreat_threshold:
            intruder.state = IntruderState.RETREATING

    def _start_retreat(self, intruder: Intruder) -> None:
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

    # ── Pathing ────────────────────────────────────────────────────

    def _repath_intruder(self, intruder: Intruder) -> None:
        """Find a new path for the intruder based on its objective/state."""
        if intruder.state == IntruderState.RETREATING:
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

        # Try personal pathfinder first
        path = PersonalPathfinder.find_path(
            intruder.personal_map, intruder.pos, goal, intruder.archetype,
        )
        if path and len(path) > 1:
            intruder.path = path
            intruder.path_index = 1
        else:
            # Fallback to global pathfinder
            path = self.pathfinder.find_path(intruder.pos, goal)
            if path and len(path) > 1:
                intruder.path = path
                intruder.path_index = 1
            else:
                logger.debug(
                    "Intruder #%d cannot find path from %s to %s",
                    intruder.id, intruder.pos, goal,
                )
                intruder.path = None

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
        for party in self.parties:
            if any(m.id == intruder.id for m in party.members):
                party.on_member_death(intruder)
                break

    # ── Cleanup ────────────────────────────────────────────────────

    def _cleanup(self) -> None:
        """Remove dead/escaped intruders and wiped parties."""
        self.intruders = [i for i in self.intruders if i.alive]
        self.parties = [p for p in self.parties if not p.is_wiped]
