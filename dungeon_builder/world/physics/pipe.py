"""Pipe & pump network physics for heat/humidity transport.

Pipes form networks that conduct heat and humidity between connected cells.
Pumps drive directional flow through their pipe networks.

Network topology is cached and invalidated on voxel changes.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from dungeon_builder.config import (
    VOXEL_PIPE,
    VOXEL_PUMP,
    PIPE_CONDUCTIVITY_BASE,
    PUMP_CONVECTION_RATE,
    PUMP_TICK_INTERVAL,
    METAL_CONDUCTIVITY_MULT,
)

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus
    from dungeon_builder.world.voxel_grid import VoxelGrid


# Direction offsets for pump block_state values (0-5)
_PUMP_DIRS = {
    0: (1, 0, 0),   # +X
    1: (-1, 0, 0),  # -X
    2: (0, 1, 0),   # +Y
    3: (0, -1, 0),  # -Y
    4: (0, 0, 1),   # +Z (deeper)
    5: (0, 0, -1),  # -Z (shallower)
}

# 6-connected neighbor offsets
_NEIGHBORS = (
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
)


class PipePhysics:
    """Manages heat/humidity transport through pipe networks.

    - **Passive conduction**: Heat and humidity average across pipe cells,
      weighted by ``PIPE_CONDUCTIVITY_BASE × METAL_CONDUCTIVITY_MULT[metal]``.
    - **Active pumping**: Pumps pull heat/humidity from their intake side
      and push it through the connected pipe network.
    - **Cache invalidation**: Networks are rebuilt when pipes/pumps change.

    Runs every ``PUMP_TICK_INTERVAL`` ticks.
    """

    def __init__(self, event_bus: EventBus, voxel_grid: VoxelGrid) -> None:
        self.event_bus = event_bus
        self.voxel_grid = voxel_grid

        # Cached networks: list of (pipe_positions, pump_positions)
        self._networks: list[tuple[list[tuple[int, int, int]],
                                   list[tuple[int, int, int]]]] | None = None
        self._cache_dirty = True

        event_bus.subscribe("tick", self._on_tick)
        event_bus.subscribe("voxel_changed", self._on_voxel_changed)

    def _on_voxel_changed(self, x: int = 0, y: int = 0, z: int = 0, **kw) -> None:
        """Invalidate network cache when any voxel changes."""
        self._cache_dirty = True

    def _on_tick(self, tick: int, **kw) -> None:
        if tick % PUMP_TICK_INTERVAL != 0:
            return
        self._update()

    def _update(self) -> None:
        """Run passive conduction and active pumping."""
        if self._cache_dirty:
            self._rebuild_networks()
            self._cache_dirty = False

        if not self._networks:
            return

        grid = self.voxel_grid
        temp = grid.temperature
        hum = grid.humidity
        metal_arr = grid.metal_type

        for pipes, pumps in self._networks:
            if not pipes:
                continue

            # --- Passive conduction: average heat/humidity across network ---
            total_heat = 0.0
            total_hum = 0.0
            total_weight = 0.0

            for x, y, z in pipes:
                mt = int(metal_arr[x, y, z]) & 0x7F  # base metal
                conductivity = PIPE_CONDUCTIVITY_BASE * METAL_CONDUCTIVITY_MULT.get(
                    mt, 0.5,
                )
                total_heat += float(temp[x, y, z]) * conductivity
                total_hum += float(hum[x, y, z]) * conductivity
                total_weight += conductivity

            if total_weight <= 0.0:
                continue

            avg_heat = total_heat / total_weight
            avg_hum = total_hum / total_weight

            # Apply averaging (blend toward average)
            for x, y, z in pipes:
                mt = int(metal_arr[x, y, z]) & 0x7F
                conductivity = PIPE_CONDUCTIVITY_BASE * METAL_CONDUCTIVITY_MULT.get(
                    mt, 0.5,
                )
                blend = min(conductivity, 1.0)
                temp[x, y, z] += (avg_heat - float(temp[x, y, z])) * blend
                hum[x, y, z] += (avg_hum - float(hum[x, y, z])) * blend

            # --- Active pumping: pull from intake, push through network ---
            for px, py, pz in pumps:
                direction = int(grid.block_state[px, py, pz])
                dx, dy, dz = _PUMP_DIRS.get(direction, (1, 0, 0))

                # Intake is OPPOSITE the pump direction
                sx, sy, sz = px - dx, py - dy, pz - dz
                if not grid.in_bounds(sx, sy, sz):
                    continue

                # Pull heat/humidity from intake cell
                source_heat = float(temp[sx, sy, sz])
                source_hum = float(hum[sx, sy, sz])

                if source_heat <= 0.0 and source_hum <= 0.0:
                    continue

                # Transfer rate
                mt = int(metal_arr[px, py, pz]) & 0x7F
                rate = PUMP_CONVECTION_RATE * METAL_CONDUCTIVITY_MULT.get(mt, 0.5)

                heat_transfer = source_heat * rate
                hum_transfer = source_hum * rate

                # Remove from source
                temp[sx, sy, sz] -= heat_transfer
                hum[sx, sy, sz] -= hum_transfer

                # Distribute to network pipes (evenly)
                if pipes:
                    per_pipe_heat = heat_transfer / len(pipes)
                    per_pipe_hum = hum_transfer / len(pipes)
                    for nx, ny, nz in pipes:
                        temp[nx, ny, nz] += per_pipe_heat
                        hum[nx, ny, nz] += per_pipe_hum

    def _rebuild_networks(self) -> None:
        """BFS from each pump to find connected pipe networks."""
        grid = self.voxel_grid
        voxels = grid.grid
        w, d, h = grid.width, grid.depth, grid.height

        visited: set[tuple[int, int, int]] = set()
        self._networks = []

        # Find all pipe and pump positions
        all_pipe_pump = set()
        pipe_positions = np.argwhere(voxels == VOXEL_PIPE)
        pump_positions = np.argwhere(voxels == VOXEL_PUMP)

        for pos in pipe_positions:
            all_pipe_pump.add((int(pos[0]), int(pos[1]), int(pos[2])))
        for pos in pump_positions:
            all_pipe_pump.add((int(pos[0]), int(pos[1]), int(pos[2])))

        if not all_pipe_pump:
            return

        # BFS from each unvisited pipe/pump to find connected components
        for start in all_pipe_pump:
            if start in visited:
                continue

            network_pipes: list[tuple[int, int, int]] = []
            network_pumps: list[tuple[int, int, int]] = []
            queue: deque[tuple[int, int, int]] = deque([start])
            visited.add(start)

            while queue:
                cx, cy, cz = queue.popleft()
                vtype = int(voxels[cx, cy, cz])

                if vtype == VOXEL_PIPE:
                    network_pipes.append((cx, cy, cz))
                elif vtype == VOXEL_PUMP:
                    network_pipes.append((cx, cy, cz))  # Pumps also conduct
                    network_pumps.append((cx, cy, cz))

                # Explore 6-connected neighbors
                for dx, dy, dz in _NEIGHBORS:
                    nx, ny, nz = cx + dx, cy + dy, cz + dz
                    npos = (nx, ny, nz)
                    if npos in visited:
                        continue
                    if not (0 <= nx < w and 0 <= ny < d and 0 <= nz < h):
                        continue
                    ntype = int(voxels[nx, ny, nz])
                    if ntype in (VOXEL_PIPE, VOXEL_PUMP):
                        visited.add(npos)
                        queue.append(npos)

            if network_pipes:
                self._networks.append((network_pipes, network_pumps))
