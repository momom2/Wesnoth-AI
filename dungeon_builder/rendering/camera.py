"""Free-camera controls: orbit, pan, zoom, Z-level scrolling, and mouse picking."""

from __future__ import annotations

import math
import logging
from typing import TYPE_CHECKING

from panda3d.core import (
    LPoint3f,
    LVector3f,
    NodePath,
)
from direct.showbase.ShowBase import ShowBase

from dungeon_builder.config import (
    GRID_WIDTH,
    GRID_DEPTH,
    GRID_HEIGHT,
    VOXEL_AIR,
    CAMERA_DEFAULT_DISTANCE,
    CAMERA_MIN_DISTANCE,
    CAMERA_MAX_DISTANCE,
    CAMERA_DEFAULT_HEADING,
    CAMERA_DEFAULT_PITCH,
    CAMERA_PAN_SPEED,
    CAMERA_ROTATE_SPEED,
    CAMERA_ZOOM_STEP,
)

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus
    from dungeon_builder.core.game_state import GameState
    from dungeon_builder.rendering.layer_slice import LayerSliceManager

logger = logging.getLogger("dungeon_builder.camera")

# Mouse movement threshold (in NDC) to distinguish click from drag
_RIGHT_CLICK_THRESHOLD = 0.02


class CameraController:
    """RTS-style orbiting camera with Z-level scrolling and voxel picking."""

    def __init__(
        self,
        app: ShowBase,
        event_bus: EventBus,
        game_state: GameState,
        layer_manager: LayerSliceManager,
    ) -> None:
        self.app = app
        self.event_bus = event_bus
        self.game_state = game_state
        self.layer_manager = layer_manager

        # Camera state
        self.focus = LPoint3f(GRID_WIDTH / 2, GRID_DEPTH / 2, -1.0)
        self.distance = CAMERA_DEFAULT_DISTANCE
        self.heading = CAMERA_DEFAULT_HEADING
        self.pitch = CAMERA_DEFAULT_PITCH

        # Track held keys
        self._keys: dict[str, bool] = {}
        self._mouse_right_down = False
        self._mouse_mid_down = False
        self._last_mouse_x = 0.0
        self._last_mouse_y = 0.0

        # Right-click tracking (distinguish click from drag)
        self._right_click_start_x = 0.0
        self._right_click_start_y = 0.0

        self._bind_controls()
        self._update_camera()

    def _bind_controls(self) -> None:
        app = self.app
        # Key tracking for continuous input
        for key in ["w", "a", "s", "d", "q", "e"]:
            self._keys[key] = False
            app.accept(key, self._set_key, [key, True])
            app.accept(f"{key}-up", self._set_key, [key, False])

        # Z-level scrolling
        app.accept("t", self._z_up)
        app.accept("y", self._z_down)

        # Zoom
        app.accept("wheel_up", self._zoom_in)
        app.accept("wheel_down", self._zoom_out)

        # Mouse buttons for rotation/panning
        app.accept("mouse3", self._on_right_down)
        app.accept("mouse3-up", self._on_right_up)
        app.accept("mouse2", self._on_mid_down)
        app.accept("mouse2-up", self._on_mid_up)

        # Left click for voxel interaction
        app.accept("mouse1", self._on_left_click)

        # Tool switching: X key toggles dig/move
        app.accept("x", self._toggle_tool)

        # Register continuous input task
        app.taskMgr.add(self._input_task, "camera_input", sort=5)

    def _set_key(self, key: str, value: bool) -> None:
        self._keys[key] = value

    def _input_task(self, task):
        dt = globalClock.get_dt()

        # WASD panning (relative to camera heading)
        move = LVector3f(0, 0, 0)
        if self._keys.get("w"):
            move.y += 1
        if self._keys.get("s"):
            move.y -= 1
        if self._keys.get("a"):
            move.x -= 1
        if self._keys.get("d"):
            move.x += 1

        if move.length_squared() > 0:
            move.normalize()
            speed = CAMERA_PAN_SPEED * dt
            # Rotate movement by heading
            rad = math.radians(self.heading)
            cos_h = math.cos(rad)
            sin_h = math.sin(rad)
            dx = move.x * cos_h - move.y * sin_h
            dy = move.x * sin_h + move.y * cos_h
            self.focus.x += dx * speed
            self.focus.y += dy * speed

        # Q/E rotation
        if self._keys.get("q"):
            self.heading -= CAMERA_ROTATE_SPEED * dt
        if self._keys.get("e"):
            self.heading += CAMERA_ROTATE_SPEED * dt

        # Mouse drag rotation / panning
        if self.app.mouseWatcherNode.has_mouse():
            mx = self.app.mouseWatcherNode.get_mouse_x()
            my = self.app.mouseWatcherNode.get_mouse_y()

            if self._mouse_right_down:
                dx = mx - self._last_mouse_x
                dy = my - self._last_mouse_y
                self.heading -= dx * 100
                self.pitch = max(-89, min(-10, self.pitch + dy * 100))

            if self._mouse_mid_down:
                dx = mx - self._last_mouse_x
                dy = my - self._last_mouse_y
                speed = self.distance * 0.5
                rad = math.radians(self.heading)
                cos_h = math.cos(rad)
                sin_h = math.sin(rad)
                self.focus.x -= (dx * cos_h + dy * sin_h) * speed
                self.focus.y -= (dx * sin_h - dy * cos_h) * speed

            self._last_mouse_x = mx
            self._last_mouse_y = my

        self._update_camera()
        return task.cont

    def _update_camera(self) -> None:
        # Spherical to Cartesian
        pitch_rad = math.radians(self.pitch)
        heading_rad = math.radians(self.heading)

        cam_x = self.focus.x + self.distance * math.cos(pitch_rad) * math.sin(heading_rad)
        cam_y = self.focus.y - self.distance * math.cos(pitch_rad) * math.cos(heading_rad)
        cam_z = self.focus.z - self.distance * math.sin(pitch_rad)

        self.app.camera.set_pos(cam_x, cam_y, cam_z)
        self.app.camera.look_at(self.focus)

    def _z_up(self) -> None:
        new_z = max(0, self.layer_manager.current_z - 1)
        self.layer_manager.set_focus_z(new_z)
        self.focus.z = -new_z
        self.event_bus.publish("z_level_changed", z=new_z)

    def _z_down(self) -> None:
        new_z = min(GRID_HEIGHT - 1, self.layer_manager.current_z + 1)
        self.layer_manager.set_focus_z(new_z)
        self.focus.z = -new_z
        self.event_bus.publish("z_level_changed", z=new_z)

    def _zoom_in(self) -> None:
        self.distance = max(CAMERA_MIN_DISTANCE, self.distance - CAMERA_ZOOM_STEP)

    def _zoom_out(self) -> None:
        self.distance = min(CAMERA_MAX_DISTANCE, self.distance + CAMERA_ZOOM_STEP)

    def _on_right_down(self) -> None:
        self._mouse_right_down = True
        if self.app.mouseWatcherNode.has_mouse():
            self._right_click_start_x = self.app.mouseWatcherNode.get_mouse_x()
            self._right_click_start_y = self.app.mouseWatcherNode.get_mouse_y()

    def _on_right_up(self) -> None:
        self._mouse_right_down = False

        # Check if this was a click (not a drag)
        if self.app.mouseWatcherNode.has_mouse():
            mx = self.app.mouseWatcherNode.get_mouse_x()
            my = self.app.mouseWatcherNode.get_mouse_y()
            dx = abs(mx - self._right_click_start_x)
            dy = abs(my - self._right_click_start_y)
            if dx < _RIGHT_CLICK_THRESHOLD and dy < _RIGHT_CLICK_THRESHOLD:
                self._on_right_click()

    def _on_mid_down(self) -> None:
        self._mouse_mid_down = True

    def _on_mid_up(self) -> None:
        self._mouse_mid_down = False

    def _toggle_tool(self) -> None:
        """Toggle between dig and move tools."""
        if self.game_state.build_mode == "dig":
            self.game_state.build_mode = "move"
        else:
            self.game_state.build_mode = "dig"
        self.event_bus.publish("tool_changed", mode=self.game_state.build_mode)

    def _on_left_click(self) -> None:
        """Handle left-click: pick a voxel via DDA ray march and dispatch action."""
        hit = self._pick_voxel()
        if hit is None:
            return

        vx, vy, vz = hit
        mode = self.game_state.build_mode
        grid = self.game_state.voxel_grid
        if grid is None:
            return

        # Auto-switch: if in dig mode and clicked voxel is loose, switch to move
        if mode == "dig" and grid.is_loose(vx, vy, vz):
            self.game_state.build_mode = "move"
            mode = "move"
            self.event_bus.publish("tool_changed", mode="move")
            self.event_bus.publish(
                "error_message", text="Switched to move tool"
            )

        self.event_bus.publish(
            "voxel_left_clicked", x=vx, y=vy, z=vz, mode=mode
        )

    def _on_right_click(self) -> None:
        """Handle right-click: pick a voxel and dispatch right-click action."""
        hit = self._pick_voxel()
        if hit is None:
            # Right-click on air: try to find the air voxel for dropping
            hit = self._pick_air_voxel()
            if hit is None:
                return

        vx, vy, vz = hit
        self.event_bus.publish(
            "voxel_right_clicked",
            x=vx, y=vy, z=vz,
            mode=self.game_state.build_mode,
        )

    def _pick_voxel(self) -> tuple[int, int, int] | None:
        """Cast a ray from mouse into the voxel grid using DDA ray marching.

        This works directly on the voxel grid data — no collision solids needed.
        Returns the (x, y, z) grid coordinates of the first solid voxel hit,
        or None if the ray misses the grid entirely.
        """
        if not self.app.mouseWatcherNode.has_mouse():
            return None

        mpos = self.app.mouseWatcherNode.get_mouse()

        # Get ray origin and direction from the camera lens
        origin = LPoint3f()
        direction = LVector3f()
        lens = self.app.camNode.get_lens()
        if not lens.extrude(mpos, origin, direction):
            return None

        # Transform to world space
        cam_mat = self.app.camera.get_mat(self.app.render)
        origin = cam_mat.xform_point(origin)
        direction = cam_mat.xform_vec(direction)
        direction.normalize()

        # Convert world coordinates to grid coordinates
        # World: (wx, wy, wz) -> Grid: (wx, wy, -wz)
        # Ray: origin + t * direction, but in grid space z is flipped
        ox, oy, oz = origin.x, origin.y, -origin.z
        dx, dy, dz = direction.x, direction.y, -direction.z

        return self._dda_march(ox, oy, oz, dx, dy, dz)

    def _pick_air_voxel(self) -> tuple[int, int, int] | None:
        """Cast a ray and return the last air voxel before hitting solid.

        Used for right-click drop: find an air space to drop material into.
        """
        if not self.app.mouseWatcherNode.has_mouse():
            return None

        mpos = self.app.mouseWatcherNode.get_mouse()
        origin = LPoint3f()
        direction = LVector3f()
        lens = self.app.camNode.get_lens()
        if not lens.extrude(mpos, origin, direction):
            return None

        cam_mat = self.app.camera.get_mat(self.app.render)
        origin = cam_mat.xform_point(origin)
        direction = cam_mat.xform_vec(direction)
        direction.normalize()

        ox, oy, oz = origin.x, origin.y, -origin.z
        dx, dy, dz = direction.x, direction.y, -direction.z

        return self._dda_march_air(ox, oy, oz, dx, dy, dz)

    def _dda_march(
        self, ox: float, oy: float, oz: float,
        dx: float, dy: float, dz: float,
        max_steps: int = 200,
    ) -> tuple[int, int, int] | None:
        """3D DDA (Amanatides & Woo) ray march through the voxel grid.

        All coordinates are in grid space where z increases downward.
        """
        grid = self.game_state.voxel_grid
        if grid is None:
            return None

        # Current voxel
        vx = int(math.floor(ox))
        vy = int(math.floor(oy))
        vz = int(math.floor(oz))

        # Step direction (+1 or -1)
        step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
        step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
        step_z = 1 if dz > 0 else (-1 if dz < 0 else 0)

        # Distance along ray to cross one voxel in each axis
        INF = 1e30
        t_delta_x = abs(1.0 / dx) if dx != 0 else INF
        t_delta_y = abs(1.0 / dy) if dy != 0 else INF
        t_delta_z = abs(1.0 / dz) if dz != 0 else INF

        # Distance to next voxel boundary in each axis
        if dx > 0:
            t_max_x = (math.floor(ox) + 1 - ox) * t_delta_x
        elif dx < 0:
            t_max_x = (ox - math.floor(ox)) * t_delta_x
        else:
            t_max_x = INF

        if dy > 0:
            t_max_y = (math.floor(oy) + 1 - oy) * t_delta_y
        elif dy < 0:
            t_max_y = (oy - math.floor(oy)) * t_delta_y
        else:
            t_max_y = INF

        if dz > 0:
            t_max_z = (math.floor(oz) + 1 - oz) * t_delta_z
        elif dz < 0:
            t_max_z = (oz - math.floor(oz)) * t_delta_z
        else:
            t_max_z = INF

        for _ in range(max_steps):
            # Check if current voxel is in bounds and solid
            if grid.in_bounds(vx, vy, vz) and grid.get(vx, vy, vz) != VOXEL_AIR:
                return (vx, vy, vz)

            # Check if we've left the grid entirely (and won't come back)
            if (vx < -1 or vx > grid.width
                or vy < -1 or vy > grid.depth
                or vz < -1 or vz > grid.height):
                return None

            # Advance to next voxel boundary
            if t_max_x < t_max_y:
                if t_max_x < t_max_z:
                    vx += step_x
                    t_max_x += t_delta_x
                else:
                    vz += step_z
                    t_max_z += t_delta_z
            else:
                if t_max_y < t_max_z:
                    vy += step_y
                    t_max_y += t_delta_y
                else:
                    vz += step_z
                    t_max_z += t_delta_z

        return None

    def _dda_march_air(
        self, ox: float, oy: float, oz: float,
        dx: float, dy: float, dz: float,
        max_steps: int = 200,
    ) -> tuple[int, int, int] | None:
        """DDA march returning the last air voxel before hitting a solid."""
        grid = self.game_state.voxel_grid
        if grid is None:
            return None

        vx = int(math.floor(ox))
        vy = int(math.floor(oy))
        vz = int(math.floor(oz))

        step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
        step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
        step_z = 1 if dz > 0 else (-1 if dz < 0 else 0)

        INF = 1e30
        t_delta_x = abs(1.0 / dx) if dx != 0 else INF
        t_delta_y = abs(1.0 / dy) if dy != 0 else INF
        t_delta_z = abs(1.0 / dz) if dz != 0 else INF

        if dx > 0:
            t_max_x = (math.floor(ox) + 1 - ox) * t_delta_x
        elif dx < 0:
            t_max_x = (ox - math.floor(ox)) * t_delta_x
        else:
            t_max_x = INF

        if dy > 0:
            t_max_y = (math.floor(oy) + 1 - oy) * t_delta_y
        elif dy < 0:
            t_max_y = (oy - math.floor(oy)) * t_delta_y
        else:
            t_max_y = INF

        if dz > 0:
            t_max_z = (math.floor(oz) + 1 - oz) * t_delta_z
        elif dz < 0:
            t_max_z = (oz - math.floor(oz)) * t_delta_z
        else:
            t_max_z = INF

        last_air: tuple[int, int, int] | None = None

        for _ in range(max_steps):
            if grid.in_bounds(vx, vy, vz):
                if grid.get(vx, vy, vz) == VOXEL_AIR:
                    last_air = (vx, vy, vz)
                else:
                    # Hit solid — return the last air we saw
                    return last_air

            if (vx < -1 or vx > grid.width
                or vy < -1 or vy > grid.depth
                or vz < -1 or vz > grid.height):
                return last_air

            if t_max_x < t_max_y:
                if t_max_x < t_max_z:
                    vx += step_x
                    t_max_x += t_delta_x
                else:
                    vz += step_z
                    t_max_z += t_delta_z
            else:
                if t_max_y < t_max_z:
                    vy += step_y
                    t_max_y += t_delta_y
                else:
                    vz += step_z
                    t_max_z += t_delta_z

        return last_air
