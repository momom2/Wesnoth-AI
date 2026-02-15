"""Free-camera controls: orbit, pan, zoom, Z-level scrolling, and mouse picking."""

from __future__ import annotations

import math
import logging
from typing import TYPE_CHECKING

from panda3d.core import (
    LPoint3f,
    LVector3f,
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

        # Hover tracking — last hovered voxel for highlight
        self._hovered_voxel: tuple[int, int, int] | None = None

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
        self._update_hover()
        return task.cont

    def _update_hover(self) -> None:
        """Update the hovered voxel highlight each frame."""
        hit = self._pick_voxel()
        if hit != self._hovered_voxel:
            self._hovered_voxel = hit
            if hit is not None:
                self.event_bus.publish(
                    "voxel_hover", x=hit[0], y=hit[1], z=hit[2]
                )
            else:
                self.event_bus.publish("voxel_hover_clear")

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
        """Handle left-click: pick a voxel and dispatch action."""
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

    def _get_mouse_ray(self) -> tuple[LPoint3f, LVector3f] | None:
        """Get the mouse ray origin and direction in world space.

        Returns (origin, direction) or None if mouse is not available.
        """
        if not self.app.mouseWatcherNode.has_mouse():
            return None

        mpos = self.app.mouseWatcherNode.get_mouse()

        near_point = LPoint3f()
        far_point = LPoint3f()
        lens = self.app.camNode.get_lens()
        if not lens.extrude(mpos, near_point, far_point):
            return None

        # Transform to world space
        cam_mat = self.app.camera.get_mat(self.app.render)
        origin = cam_mat.xform_point(near_point)
        far_world = cam_mat.xform_point(far_point)

        # Direction is from near to far
        direction = LVector3f(far_world - origin)
        direction.normalize()
        return origin, direction

    def _ray_hit_layer(
        self, origin: LPoint3f, direction: LVector3f
    ) -> tuple[int, int] | None:
        """Intersect a ray with the horizontal plane of the current layer.

        The current layer at grid z = current_z is rendered at world z in
        the range [-current_z, -current_z + 1].  We intersect the ray with
        the middle of that slab (world_z = -current_z + 0.5) and return
        the (grid_x, grid_y) of the hit cell, or None if the ray is
        parallel to the plane or the hit is outside the grid.
        """
        z_level = self.layer_manager.current_z
        # Voxels at grid z render with their top face at world z = -z_level + 1
        # and bottom face at world z = -z_level.  Use the midpoint.
        plane_z = -z_level + 0.5

        # Ray: P = origin + t * direction
        # Solve for t where P.z == plane_z
        dz = direction.z
        if abs(dz) < 1e-12:
            return None  # Ray parallel to the layer plane

        t = (plane_z - origin.z) / dz
        if t < 0:
            return None  # Plane is behind the camera

        hit_x = origin.x + t * direction.x
        hit_y = origin.y + t * direction.y

        gx = int(math.floor(hit_x))
        gy = int(math.floor(hit_y))

        grid = self.game_state.voxel_grid
        if grid is None:
            return None
        if not grid.in_bounds(gx, gy, z_level):
            return None

        return gx, gy

    def _pick_voxel(self) -> tuple[int, int, int] | None:
        """Pick the solid voxel under the mouse on the current layer.

        Returns (x, y, z) grid coordinates or None.
        """
        ray = self._get_mouse_ray()
        if ray is None:
            return None

        hit = self._ray_hit_layer(*ray)
        if hit is None:
            return None

        gx, gy = hit
        z_level = self.layer_manager.current_z
        grid = self.game_state.voxel_grid
        if grid is None:
            return None

        if grid.get(gx, gy, z_level) != VOXEL_AIR:
            return (gx, gy, z_level)
        return None

    def _pick_air_voxel(self) -> tuple[int, int, int] | None:
        """Pick the air voxel under the mouse on the current layer.

        Used for right-click drop: find an air space to drop material into.
        """
        ray = self._get_mouse_ray()
        if ray is None:
            return None

        hit = self._ray_hit_layer(*ray)
        if hit is None:
            return None

        gx, gy = hit
        z_level = self.layer_manager.current_z
        grid = self.game_state.voxel_grid
        if grid is None:
            return None

        if grid.get(gx, gy, z_level) == VOXEL_AIR:
            return (gx, gy, z_level)
        return None
