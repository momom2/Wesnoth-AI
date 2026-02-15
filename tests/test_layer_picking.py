"""Tests for layer-constrained voxel picking (ray-plane intersection).

These test the pure math of intersecting a ray with a horizontal plane at
a given z-level, without needing a full Panda3D window.
"""

import math

import pytest


def ray_plane_intersect(
    origin_x: float, origin_y: float, origin_z: float,
    dir_x: float, dir_y: float, dir_z: float,
    z_level: int,
) -> tuple[int, int] | None:
    """Pure-Python replica of CameraController._ray_hit_layer math.

    Intersects a ray with the plane at world_z = -z_level + 0.5.
    Returns (grid_x, grid_y) or None.
    """
    plane_z = -z_level + 0.5

    if abs(dir_z) < 1e-12:
        return None  # Ray parallel to plane

    t = (plane_z - origin_z) / dir_z
    if t < 0:
        return None  # Plane behind camera

    hit_x = origin_x + t * dir_x
    hit_y = origin_y + t * dir_y

    gx = int(math.floor(hit_x))
    gy = int(math.floor(hit_y))
    return gx, gy


class TestRayPlaneIntersection:
    def test_straight_down_hit(self):
        """Ray pointing straight down from above hits the expected cell."""
        # Camera at (10.5, 20.5, 50), looking straight down
        result = ray_plane_intersect(
            10.5, 20.5, 50.0,
            0.0, 0.0, -1.0,
            z_level=1,
        )
        # Plane at world z = -1 + 0.5 = -0.5
        # t = (-0.5 - 50) / -1 = 50.5
        # hit = (10.5, 20.5) -> grid (10, 20)
        assert result == (10, 20)

    def test_angled_hit(self):
        """Angled ray hits the correct cell on the layer plane."""
        # Camera at (0, 0, 10), direction angled toward (5, 5, -10)
        dx, dy, dz = 5.0, 5.0, -10.0
        length = math.sqrt(dx**2 + dy**2 + dz**2)
        dx /= length
        dy /= length
        dz /= length

        result = ray_plane_intersect(
            0.0, 0.0, 10.0,
            dx, dy, dz,
            z_level=0,
        )
        # Plane at world z = 0.5
        # t = (0.5 - 10) / dz
        # dz_norm = -10/sqrt(150) ~ -0.8165
        # t = -9.5 / -0.8165 ~ 11.635
        # hit_x = 0 + t * 5/sqrt(150) ~ 11.635 * 0.4082 ~ 4.75 -> floor = 4
        # hit_y same -> 4
        assert result == (4, 4)

    def test_parallel_ray_misses(self):
        """Ray parallel to the layer plane returns None."""
        result = ray_plane_intersect(
            10.0, 10.0, 5.0,
            1.0, 0.0, 0.0,  # Horizontal ray
            z_level=3,
        )
        assert result is None

    def test_plane_behind_camera(self):
        """Ray pointing away from the plane returns None."""
        # Camera at (10, 10, -5), below the plane at z_level=0 (world z=0.5)
        # Ray pointing further down
        result = ray_plane_intersect(
            10.0, 10.0, -5.0,
            0.0, 0.0, -1.0,
            z_level=0,
        )
        # Plane is at world z = 0.5, camera at -5, ray goes to -inf
        # t = (0.5 - (-5)) / (-1) = -5.5 -> negative -> behind
        assert result is None

    def test_deep_layer(self):
        """Picking on a deep layer (z=10) works correctly."""
        # Camera at (32.5, 32.5, 30), looking down
        result = ray_plane_intersect(
            32.5, 32.5, 30.0,
            0.0, 0.0, -1.0,
            z_level=10,
        )
        # Plane at world z = -10 + 0.5 = -9.5
        # t = (-9.5 - 30) / -1 = 39.5
        # hit = (32.5, 32.5) -> grid (32, 32)
        assert result == (32, 32)

    def test_fractional_cell_boundary(self):
        """Hit at exactly a cell boundary floors correctly."""
        # Hit at x=5.0 exactly -> floor(5.0) = 5
        result = ray_plane_intersect(
            5.0, 7.3, 10.0,
            0.0, 0.0, -1.0,
            z_level=1,
        )
        assert result is not None
        assert result[0] == 5
        assert result[1] == 7

    def test_negative_coordinate_not_in_grid(self):
        """Hits at negative coordinates are returned (bounds checking is separate)."""
        result = ray_plane_intersect(
            -0.5, -0.5, 10.0,
            0.0, 0.0, -1.0,
            z_level=1,
        )
        # floor(-0.5) = -1
        assert result == (-1, -1)
