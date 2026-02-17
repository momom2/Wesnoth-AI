"""Tests for the per-vertex color noise function used in block rendering."""

from __future__ import annotations

import pytest

from dungeon_builder.rendering.voxel_renderer import _vertex_noise
from dungeon_builder.config import (
    VERTEX_NOISE_AMPLITUDE,
    VOXEL_NOISE,
    VOXEL_DIRT,
    VOXEL_STONE,
    VOXEL_MARBLE,
    VOXEL_OBSIDIAN,
    VOXEL_GRANITE,
    VOXEL_LAVA,
    VOXEL_WATER,
    VOXEL_MANA_CRYSTAL,
)


class TestVertexNoise:
    """Tests for the _vertex_noise hash function."""

    def test_noise_deterministic(self):
        """Same inputs always produce the same output."""
        a = _vertex_noise(5, 10, 3, 2, 1)
        b = _vertex_noise(5, 10, 3, 2, 1)
        assert a == b

    def test_noise_range(self):
        """Output is in [-1.0, +1.0] for many different inputs."""
        for gx in range(0, 64, 7):
            for gy in range(0, 64, 13):
                for gz in range(0, 21, 5):
                    for fi in range(6):
                        for vi in range(4):
                            val = _vertex_noise(gx, gy, gz, fi, vi)
                            assert -1.0 <= val <= 1.0, (
                                f"Out of range: {val} at ({gx},{gy},{gz},f{fi},v{vi})"
                            )

    def test_noise_varies_by_position(self):
        """Different (gx, gy, gz) produce different values."""
        vals = set()
        for gx in range(10):
            vals.add(_vertex_noise(gx, 5, 3, 0, 0))
        # With 10 different positions, we should get more than 1 unique value
        assert len(vals) > 5

    def test_noise_varies_by_face(self):
        """Same position but different face_idx produces different values."""
        vals = set()
        for fi in range(6):
            vals.add(_vertex_noise(5, 5, 5, fi, 0))
        # 6 faces should produce mostly unique values
        assert len(vals) >= 4

    def test_noise_varies_by_vertex(self):
        """Same position and face but different vert_idx produces different values."""
        vals = set()
        for vi in range(4):
            vals.add(_vertex_noise(5, 5, 5, 0, vi))
        # 4 vertices should produce at least 3 unique values
        assert len(vals) >= 3

    def test_noise_distribution_reasonable(self):
        """Over many samples, mean is approximately 0 and values use most of range."""
        values = []
        for i in range(1000):
            values.append(_vertex_noise(i, i * 7, i * 13, i % 6, i % 4))

        mean = sum(values) / len(values)
        assert abs(mean) < 0.15, f"Mean too far from 0: {mean}"

        min_val = min(values)
        max_val = max(values)
        assert min_val < -0.5, f"Min not low enough: {min_val}"
        assert max_val > 0.5, f"Max not high enough: {max_val}"


class TestNoiseConfig:
    """Tests for per-material noise amplitude configuration."""

    def test_per_material_amplitude_config(self):
        """VOXEL_NOISE has entries for key geological materials."""
        for vtype in (
            VOXEL_DIRT, VOXEL_STONE, VOXEL_MARBLE, VOXEL_OBSIDIAN,
            VOXEL_GRANITE, VOXEL_LAVA, VOXEL_WATER, VOXEL_MANA_CRYSTAL,
        ):
            assert vtype in VOXEL_NOISE, f"Missing noise entry for vtype {vtype}"

    def test_default_amplitude_used_for_unlisted(self):
        """Voxel types not in VOXEL_NOISE use VERTEX_NOISE_AMPLITUDE default."""
        # Use a voxel type that's not in the per-material table
        unlisted_type = 999
        amp = VOXEL_NOISE.get(unlisted_type, VERTEX_NOISE_AMPLITUDE)
        assert amp == VERTEX_NOISE_AMPLITUDE

    def test_marble_has_high_noise(self):
        """Marble has higher noise amplitude than obsidian (veined vs glassy)."""
        assert VOXEL_NOISE[VOXEL_MARBLE] > VOXEL_NOISE[VOXEL_OBSIDIAN]

    def test_obsidian_has_low_noise(self):
        """Obsidian has the lowest noise amplitude (glassy, smooth)."""
        assert VOXEL_NOISE[VOXEL_OBSIDIAN] <= min(
            v for k, v in VOXEL_NOISE.items() if k != VOXEL_OBSIDIAN
        )

    def test_all_amplitudes_positive(self):
        """All noise amplitudes are positive."""
        assert VERTEX_NOISE_AMPLITUDE > 0
        for vtype, amp in VOXEL_NOISE.items():
            assert amp > 0, f"Amplitude for vtype {vtype} is non-positive: {amp}"
