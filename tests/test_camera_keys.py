"""Tests for camera key binding configuration (no Panda3D window needed)."""

import pytest

from dungeon_builder.rendering.camera import CameraController


class TestArrowKeyConfig:
    """Verify arrow keys are configured in the camera key list."""

    def test_arrow_keys_in_bind_list(self):
        """The _bind_controls method should register arrow keys."""
        # We can't instantiate CameraController without Panda3D, but we
        # can verify the source code includes arrow keys by checking that
        # the class has the expected key list in its bind method.
        import inspect
        src = inspect.getsource(CameraController._bind_controls)
        assert "arrow_up" in src
        assert "arrow_down" in src
        assert "arrow_left" in src
        assert "arrow_right" in src

    def test_arrow_keys_in_input_task(self):
        """The _input_task movement logic should handle arrow keys."""
        import inspect
        src = inspect.getsource(CameraController._input_task)
        assert "arrow_up" in src
        assert "arrow_down" in src
        assert "arrow_left" in src
        assert "arrow_right" in src

    def test_wasd_still_in_bind_list(self):
        """WASD keys should still be present (not replaced)."""
        import inspect
        src = inspect.getsource(CameraController._bind_controls)
        for key in ["w", "a", "s", "d", "q", "e"]:
            assert f'"{key}"' in src, f"Key '{key}' missing from bindings"
