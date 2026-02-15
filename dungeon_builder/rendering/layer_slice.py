"""Z-level layer transparency management for slice rendering."""

from __future__ import annotations

from panda3d.core import NodePath, TransparencyAttrib

from dungeon_builder.config import GRID_HEIGHT, LAYER_ALPHA, LAYER_MAX_VISIBLE_OFFSET


class LayerSliceManager:
    """Manages per-Z-level parent NodePaths and their transparency.

    The focused Z-level is fully opaque. Adjacent levels are progressively
    transparent. Levels beyond the visible offset are hidden entirely.
    """

    def __init__(self, render_node: NodePath) -> None:
        self.render_node = render_node
        self.layers: dict[int, NodePath] = {}
        self.current_z: int = 1  # Start looking at just below surface

        self._create_layers()
        self.set_focus_z(self.current_z)

    def _create_layers(self) -> None:
        for z in range(GRID_HEIGHT):
            layer_np = self.render_node.attach_new_node(f"layer_{z}")
            self.layers[z] = layer_np

    def get_layer(self, z: int) -> NodePath | None:
        return self.layers.get(z)

    def set_focus_z(self, z: int) -> None:
        z = max(0, min(GRID_HEIGHT - 1, z))
        self.current_z = z

        for zl, np in self.layers.items():
            offset = abs(zl - z)
            if offset > LAYER_MAX_VISIBLE_OFFSET:
                np.hide()
            else:
                np.show()
                alpha = LAYER_ALPHA.get(offset, 0.0)
                if alpha < 1.0:
                    np.set_transparency(TransparencyAttrib.M_alpha)
                    np.set_alpha_scale(alpha)
                else:
                    np.clear_transparency()
                    np.set_alpha_scale(1.0)
