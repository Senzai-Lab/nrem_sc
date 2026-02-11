"""Trail + marker for visualizing one trajectory with PlaybackController.

Works identically in 2-D and 3-D — just pass positions with 2 or 3 columns.
Multiple Trail instances can share the same scene and PlaybackController.

Example
-------
>>> trail_a = Trail(positions_a, scene, cmap="plasma", marker_color="#c10af3")
>>> trail_b = Trail(positions_b, scene, cmap="viridis", marker_color="#0af35a")
>>>
>>> def on_update(pb):
...     for t in [trail_a, trail_b]:
...         t.update(pb.frame_index, pb.frame_position)
>>>
>>> playback.add_handler(on_update)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pygfx as gfx
from matplotlib import colormaps

class Trail:
    """Animated trail + marker for a single trajectory.

    Parameters
    ----------
    positions : array-like, shape (N, 2) or (N, 3)
        Full set of positions for every frame.  2-D arrays are automatically
        padded with z=0 so pygfx can render them.
    scene : gfx.Scene
        The scene to add trail and marker world-objects to.
    trail_len : int
        Maximum number of past points shown in the trail.
    cmap : str
        Matplotlib colormap name applied along the trail.
    fade_power : float
        Exponent for the alpha fade curve.  Lower = faster fade of old points.
        ``alpha = t ** fade_power`` where *t* goes 0 → 1 (old → new).
    line_thickness : float
        Trail line thickness (world units).
    marker : str
        pygfx marker shape for the active-position indicator.
    marker_color : str | tuple
        Color of the active marker.
    marker_size : float
        Marker diameter (world units).
    cloud : bool
        If ``True`` (default), also draw a dim static cloud of all positions.
    cloud_size : float
        Point size for the static cloud.
    cloud_alpha : float
        Alpha for the static cloud points (0–1).
    cloud_brightness : float
        RGB brightness for the cloud points (0–1).
    """

    def __init__(
        self,
        positions,
        scene: gfx.Scene,
        *,
        trail_len: int = 500,
        cmap: str = "plasma",
        fade_power: float = 0.6,
        line_thickness: float = 2.5,
        marker: str = "circle",
        marker_color: str | tuple = "#c10af3",
        marker_size: float = 8.0,
        cloud: bool = True,
        cloud_size: float = 1.0,
        cloud_alpha: float = 0.5,
        cloud_brightness: float = 0.15,
    ):
        # ---- positions (ensure Nx3 float32)
        positions = np.asarray(positions, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] not in (2, 3):
            raise ValueError("positions must be (N, 2) or (N, 3)")
        if positions.shape[1] == 2:
            positions = np.column_stack(
                [positions, np.zeros(len(positions), dtype=np.float32)]
            )
        self._positions = positions
        self._n_frames = len(positions)

        # ---- cmap / fade
        self._cmap = colormaps[cmap]
        self._fade_power = float(fade_power)

        # ---- trail line
        self._trail_len = int(trail_len)
        self.line = gfx.Line(
            gfx.Geometry(
                positions=np.full((self._trail_len, 3), np.nan, dtype=np.float32),
                colors=np.zeros((self._trail_len, 4), dtype=np.float32),
            ),
            gfx.LineMaterial(
                thickness=line_thickness,
                color_mode="vertex",
                aa=True,
            ),
        )
        scene.add(self.line)

        # ---- active marker
        self.marker = gfx.Points(
            gfx.Geometry(positions=np.zeros((1, 3), dtype=np.float32)),
            gfx.PointsMarkerMaterial(
                marker=marker,
                size=marker_size,
                color=marker_color,
                size_space="world",
            ),
        )
        scene.add(self.marker)

        # ---- optional static cloud
        self.cloud: Optional[gfx.Points] = None
        if cloud:
            n = self._n_frames
            colors_dim = np.zeros((n, 4), dtype=np.float32)
            colors_dim[:, :3] = cloud_brightness
            colors_dim[:, 3] = cloud_alpha
            self.cloud = gfx.Points(
                gfx.Geometry(positions=self._positions, colors=colors_dim),
                gfx.PointsMaterial(
                    size=cloud_size,
                    color_mode="vertex",
                    size_space="world",
                ),
            )
            scene.add(self.cloud)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def positions(self) -> np.ndarray:
        """Full position array (N, 3)."""
        return self._positions

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def trail_len(self) -> int:
        return self._trail_len

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, frame_index: int, frame_position: float):
        """Refresh marker and trail for the given playback state.

        Parameters
        ----------
        frame_index : int
            Integer frame index (``pb.frame_index``).
        frame_position : float
            Fractional frame position for interpolation (``pb.frame_position``).
        """
        idx = int(frame_index)
        fp = float(frame_position)
        pos = self._positions
        n_frames = self._n_frames

        # --- interpolated marker position
        if idx < n_frames - 1:
            f = fp - idx
            mpos = pos[idx] * (1.0 - f) + pos[idx + 1] * f
        else:
            mpos = pos[-1]
        self.marker.geometry.positions.data[0] = mpos
        self.marker.geometry.positions.update_full()

        # --- trail: colormap + alpha fade
        tpos = self.line.geometry.positions.data
        tcol = self.line.geometry.colors.data

        start = max(0, idx - self._trail_len + 1)
        n = idx + 1 - start

        if n > 0:
            tpos[:n] = pos[start : idx + 1]
            t_lin = np.linspace(0.0, 1.0, n)
            cmap_colors = self._cmap(t_lin).astype(np.float32)
            cmap_colors[:, 3] = t_lin ** self._fade_power
            tcol[:n] = cmap_colors

        tpos[n:] = np.nan
        self.line.geometry.positions.update_full()
        self.line.geometry.colors.update_full()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def remove_from(self, scene: gfx.Scene):
        """Remove all world-objects from the scene."""
        scene.remove(self.line)
        scene.remove(self.marker)
        if self.cloud is not None:
            scene.remove(self.cloud)
