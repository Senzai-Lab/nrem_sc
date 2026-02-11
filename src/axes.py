"""Dynamic rulers that auto-update to match the visible camera range.

Mimics the approach used by fastplotlib's ``Axes`` class but works directly
with a raw ``pygfx`` / ``rendercanvas`` setup (no fastplotlib dependency).

Designed for **orthographic 2-D** views — the rulers recompute their span
every frame by mapping screen corners → world coordinates, so they always
show the correct range and adapt to pan / zoom.

Usage
-----
>>> from src.axes import DynamicAxes
>>> axes = DynamicAxes(scene, renderer, camera, color="#888")
>>>
>>> def anim():
...     axes.update()                # call once per frame
...     renderer.render(scene, camera)
>>>
>>> canvas.request_draw(anim)
"""

from __future__ import annotations

import numpy as np
import pygfx as gfx
from pylinalg import vec_transform, vec_unproject


# ── helpers ────────────────────────────────────────────────────────────────

def _screen_to_world(
    screen_xy: tuple[float, float],
    viewport_rect: tuple[float, float, float, float],
    camera: gfx.OrthographicCamera,
) -> np.ndarray:
    """Map a screen-pixel position to world coordinates (z=0 plane).

    Parameters
    ----------
    screen_xy : (x, y)
        Pixel position relative to the *renderer* surface (not the viewport).
    viewport_rect : (x, y, w, h)
        Viewport rectangle in screen pixels.
    camera : OrthographicCamera
        The active camera.

    Returns
    -------
    np.ndarray, shape (3,)
        World-space position on the z=0 plane.
    """
    vx, vy, vw, vh = viewport_rect
    # position relative to viewport origin
    rx = screen_xy[0] - vx
    ry = screen_xy[1] - vy

    # normalised-device coordinates  ([-1, 1])
    ndc_x = rx / vw * 2 - 1
    ndc_y = -(ry / vh * 2 - 1)

    ndc = np.array([ndc_x, ndc_y, 0.0])
    ndc += vec_transform(camera.world.position, camera.camera_matrix)
    world = vec_unproject(ndc[:2], camera.camera_matrix)
    return np.asarray(world, dtype=float)


# ── main class ─────────────────────────────────────────────────────────────

class DynamicAxes:
    """Pair of ``pygfx.Ruler`` objects that track the visible viewport.

    Parameters
    ----------
    scene : gfx.Scene
        Scene to add the rulers to.
    viewport : gfx.Viewport  |  gfx.renderers.WgpuRenderer
        A ``Viewport`` (or the renderer itself, which acts as the root
        viewport) — used to query *rect* and *logical_size*.
    camera : gfx.OrthographicCamera
        Must be an orthographic camera (``fov == 0``).
    tick_size : float
        Height of tick marks in screen pixels.
    line_width : float
        Width of the ruler spine / tick lines.
    color : str | tuple
        Ruler colour (spine + ticks + labels).
    tick_format : str | callable
        Format string passed to ``pygfx.Ruler`` (e.g. ``"0.4g"``).
    tick_side_x / tick_side_y : ``"left"`` | ``"right"``
        Which side of the ruler line the ticks appear on.
    edge_fraction : float  (0–1)
        How far from the viewport edges the rulers are placed.
        0.0 = at the very edge, 0.1 = 10 % inward (default).
    grid : bool
        If *True*, add an infinite ``pygfx.Grid`` synchronised with the
        ruler tick spacing.
    grid_kwargs : dict | None
        Extra keyword arguments forwarded to :class:`pygfx.GridMaterial`.
    """

    def __init__(
        self,
        scene: gfx.Scene,
        viewport,
        camera: gfx.OrthographicCamera,
        *,
        tick_size: float = 8.0,
        line_width: float = 2.0,
        color: str | tuple = "#fff",
        tick_format: str = "0.4g",
        tick_side_x: str = "right",
        tick_side_y: str = "left",
        tick_marker: str = "tick",
        edge_fraction: float = 0.10,
        grid: bool = True,
        grid_kwargs: dict | None = None,
    ):
        self._viewport = viewport
        self._camera = camera
        self._edge_frac = edge_fraction

        ruler_kw = dict(
            tick_size=tick_size,
            line_width=line_width,
            color=color,
            tick_format=tick_format,
            tick_marker=tick_marker,
        )

        self._x = gfx.Ruler(tick_side=tick_side_x, **ruler_kw)
        self._y = gfx.Ruler(tick_side=tick_side_y, **ruler_kw)

        # pretty text rendering
        for ruler in (self._x, self._y):
            ruler.text.material.aa = True

        # must give initial positions to avoid crash on first render
        self._x.start_pos = (0, 0, 0)
        self._x.end_pos = (100, 0, 0)
        self._x.start_value = 0
        self._y.start_pos = (0, 0, 0)
        self._y.end_pos = (0, 100, 0)
        self._y.start_value = 0

        self._group = gfx.Group()
        self._group.add(self._x, self._y)

        # optional grid
        self._grid = None
        if grid:
            gkw = dict(
                major_step=100,
                minor_step=25,
                thickness_space="screen",
                major_thickness=1,
                minor_thickness=0.1,
                infinite=True,
                major_color="#ffffff3c",
                minor_color="#ffffff1c",
                axis_thickness=0,
            )
            if grid_kwargs:
                gkw.update(grid_kwargs)

            self._grid = gfx.Grid(
                geometry=None,
                material=gfx.GridMaterial(**gkw),
                orientation="xy",
            )
            self._grid.local.z = -1000  # behind everything
            self._group.add(self._grid)

        scene.add(self._group)

    # -- public properties ---------------------------------------------------

    @property
    def x(self) -> gfx.Ruler:
        """X-axis ruler."""
        return self._x

    @property
    def y(self) -> gfx.Ruler:
        """Y-axis ruler."""
        return self._y

    @property
    def grid(self) -> gfx.Grid | None:
        """The ``Grid`` world-object, or *None* if grids are disabled."""
        return self._grid

    @property
    def world_object(self) -> gfx.Group:
        """Parent group that contains both rulers (and grid)."""
        return self._group

    @property
    def visible(self) -> bool:
        return self._group.visible

    @visible.setter
    def visible(self, value: bool):
        self._group.visible = value

    # -- core update ---------------------------------------------------------

    def _viewport_rect(self):
        """Return (x, y, w, h) regardless of whether we have a Viewport
        or a bare renderer."""
        try:
            return self._viewport.rect          # gfx.Viewport
        except AttributeError:
            w, h = self._viewport.logical_size   # WgpuRenderer
            return (0, 0, w, h)

    def _logical_size(self):
        try:
            return self._viewport.logical_size
        except AttributeError:
            return self._viewport.logical_size

    def update(self):
        """Recompute ruler positions from the current camera / viewport state.

        Call this once per frame **before** ``renderer.render()``.
        """
        if not self.visible:
            return

        rect = self._viewport_rect()
        vx, vy, vw, vh = rect

        if vw < 1 or vh < 1:
            return  # canvas not ready yet

        # corners of the viewport in screen pixels
        screen_bl = (vx, vy + vh)          # bottom-left
        screen_tr = (vx + vw, vy)          # top-right

        world_min = _screen_to_world(screen_bl, rect, self._camera)
        world_max = _screen_to_world(screen_tr, rect, self._camera)

        if world_min is None or world_max is None:
            return

        # ruler intersection — 10 % from bottom-left by default
        ef = self._edge_frac
        screen_int = (vx + vw * ef, vy + vh * (1 - ef))
        intersection = _screen_to_world(screen_int, rect, self._camera)

        wx10, wy10 = intersection[0], intersection[1]

        # --- X ruler (horizontal) -------------------------------------------
        self._x.start_pos = (world_min[0], wy10, 0)
        self._x.end_pos   = (world_max[0], wy10, 0)
        self._x.start_value = world_min[0]
        stats_x = self._x.update(self._camera, self._logical_size())

        # --- Y ruler (vertical) ---------------------------------------------
        self._y.start_pos = (wx10, world_min[1], 0)
        self._y.end_pos   = (wx10, world_max[1], 0)
        self._y.start_value = world_min[1]
        stats_y = self._y.update(self._camera, self._logical_size())

        # --- Grid sync -------------------------------------------------------
        if self._grid is not None:
            sx, sy = stats_x["tick_step"], stats_y["tick_step"]
            self._grid.material.major_step = (sx, sy)
            self._grid.material.minor_step = (0.2 * sx, 0.2 * sy)
