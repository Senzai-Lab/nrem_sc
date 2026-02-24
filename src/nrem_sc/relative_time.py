"""Relative-time tick formatter for pygfx Rulers.

When zoomed into a narrow window of a long recording (e.g. viewing
4823.40–4823.95 s), absolute tick labels like ``4823.441`` are
unnecessarily long.  This module provides a formatter that shows only
the *residual* (e.g. ``.441``) and renders the base offset
(``+ 4823 s``) as a separate ``gfx.Text`` label.

Works with both pynaviz plot rulers and custom ``DynamicAxes`` rulers.

Usage – pynaviz plots
---------------------
>>> from nrem_sc.relative_time import apply_relative_time
>>> apply_relative_time(spike_plot)          # patches one plot
>>> apply_relative_time(sleep_plot, threshold=0.005)

Usage – DynamicAxes
-------------------
>>> from nrem_sc.relative_time import RelativeTimeFormatter
>>> rtf = RelativeTimeFormatter(scene)
>>> axes = DynamicAxes(scene, renderer, camera, tick_format=rtf.tick_format)
>>> # in anim loop, after axes.update():
>>> rtf.update_label(world_xmin, world_ymin)
"""

from __future__ import annotations

import numpy as np
import pygfx as gfx


class RelativeTimeFormatter:
    """Format ruler ticks relative to a rounded offset.

    Parameters
    ----------
    scene : gfx.Scene
        Scene to which the offset label is added.
    threshold : float
        When ``visible_span / |center_time|`` drops below this value,
        relative mode activates.  Default ``0.001``.
    font_size : float
        Font size for the floating offset label.
    label_color : str
        Colour of the offset label text.
    """

    def __init__(
        self,
        scene: gfx.Scene,
        *,
        threshold: float = 0.001,
        font_size: float = 12,
        label_color: str = "#ffcc00",
    ):
        self._scene = scene
        self._threshold = threshold
        self._offset: float = 0.0
        self._active: bool = False

        self._label = gfx.Text(
            text="",
            screen_space=True,
            font_size=font_size,
            anchor="bottom-left",
            material=gfx.TextMaterial(
                color=label_color,
                outline_color="#000",
                outline_thickness=0.2,
            ),
        )
        self._label.visible = False
        scene.add(self._label)

    # -- callable for ruler.tick_format ------------------------------------

    def tick_format(self, value: float, min_value: float, max_value: float) -> str:
        """Tick formatter – pass this to ``ruler.tick_format``."""
        span = max_value - min_value
        center = (min_value + max_value) * 0.5

        if abs(center) > 1e-9 and span / abs(center) < self._threshold:
            # Choose a "nice" offset: largest power-of-10 ≤ center,
            # rounded down to that power.
            magnitude = 10 ** np.floor(np.log10(abs(center)))
            offset = np.floor(center / magnitude) * magnitude
            self._offset = offset
            self._active = True

            residual = value - offset
            # Precision adapts to the visible span
            if span < 0.01:
                return f"{residual:.4f}"
            elif span < 0.1:
                return f"{residual:.3f}"
            elif span < 1:
                return f"{residual:.2f}"
            elif span < 10:
                return f"{residual:.1f}"
            else:
                return format(residual, "0.4g")
        else:
            self._active = False
            return format(value, "0.4g")

    # -- per-frame label update --------------------------------------------

    def update_label(self, world_xmin: float, world_ymin: float) -> None:
        """Reposition and update the offset text.  Call once per frame."""
        if self._active:
            self._label.visible = True
            self._label.set_text(f"+ {self._offset:.6g} s")
            self._label.local.position = (world_xmin, world_ymin, 10)
        else:
            self._label.visible = False


# ── convenience helper for pynaviz plots ──────────────────────────────────

def apply_relative_time(plot, *, threshold: float = 0.002) -> RelativeTimeFormatter:
    """Patch a pynaviz ``_BasePlot`` to use relative-time tick labels.

    Parameters
    ----------
    plot : pynaviz _BasePlot subclass
        Any of ``PlotTsGroup``, ``PlotIntervalSet``, ``PlotTsd``,
        ``PlotTsdFrame``, etc.
    threshold : float
        See :class:`RelativeTimeFormatter`.

    Returns
    -------
    RelativeTimeFormatter
        The formatter instance (useful if you need to tweak settings).
    """
    from pynaviz.utils import get_plot_min_max

    rtf = RelativeTimeFormatter(plot.scene, threshold=threshold)
    plot.ruler_x.tick_format = rtf.tick_format

    _orig_animate = plot.animate

    def _patched_animate():
        _orig_animate()
        xmin, _, ymin, _ = get_plot_min_max(plot)
        rtf.update_label(xmin, ymin)

    plot.animate = _patched_animate
    plot.canvas.request_draw(plot.animate)
    return rtf
