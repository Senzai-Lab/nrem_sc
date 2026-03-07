"""On-screen colour legend for pynaviz plots.

Adds ``gfx.Text`` labels to a plot's scene showing which colour
corresponds to which category, replicating the exact colour mapping
that ``color_by`` uses internally.

Usage
-----
>>> from nrem_sc.legend import add_legend
>>> add_legend(sleep_plot, metadata_name="state", cmap_name="Set2")
"""

from __future__ import annotations

import numpy as np
import pygfx as gfx
from matplotlib.pyplot import colormaps


def _get_color_mapping(
    metadata_values, cmap_name: str = "viridis"
) -> dict[str, tuple]:
    """Replicate pynaviz's ``map_non_color_string_array`` logic.

    Returns ``{label: (r, g, b, a)}`` in first-appearance order —
    the same order and colours that ``color_by`` assigns.
    """
    cmap = colormaps[cmap_name]
    vals = np.asarray(list(metadata_values.values()))
    unq_vals, index = np.unique(vals, return_index=True)
    unq_vals = unq_vals[np.argsort(index)]  # first-appearance order
    col_val = np.linspace(0, 1, len(unq_vals))
    return {v: cmap(c) for v, c in zip(unq_vals, col_val)}


def add_legend(
    plot,
    metadata_name: str,
    cmap_name: str = "viridis",
    *,
    font_size: float = 14,
    anchor: str = "top-right",
    outline_thickness: float = 0.15,
) -> list[gfx.Text]:
    """Add colour-keyed text labels to a pynaviz plot.

    Parameters
    ----------
    plot : pynaviz _BasePlot subclass
        The plot that ``color_by`` was called on.
    metadata_name : str
        Same metadata field passed to ``color_by``.
    cmap_name : str
        Same colormap name passed to ``color_by``.
    font_size : float
        Label font size in screen pixels.
    anchor : str
        ``"top-left"`` or ``"top-right"``.  Controls which corner the
        legend is placed in.
    outline_thickness : float
        Text outline for readability against dark backgrounds.

    Returns
    -------
    list[gfx.Text]
        The created text objects (useful for further customisation).
    """
    values = (
        dict(plot.data.get_info(metadata_name))
        if hasattr(plot.data, "get_info")
        else {}
    )
    if not values:
        return []

    color_map = _get_color_mapping(values, cmap_name)

    # Build labels in a screen-space overlay group
    labels: list[gfx.Text] = []
    for i, (name, rgba) in enumerate(color_map.items()):
        label = gfx.Text(
            text=f"{name}",
            screen_space=True,
            font_size=font_size,
            anchor=anchor,
            material=gfx.TextMaterial(
                color=gfx.Color(*rgba[:3]),
                outline_color="#000",
                outline_thickness=outline_thickness,
            ),
        )
        labels.append(label)

    # We position labels in _update_legend, which runs every frame
    _legend_group = gfx.Group()
    for lbl in labels:
        _legend_group.add(lbl)
    plot.scene.add(_legend_group)

    # Wrap animate to keep the legend pinned to a viewport corner
    from pynaviz.utils import get_plot_min_max

    right = "right" in anchor

    _orig_animate = plot.animate

    def _patched_animate():
        _orig_animate()
        xmin, xmax, ymin, ymax = get_plot_min_max(plot)
        x = xmax if right else xmin
        # Stack labels from top
        for i, lbl in enumerate(labels):
            lbl.local.position = (x, ymax - i * (ymax - ymin) * 0.07, 10)

    plot.animate = _patched_animate
    plot.canvas.request_draw(plot.animate)

    return labels
