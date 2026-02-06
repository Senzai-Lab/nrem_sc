"""
Example: Synchronized manifold viewer with pynaviz spike raster and sleep states.

This demonstrates how to use ManifoldViewer with pynaviz's ControllerGroup
for synchronized time navigation across multiple plots.
"""

import pynapple as nap
import pynaviz as viz
from pynaviz.controller_group import ControllerGroup

import manifold_viewer
from src.constants import INTERIM_DATA_PATH, PROCESSED_DATA_PATH
from manifold_viewer import ManifoldViewer
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib as mpl

# Register HUSL as a matplotlib colormap so pynaviz can use it
husl_cmap = ListedColormap(sns.color_palette("husl", 256), name="husl")
mpl.colormaps.register(husl_cmap)

# === CONFIG ===
unit_id = "116b"
data_dir = INTERIM_DATA_PATH / unit_id / "tmp"

# === LOAD DATA ===
hd_spikes_shifted = nap.load_file(data_dir / "hd_spikes_shifted.npz")
manifold_shifted = nap.load_file(data_dir / "manifold_shifted.npz")
manifold_openfield = nap.load_file(PROCESSED_DATA_PATH / unit_id / "manifold_openfield2.npz")
hd_angle_openfield = nap.load_file(PROCESSED_DATA_PATH / unit_id / "angle_openfield2.npz").to_numpy()
sleep_states_shifted = nap.load_file(data_dir / "sleep_shifted.npz")
decoded_hd = nap.load_file(PROCESSED_DATA_PATH / unit_id / "decoded_hd_nrem.npz")
decoded_hd = nap.Tsd(t=decoded_hd.t - 30000, d=decoded_hd.values)  # Shift back
print(decoded_hd)

# HD angle to RGBA colors using HUSL palette
n_colors = 256
husl_palette = sns.color_palette("husl", n_colors)
# Map angles (0-360) to palette indices
angle_indices = ((hd_angle_openfield / 360.0) * (n_colors - 1)).astype(int)
rgb_colors = np.array([husl_palette[i] for i in angle_indices])
colors_rgba = np.column_stack((rgb_colors, np.ones(len(rgb_colors))))

print(f"Loaded {len(hd_spikes_shifted)} units")
print(f"Loaded manifold: {len(manifold_shifted)} points")
print(f"Time range: [{manifold_shifted.times()[0]:.1f}, {manifold_shifted.times()[-1]:.1f}] s")
print(f"Sleep states: {len(sleep_states_shifted)} intervals")

# === CREATE PLOTS ===
# Spike raster plot from pynaviz
spike_plot = viz.PlotTsGroup(hd_spikes_shifted, index=0)
spike_plot.sort_by(metadata_name='preferred_angle', mode='ascending')
spike_plot.color_by(metadata_name='preferred_angle', cmap_name='husl')

# Sleep states (IntervalSet) plot from pynaviz
sleep_plot = viz.PlotIntervalSet(sleep_states_shifted, index=1)
sleep_plot.color_by(metadata_name='state', cmap_name='Set2')

# Decoded HD plot
decoded_hd_plot = viz.PlotTsd(np.unwrap(decoded_hd), index=2)

# Manifold viewer
viewer = ManifoldViewer()


# Add the point cloud
viewer.add_point_cloud(
    data=manifold_openfield,
    colors=colors_rgba,
    point_size=2,
    name="wake")

viewer.add_point_cloud(
    data=manifold_shifted,
    point_size=1,
    opacity=0.1,
    name="nrem")

# Add colorbar
viewer.add_colorbar(cmap="husl", label="HD (Deg)", vmin=0, vmax=360, 
                    position="bottom", width=150, height=15, padding=30)


# Setup trajectory for animation
viewer.setup_trajectory(manifold_shifted, trail_length=25)

# === SYNCHRONIZE WITH CONTROLLERGROUP ===
# Create a ControllerGroup to synchronize time across all plots
# Note: ControllerGroup expects objects with .controller and .renderer attributes
cg = ControllerGroup(
    plots=[spike_plot, sleep_plot, decoded_hd_plot],
    interval=(0, manifold_shifted.times()[-1])
)

# Add the manifold viewer to the controller group
# This allows it to receive sync events when you pan/zoom the spike raster
cg.add(viewer, controller_id=3)

# === SHOW ===
print("\nControls:")
print("  - Pan/zoom in spike raster or sleep states to navigate time")
print("  - Manifold marker will follow the current time")
print("  - Use manifold viewer controls (Space, arrows) for playback")
print("  - 'C' in manifold viewer to toggle trail")

# For plots (not widgets), we need to run the event loop
from rendercanvas.auto import loop
loop.run()
