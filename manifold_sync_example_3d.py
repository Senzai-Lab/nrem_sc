import pynapple as nap
import pynaviz as viz
from pynaviz.controller_group import ControllerGroup

# Import the 3D viewer
from manifold_viewer_3d_fps import TrajectoryViewer3D
# Import the controller from the 2D viewer file (reusable sync logic)
from trajectoryViewer import TrajectoryController

from src.constants import INTERIM_DATA_PATH, PROCESSED_DATA_PATH
import numpy as np
import matplotlib.colors as mcolors

# === CONFIG ===
unit_id = "116b"
data_dir = INTERIM_DATA_PATH / unit_id / "tmp"

# === LOAD DATA ===
# Note: Using 3D manifold data files where appropriate
hd_spikes_shifted = nap.load_file(data_dir / "hd_spikes_shifted.npz")
manifold_shifted = nap.load_file(data_dir / "manifold_3d_shifted_orig.npz")
manifold_openfield = nap.load_file(data_dir / "manifold_3d_shifted_wake.npz")
hd_angle_openfield = nap.load_file(PROCESSED_DATA_PATH / unit_id / "angle_openfield2.npz").to_numpy()
sleep_states_shifted = nap.load_file(data_dir / "sleep_shifted.npz")

# HD angle to RGBA colors
# Normalize
hsv_colors = np.ones((len(hd_angle_openfield), 3))
hsv_colors[:, 0] = hd_angle_openfield / 360.0

# HSV to RGBA
rgb_colors = mcolors.hsv_to_rgb(hsv_colors)
colors_rgba = np.column_stack((rgb_colors, np.ones(len(rgb_colors))))

print(f"Loaded {len(hd_spikes_shifted)} units")
print(f"Loaded manifold: {len(manifold_shifted)} points")
print(f"Time range: [{manifold_shifted.times()[0]:.1f}, {manifold_shifted.times()[-1]:.1f}] s")
print(f"Sleep states: {len(sleep_states_shifted)} intervals")

# === CREATE PLOTS ===
# Spike raster plot from pynaviz
spike_plot = viz.PlotTsGroup(hd_spikes_shifted, index=0)
spike_plot.sort_by(metadata_name='preferred_angle', mode='descending')
spike_plot.color_by(metadata_name='preferred_angle', cmap_name='hsv')

# Sleep states (IntervalSet) plot from pynaviz
sleep_plot = viz.PlotIntervalSet(sleep_states_shifted, index=1)
sleep_plot.color_by(metadata_name='state', cmap_name='Set2')

# Manifold viewer 3D
viewer = TrajectoryViewer3D(title="3D Manifold Viewer Sync")

# Attach controller for synchronization (pynaviz needs .controller property)
viewer.controller = TrajectoryController(viewer, viewer.renderer)

# Add scatter layers
viewer.add_scatter(
    data=manifold_openfield,
    colors=colors_rgba,
    point_size=3,
    name="wake")

viewer.add_scatter(
    data=manifold_shifted,
    cmap=None,
    point_size=2,
    opacity=0.1,
    name="nrem")

# Add trajectory for animation
viewer.add_trajectory(manifold_shifted, trail_length=25)

# === SYNCHRONIZE WITH CONTROLLERGROUP ===
# Create a ControllerGroup to synchronize time across all plots
# Note: ControllerGroup expects objects with .controller and .renderer attributes
cg = ControllerGroup(
    plots=[spike_plot, sleep_plot],
    interval=(0, manifold_shifted.times()[-1])
)

# Add the manifold viewer to the controller group
# This allows it to receive sync events when you pan/zoom the spike raster
cg.add(viewer, controller_id=2)

# === SHOW ===
print("\nControls:")
print("  - Pan/zoom in spike raster or sleep states to navigate time")
print("  - Manifold marker will follow the current time")
print("  - Use 3D viewer controls (WASD, Mouse) to move camera")
print("  - Press 'G' to toggle follow mode")
print("  - Press 'F' to fly to the current marker")
print("  - Press 'space' to play/pause")
print("  - 'C' in manifold viewer to toggle trail")

# For plots (not widgets), we need to run the event loop
from rendercanvas.auto import loop
loop.run()
