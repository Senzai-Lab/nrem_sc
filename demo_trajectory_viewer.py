"""
Demo: TrajectoryViewer with synthetic toy data.

No real data files needed — generates two spirals to demonstrate:
  - Background scatter layer (color-coded by angle)
  - Animated trajectory with trail
  - Colorbar overlay
  - Keyboard playback controls

Usage:
    python demo_trajectory_viewer.py
"""

import numpy as np
import pynapple as nap
import matplotlib.pyplot as plt

from trajectoryViewer import TrajectoryViewer


def make_spiral(n_points: int, n_turns: float, radius: float, noise: float = 0.0):
    """Generate a 2D spiral as (x, y) arrays."""
    t = np.linspace(0, n_turns * 2 * np.pi, n_points)
    r = np.linspace(0.2, radius, n_points)
    x = r * np.cos(t) + np.random.randn(n_points) * noise
    y = r * np.sin(t) + np.random.randn(n_points) * noise
    return x, y, t


# === Generate toy data ===
np.random.seed(42)

# 1) Background scatter — a dense spiral colored by angle (simulates a
#    wake manifold colored by head direction).
n_bg = 5000
bg_x, bg_y, bg_angle = make_spiral(n_bg, n_turns=4, radius=3.0, noise=0.15)

# Wrap angle to [0, 1] for colormap
bg_angle_norm = (bg_angle % (2 * np.pi)) / (2 * np.pi)
cmap = plt.get_cmap("hsv")
bg_colors = cmap(bg_angle_norm)[:, :4].astype("float32")

# Fake timestamps (e.g. 32 Hz sampling for ~2.5 min)
bg_times = np.arange(n_bg) / 32.0
bg_tsd = nap.TsdFrame(
    t=bg_times,
    d=np.column_stack([bg_x, bg_y]),
    columns=["x", "y"],
)

# 2) Trajectory — a second spiral that the marker will animate along
#    (simulates a NREM embedding trajectory).
n_traj = 2000
traj_x, traj_y, _ = make_spiral(n_traj, n_turns=3, radius=2.5, noise=0.08)

traj_times = np.arange(n_traj) / 32.0
traj_tsd = nap.TsdFrame(
    t=traj_times,
    d=np.column_stack([traj_x, traj_y]),
    columns=["x", "y"],
)

# === Build the viewer ===
viewer = TrajectoryViewer(title="Toy Spiral Demo", size=(900, 900))

# Background scatter (behind, semi-transparent)
viewer.add_scatter(
    bg_tsd,
    colors=bg_colors,
    point_size=2,
    opacity=0.4,
    z_offset=-0.1,
    name="background",
)

# Trajectory scatter (lighter, smaller)
viewer.add_scatter(
    traj_tsd,
    cmap="viridis",
    point_size=1.5,
    opacity=0.2,
    name="trajectory_points",
)

# Animated trajectory
viewer.add_trajectory(
    traj_tsd,
    trail_length=40,
    marker_color="purple",
    marker_size=6,
    trail_thickness=2,
)

# Colorbar for the background angle coloring
viewer.add_colorbar(
    cmap="hsv",
    label="Angle (deg)",
    vmin=0,
    vmax=360,
    position="right",
    width=15,
    height=180,
    padding=25,
)

viewer.show()
