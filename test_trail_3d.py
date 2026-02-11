"""
3-D multi-trajectory demo using Trail + PlaybackController.

Three synthetic spirals (different phase / colour) share a single playback
timeline.  Demonstrates the Trail helper for 3-D scenes.

Camera controls (FlyController):
    Mouse drag              Rotate camera
    W / A / S / D           Move forward / left / back / right
    Space / Shift           Move up / down
    Q / E                   Roll
    Scroll                  Adjust camera speed

Playback controls (PlaybackController):
    P                       Play / pause
    Right Click             Play / pause
    Arrow Up / Down         Speed up / down  (2x)
    Ctrl + Scroll Up/Down   Speed up / down  (2x)
    Arrow Right / Left      Step forward / backward
    Home / End              Jump to start / end
"""

from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx
import numpy as np

from src.playback import PlaybackController
from src.timetext import TimeText
from src.trail import Trail

# ---------------------------------------------------------------------------
# Synthetic data: three 3-D spirals with different phases
# ---------------------------------------------------------------------------
n_frames = 10_000
dt = 0.016  # ~30 Hz
times = np.arange(n_frames) * dt

trajectories = []
cmaps = ["plasma", "viridis", "cool"]
marker_colors = ["#c10af3", "#0af35a", "#f3a00a"]
phase_offsets = [0, 2 * np.pi / 3, 4 * np.pi / 3]

for phase in phase_offsets:
    theta = times * 2 * np.pi / 10 + phase
    radius = 50 + 10 * np.sin(times * 0.3 + phase)
    positions = np.column_stack([
        radius * np.cos(theta),
        times * 1.5 - 100,
        radius * np.sin(theta),
    ]).astype(np.float32)
    trajectories.append(positions)

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
canvas = RenderCanvas(max_fps=60, title="Trail 3D – Multi-trajectory")
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()
scene.add(gfx.Background.from_color("#080808"))

# axes = gfx.AxesHelper(size=30, thickness=8)
# axes.local.y = -110
# scene.add(axes)

grid = gfx.Grid(
    None,
    gfx.GridMaterial(
        major_step=100, minor_step=10,
        thickness_space="world",
        major_thickness=1, minor_thickness=0.1,
        infinite=True,
    ),
    orientation="xz",
)
grid.local.y = -120
scene.add(grid)

# ---------------------------------------------------------------------------
# Trails
# ---------------------------------------------------------------------------
trails = []
for pos, cmap, mcol in zip(trajectories, cmaps, marker_colors):
    t = Trail(
        pos, scene,
        trail_len=400,
        cmap=cmap,
        marker_color=mcol,
        marker="circle",
        marker_size=5,
        cloud=True,
        cloud_alpha=0
    )
    trails.append(t)

# ---------------------------------------------------------------------------
# Controllers and overlays
# ---------------------------------------------------------------------------
camera = gfx.PerspectiveCamera(70)
camera.show_object(scene)
fly = gfx.FlyController(camera, register_events=renderer)

playback = PlaybackController(times, register_events=renderer)
overlay = TimeText(viewport=renderer, position="top-left")


def on_playback_update(pb: PlaybackController):
    for trail in trails:
        trail.update(pb.frame_index, pb.frame_position)


playback.add_handler(on_playback_update)
on_playback_update(playback)  # set initial state

# ---------------------------------------------------------------------------
# Animation loop
# ---------------------------------------------------------------------------
def anim():
    renderer.render(scene, camera)
    overlay.update(playback)
    overlay.render(flush=False)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(anim)
    loop.run()
