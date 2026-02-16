"""
2-D multi-trajectory demo using Trail + PlaybackController.

Two Lissajous curves rendered in a flat orthographic view.
Demonstrates that Trail works identically with (N, 2) positions.

Camera controls (PanZoomController):
    Mouse drag              Pan
    Scroll                  Zoom

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

from nrem_sc.playback import PlaybackController
from nrem_sc.timetext import TimeText
from nrem_sc.trail import Trail

# ---------------------------------------------------------------------------
# Synthetic data: two 2-D Lissajous curves
# ---------------------------------------------------------------------------
n_frames = 8_000
dt = 0.033
times = np.arange(n_frames) * dt

# Curve A: Lissajous  x = sin(a*t + δ), y = sin(b*t)
a1, b1, delta1 = 3, 2, np.pi / 4
pos_a = np.column_stack([
    200 * np.sin(a1 * times * 0.3 + delta1),
    200 * np.sin(b1 * times * 0.3),
]).astype(np.float32)

# Curve B: different ratio
a2, b2, delta2 = 5, 4, np.pi / 2
pos_b = np.column_stack([
    200 * np.sin(a2 * times * 0.3 + delta2),
    200 * np.sin(b2 * times * 0.3),
]).astype(np.float32)

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
canvas = RenderCanvas(max_fps=60, title="Trail 2D – Multi-trajectory")
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()
scene.add(gfx.Background.from_color("#080808"))

# grid = gfx.Grid(
#     None,
#     gfx.GridMaterial(
#         major_step=100, minor_step=25,
#         thickness_space="world",
#         major_thickness=1.5, minor_thickness=0.3,
#         infinite=True,
#     ),
#     orientation="xy",
# )
# scene.add(grid)

# ---------------------------------------------------------------------------
# Trails (2-D positions → Trail pads z=0 automatically)
# ---------------------------------------------------------------------------
trail_a = Trail(
    pos_a, scene,
    trail_len=600,
    cmap="plasma",
    marker_color="#c10af3",
    marker="diamond",
    marker_size=10,
    line_thickness=3,
    cloud_alpha=0.25,
    cloud_brightness=0.10,
    cloud_size=0.8,
)

trail_b = Trail(
    pos_b, scene,
    trail_len=600,
    cmap="viridis",
    marker_color="#0af35a",
    marker="diamond",
    marker_size=10,
    line_thickness=3,
    cloud_alpha=0.25,
    cloud_brightness=0.10,
    cloud_size=0.8,
)

trails = [trail_a, trail_b]

# ---------------------------------------------------------------------------
# Controllers and overlays
# ---------------------------------------------------------------------------
camera = gfx.OrthographicCamera(600, 600, maintain_aspect=True)
camera.show_object(scene)
controller = gfx.PanZoomController(camera, register_events=renderer)

playback = PlaybackController(times, register_events=renderer)
overlay = TimeText(viewport=renderer, position="top-left")


def on_playback_update(pb: PlaybackController):
    for trail in trails:
        trail.update(pb.frame_index, pb.frame_position)


playback.add_handler(on_playback_update)
on_playback_update(playback)

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
