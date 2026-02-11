"""
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
import pandas as pd

from src.playback import PlaybackController
from src.timetext import TimeText
from src.trail import Trail

# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------
fname = r"R:\Basic_Sciences\Phys\SenzaiLab\Tuguldur\260209.csv"
data = pd.read_csv(fname, skiprows=7)
t = data['Time (Seconds)'].to_numpy()
xyz = data[['X.1', 'Z.1']].to_numpy()
cage_radius = 1.152850207254279e+03
cage_center = (21.292528393641192, 23.629275777896162, 0)
# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
canvas = RenderCanvas(max_fps=60)
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()
scene.add(gfx.Background.from_color("#141414"))

# Cage circle
circle = gfx.Line(
    gfx.Geometry(
        positions=[(cage_radius * np.cos(theta) + cage_center[0],
                    cage_radius * np.sin(theta) + cage_center[1],
                    0) for theta in np.linspace(0, 2 * np.pi, 128)],
    ),
    gfx.LineMaterial(color="#378A8FEB", thickness=5),
)
circle.local.position = cage_center
scene.add(circle)

# Axes
axes = gfx.AxesHelper(size=50, thickness=1)
axes.local.position = cage_center
scene.add(axes)

# ---------------------------------------------------------------------------
# Trails (2-D positions → Trail pads z=0 automatically)
# ---------------------------------------------------------------------------
trail = Trail(
    xyz, scene,
    trail_len=600,
    cmap="plasma",
    marker_color="#0a8ef3",
    marker="diamond",
    marker_size=15,
    line_thickness=3,
    cloud_alpha=0.1,
    cloud_brightness=0.60,
    cloud_size=0.4,
)

# ---------------------------------------------------------------------------
# Controllers and overlays
# ---------------------------------------------------------------------------
camera = gfx.OrthographicCamera(maintain_aspect=True)
camera.show_object(scene)
controller = gfx.PanZoomController(camera, register_events=renderer)
    
playback = PlaybackController(t, register_events=renderer)
overlay = TimeText(viewport=renderer, position="top-left")


def on_playback_update(pb: PlaybackController):
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
