"""
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
import pynapple as nap
import pandas as pd

from src.playback import PlaybackController
from src.timetext import TimeText
from src.trail import Trail

# Load data
# fname = '/Volumes/fsmresfiles/Basic_Sciences/Phys/SenzaiLab/Tuguldur/points_3d.csv'
fname = r"R:\Basic_Sciences\Phys\SenzaiLab\Tuguldur\260209.csv"
data = pd.read_csv(fname, skiprows=7)
t = data['Time (Seconds)'].to_numpy()
xyz = data[['X.1', 'Y.1', 'Z.1']].to_numpy()
# center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
# xyz = xyz - center

# Swap Y <-> Z (pygfx expects y-up)
# xyz = xyz[:, [0, 2, 1]]


canvas = RenderCanvas(max_fps=60, title="Trail 3D - Multi-trajectory")
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()
scene.add(gfx.Background.from_color("#141414"))
scene.add(gfx.AmbientLight(intensity=1.0))

# Cylinder
cylinder_center = (21.292528393641192, -5.464206080862482, 23.629275777896162)
cage_height = 2.068528092539011e+03
cage_radius = 1.152850207254279e+03
# cage_radius = np.max(np.abs(xyz[:, [0, 2]])) * 1.02  # XZ radius + 2% margin
# cage_height = np.ptp(xyz[:, 1]) * 1.1                 # Y extent + 10% margin

cylinder_geo = gfx.cylinder_geometry(
    radius_bottom=cage_radius,
    radius_top=cage_radius,
    height=cage_height,
    radial_segments=64,
    height_segments=1,
    open_ended=True,
)
cylinder_mesh = gfx.Mesh(
    cylinder_geo,
    gfx.MeshBasicMaterial(color="#378A8FEB", side="both"),
)

cylinder_mesh.local.position = cylinder_center
cylinder_mesh.local.euler_x = -np.pi / 2
scene.add(cylinder_mesh)

# axes helper
axes = gfx.AxesHelper(size=30, thickness=2)
axes.local.position = cylinder_center
scene.add(axes)


# Grid
grid = gfx.Grid(
    None,
    gfx.GridMaterial(
        major_step=100, minor_step=10,
        thickness_space="world",
        major_thickness=0.5, minor_thickness=0.1,
        infinite=True,
    ),
    orientation="xz",
)
grid.local.y = cylinder_center[1]
scene.add(grid)

# Trail
trail = Trail(
    xyz, scene,
    trail_len=400,
    cmap="plasma",
    marker_color="#670af3",
    marker="circle",
    marker_size=5,
    cloud=False,
)


camera = gfx.PerspectiveCamera(70)
# camera.local.position = (xyz[0, 0], xyz[0, 1] + cage_height*4, xyz[0, 2]*1.1)
camera.show_object(scene, view_dir=(0, -1, 0.5))
fly = gfx.FlyController(camera, register_events=renderer)

playback = PlaybackController(t, register_events=renderer)
overlay = TimeText(viewport=renderer, position="top-left")


def on_playback_update(pb: PlaybackController):
    trail.update(pb.frame_index, pb.frame_position)


playback.add_handler(on_playback_update)
on_playback_update(playback)  # set initial state


def anim():
    renderer.render(scene, camera)
    overlay.update(playback)
    overlay.render(flush=False)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(anim)
    loop.run()
