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
    ] / [                   Speed up / down  (2x)
    Arrow Up / Down         Speed up / down  (2x)
    Ctrl + Scroll Up/Down   Speed up / down  (2x)
    Arrow Right / Left      Step forward / backward
    Home / End              Jump to start / end
"""

from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx
import numpy as np
from matplotlib import colormaps

from nrem_sc.playback import PlaybackController
from nrem_sc.timetext import TimeText

# ---------------------------------------------------------------------------
# Synthetic data: 3-D spiral with varying radius
# ---------------------------------------------------------------------------
n_frames = 10_000
dt = 0.033  # ~30 Hz
times = np.arange(n_frames) * dt

theta = times * 2 * np.pi / 10  # 1 revolution per 10 s
radius = 50 + 10 * np.sin(times * 0.3)
positions = np.column_stack([
    radius * np.cos(theta),
    times * 1.5 - 100,          # slow linear drift in Y
    radius * np.sin(theta),
]).astype(np.float32)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
canvas = RenderCanvas(max_fps=60, title="PlaybackController")
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()
scene.add(gfx.Background.from_color("#080808"))

axes = gfx.AxesHelper(size=30, thickness=8)
axes.local.y = -110
scene.add(axes)

# Grid
grid = gfx.Grid(
    None,
    gfx.GridMaterial(
        major_step=100, minor_step=10,
        thickness_space="world",
        major_thickness=2, minor_thickness=0.1,
        infinite=True,
    ),
    orientation="xz",
)
grid.local.y = -120
scene.add(grid)

# Dim static cloud (entire trajectory)
colors_dim = np.zeros((n_frames, 4), dtype=np.float32)
colors_dim[:, :3] = 0.15
colors_dim[:, 3] = 0.5
scene.add(gfx.Points(
    gfx.Geometry(positions=positions, colors=colors_dim),
    gfx.PointsMaterial(size=1, color_mode="vertex", size_space="world"),
))

# Active marker
marker = gfx.Points(
    gfx.Geometry(positions=np.zeros((1, 3), dtype=np.float32)),
    gfx.PointsMarkerMaterial(
        marker='triangle_down',
        size=10,
        color="#c10af3",
        size_space="world"),
)

scene.add(marker)

# Trail (vertex-colored for fade + colormap)
TRAIL_LEN = 500
TRAIL_CMAP = colormaps['plasma']          # any matplotlib cmap
trail = gfx.Line(
    gfx.Geometry(
        positions=np.full((TRAIL_LEN, 3), np.nan, dtype=np.float32),
        colors=np.zeros((TRAIL_LEN, 4), dtype=np.float32),
    ),
    gfx.LineMaterial(thickness=2.5, color_mode="vertex", aa=True),
)
scene.add(trail)


# ---------------------------------------------------------------------------
# Controllers and overlays
# ---------------------------------------------------------------------------
camera = gfx.PerspectiveCamera(70)
camera.show_object(scene)
fly = gfx.FlyController(camera, register_events=renderer)

playback = PlaybackController(times, register_events=renderer)
overlay = TimeText(viewport=renderer, position='top-left')

# ---------------------------------------------------------------------------
# Playback handler – update marker + trail whenever time changes
# ---------------------------------------------------------------------------
def on_playback_update(pb: PlaybackController):
    idx = pb.frame_index
    fp = pb.frame_position

    # Interpolated marker position
    if idx < n_frames - 1:
        f = fp - idx
        pos = positions[idx] * (1 - f) + positions[idx + 1] * f
    else:
        pos = positions[-1]
    marker.geometry.positions.data[0] = pos
    marker.geometry.positions.update_full()

    # Trail: colormap + alpha fade
    tpos = trail.geometry.positions.data
    tcol = trail.geometry.colors.data
    start = max(0, idx - TRAIL_LEN + 1)
    n = idx + 1 - start
    if n > 0:
        tpos[:n] = positions[start:idx + 1]
        t_lin = np.linspace(0.0, 1.0, n)          # 0=oldest, 1=newest
        cmap_colors = TRAIL_CMAP(t_lin).astype(np.float32)  # (n, 4)
        cmap_colors[:, 3] = t_lin ** 0.6           # fade: old → transparent
        tcol[:n] = cmap_colors
    tpos[n:] = np.nan
    trail.geometry.positions.update_full()
    trail.geometry.colors.update_full()


playback.add_handler(on_playback_update)
on_playback_update(playback)  # set initial state
# ---------------------------------------------------------------------------
# Animation loop
# ---------------------------------------------------------------------------
def anim():
    # 1) Render main scene (triggers before_render → playback.tick())
    renderer.render(scene, camera)

    # 2) Update overlay AFTER tick so text reflects current frame
    overlay.update(playback)
    overlay.render(flush=False)
    
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(anim)
    loop.run()
