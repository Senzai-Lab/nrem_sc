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

Utility:
    C                       Print camera position & direction
"""

from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx
import numpy as np
from pylinalg import vec_transform_quat

from nrem_sc.playback import PlaybackController
from nrem_sc.timetext import TimeText
from nrem_sc.trail import Trail


def print_camera(camera):
    """Print camera position and orientation to the console."""
    pos = camera.local.position
    rot = camera.local.rotation                        # (x, y, z, w) quaternion
    forward = vec_transform_quat((0, 0, -1), rot)     # camera looks along -Z
    print(f"cam pos  = ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    print(f"view_dir = ({forward[0]:.3f}, {forward[1]:.3f}, {forward[2]:.3f})")
    print(f"# --- paste to restore view ---")
    print(f"camera.local.position = ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    print(f"camera.local.rotation = ({rot[0]:.6f}, {rot[1]:.6f}, {rot[2]:.6f}, {rot[3]:.6f})")

# ---------------------------------------------------------------------------
# Lorenz attractor – 5 trajectories from nearby initial conditions
# ---------------------------------------------------------------------------
def lorenz(state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    x, y, z = state
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

n_frames = 10_000
dt_integration = 0.01        # integration step
dt = dt_integration           # playback dt  (≈ 200 Hz, adjust to taste)
times = np.arange(n_frames) * dt

# Five slightly different starting points → diverging trajectories
rng = np.random.default_rng(10)
base_ic = np.array([1.0, 1.0, 1.0])
initial_conditions = [base_ic + rng.normal(scale=0.01, size=3) for _ in range(5)]

trajectories = []
for ic in initial_conditions:
    pts = np.empty((n_frames, 3), dtype=np.float64)
    state = ic.copy()
    for i in range(n_frames):
        pts[i] = state
        # RK4 integration
        k1 = lorenz(state)
        k2 = lorenz(state + 0.5 * dt_integration * k1)
        k3 = lorenz(state + 0.5 * dt_integration * k2)
        k4 = lorenz(state + dt_integration * k3)
        state = state + (dt_integration / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    trajectories.append(pts.astype(np.float32))

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
canvas = RenderCanvas(max_fps=60, title="Trail 3D - Lorenz Attractor")
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()
scene.add(gfx.Background.from_color("#141414"))

# axes = gfx.AxesHelper(size=30, thickness=4)
# # axes.local.y = -110
# scene.add(axes)

grid = gfx.Grid(
    None,
    gfx.GridMaterial(
        major_step=100, minor_step=10,
        thickness_space="screen",
        major_thickness=0.5, minor_thickness=0.1,
        infinite=True,
    ),
    orientation="xy",
)
scene.add(grid)

# ---------------------------------------------------------------------------
# Trails
# ---------------------------------------------------------------------------
trails = []
cmaps = ["plasma", "viridis", "cool", "spring", "winter"]
marker_colors = ["#ecb756", "#92edb2", "#c95ee6", "#e65c5c", "#5cc8e6"]
for pos, cmap, mcol in zip(trajectories, cmaps, marker_colors):
    t = Trail(
        pos, scene,
        trail_len=600,
        cmap=cmap,
        line_thickness=2,
        marker_color=mcol,
        marker="circle",
        marker_size=2,
        cloud=False,
        # cloud_alpha=0.01,
        # cloud_brightness=0.0,
        # cloud_size=0.1,
    )
    trails.append(t)    

# ---------------------------------------------------------------------------
# Controllers and overlays
# ---------------------------------------------------------------------------
camera = gfx.PerspectiveCamera(70)
# camera.show_object(scene)
camera.local.position = (-36.832, 24.010, 39.731)
camera.local.rotation = (0.299078, -0.562985, -0.682973, 0.356601)
fly = gfx.FlyController(camera, register_events=renderer)

playback = PlaybackController(times, register_events=renderer)
overlay = TimeText(viewport=renderer, position="top-left")


@renderer.add_event_handler("key_down")
def _on_key(event):
    if event.key == "c":
        print_camera(camera)


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
