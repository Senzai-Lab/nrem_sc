import numpy as np
import pynapple as nap
import pynaviz as viz
import pygfx as gfx
import seaborn as sns
import matplotlib as mpl

from matplotlib.colors import ListedColormap
from rendercanvas.auto import RenderCanvas, loop
from pynaviz.controller_group import ControllerGroup
from sklearn import manifold

from nrem_sc.constants import PROCESSED_DATA_PATH
from nrem_sc.playback import PlaybackController
from nrem_sc.timetext import TimeText
from nrem_sc.trail import Trail

# Register HUSL as a matplotlib colormap so pynaviz can use it
husl_cmap = ListedColormap(sns.color_palette("husl", 256), name="husl")
mpl.colormaps.register(husl_cmap)

# === CONFIG ===
unit_id = "116b"
data_dir = PROCESSED_DATA_PATH / unit_id

# === LOAD DATA ===
sleep_states    = nap.load_file(PROCESSED_DATA_PATH / unit_id / "sleep.npz")
hd_spikes       = nap.load_file(PROCESSED_DATA_PATH / unit_id / "hd_spikes_filtered.npz")
manifold        = nap.load_file(PROCESSED_DATA_PATH / unit_id / "combined_umap_prettx.npz")
angle           = np.load(PROCESSED_DATA_PATH / unit_id / "combined_angle_prettx.npy")
classified_hd = nap.load_file(PROCESSED_DATA_PATH / unit_id / 'state_decoded.npz')

# HD angle → RGBA colors using HUSL palette
n_colors = 256
husl_palette = sns.color_palette("husl", n_colors)
angle_indices = ((angle / 360.0) * (n_colors - 1)).astype(int)
rgb_colors = np.array([husl_palette[i] for i in angle_indices])
colors_rgba = np.column_stack((rgb_colors, np.ones(len(rgb_colors)))).astype("float32")

print(f"Loaded {len(hd_spikes)} units")
print(f"Loaded manifold: {len(manifold)} points")
print(f"Time range: [{manifold.times()[0]:.1f}, {manifold.times()[-1]:.1f}] s")
print(f"Sleep states: {len(sleep_states)} intervals")

# === PYNAVIZ PLOTS ===
spike_plot = viz.PlotTsGroup(hd_spikes, index=0)
spike_plot.sort_by(metadata_name='preferred_angle', mode='ascending')
spike_plot.color_by(metadata_name='preferred_angle', cmap_name='husl')

sleep_plot = viz.PlotIntervalSet(sleep_states, index=1)
sleep_plot.color_by(metadata_name='state', cmap_name='Set2')

decoded_hd_plot = viz.PlotTsd(np.deg2rad(classified_hd['position']), index=2)

tsdf = classified_hd[['continuous', 'fragmented', 'stationary']]
tsdf.set_info(color_id=[0, 1, 2])
state_prob_plot = viz.PlotTsdFrame(tsdf, index=3)
state_prob_plot.color_by(metadata_name='color_id', cmap_name='Set2')

# # State-probability legend
# set2_cmap = mpl_colormaps['Set2']
# n_cols = len(tsdf.columns)
# legend_norm = np.arange(n_cols) / (n_cols - 1)
# t_start = tsdf.times()[0]
# x_legend = t_start - (tsdf.times()[-1] - t_start) * 0.02

# for i, col_name in enumerate(tsdf.columns):
#     rgba = set2_cmap(legend_norm[i])
#     label = gfx.Text(
#         text=f"— {col_name}",
#         screen_space=True,
#         font_size=14,
#         anchor="top-left",
#         material=gfx.TextMaterial(
#             color=gfx.Color(*rgba[:3]),
#             outline_color="#000",
#             outline_thickness=0.15,
#         ),
#     )
#     label.local.position = (x_legend, 1 + i * 0.05, 0)
#     state_prob_plot.scene.add(label)

# === MANIFOLD VIEWER (pygfx scene + Trail + PlaybackController) ===
canvas = RenderCanvas(max_fps=60, title="Manifold Viewer")
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()
scene.add(gfx.Background.from_color("#0a0a0a"))

# camera = gfx.OrthographicCamera(maintain_aspect=True)
# pan_zoom = gfx.PanZoomController(camera, register_events=renderer)
camera = gfx.PerspectiveCamera(70)
fly = gfx.FlyController(camera, register_events=renderer)

# --- Wake scatter---
INDEX = 301863 # Homecage samples length

wake_cloud = gfx.Points(
    gfx.Geometry(positions=manifold.values[INDEX:, :].astype("float32"), colors=colors_rgba),
    gfx.PointsMaterial(size=2, color_mode="vertex", opacity=0.6),
)
scene.add(wake_cloud)

# --- NREM scatter---
nrem_pos = manifold[:INDEX, :].astype('float32').to_numpy()
nrem_colors = np.full((INDEX, 4), [0.8, 0.8, 0.8, 0.4], dtype="float32")
nrem_cloud = gfx.Points(
    gfx.Geometry(positions=nrem_pos, colors=nrem_colors),
    gfx.PointsMaterial(size=1, color_mode="vertex", opacity=0.4),
)
scene.add(nrem_cloud)

# --- Trail (follows NREM manifold trajectory) ---
trail = Trail(
    nrem_pos, scene,
    trail_len=25,
    cmap="plasma",
    marker_color="#c10af3",
    marker_size=10,
    line_thickness=2,
    cloud=False,
)

# --- PlaybackController ---
playback = PlaybackController(
    manifold.times(), register_events=renderer
)
overlay = TimeText(viewport=renderer, position="top-left")


def on_playback_update(pb: PlaybackController):
    trail.update(pb.frame_index, pb.frame_position)


playback.add_handler(on_playback_update)
on_playback_update(playback)

# === SYNCHRONIZE WITH CONTROLLERGROUP ===
cg = ControllerGroup(
    plots=[spike_plot, sleep_plot, decoded_hd_plot, state_prob_plot],
    interval=(manifold.times()[0], manifold.times()[-1]),
)

# PlaybackController exposes .controller and .renderer for ControllerGroup
cg.add(playback, controller_id=4)

# === ANIMATION LOOP ===
def anim():
    renderer.render(scene, camera, flush=False)
    overlay.update(playback)
    overlay.render(flush=True)
    canvas.request_draw()


# === SHOW ===
print("\nControls:")
print("  P / Right Click       Play / pause")
print("  Arrow Up / Down       Speed up / down")
print("  Arrow Right / Left    Step forward / backward")
print("  Pan/zoom in pynaviz plots to sync time")

if __name__ == "__main__":
    canvas.request_draw(anim)
    loop.run()
