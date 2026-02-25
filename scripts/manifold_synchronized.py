from nrem_sc.constants import PROCESSED_DATA_PATH
from nrem_sc.legend import add_legend
from nrem_sc.playback import PlaybackController
from nrem_sc.relative_time import apply_relative_time
from nrem_sc.timetext import TimeText
from nrem_sc.trail import Trail

import numpy as np
import pynapple as nap
import pynaviz as viz
import pygfx as gfx
import seaborn as sns

import matplotlib as mpl
from matplotlib.colors import ListedColormap

from rendercanvas.auto import RenderCanvas, loop
from pynaviz.controller_group import ControllerGroup

# === CONFIG ===
unit_id = "116b"

window = nap.IntervalSet(start=[35000], end=[40000])

# === LOAD DATA ===
hd_spikes = nap.load_file(PROCESSED_DATA_PATH / unit_id/ "hd_spikes_filtered.npz").restrict(window)
manifold_post = nap.load_file(PROCESSED_DATA_PATH / unit_id / "manifold_post.npz").restrict(window)
manifold_openfield = nap.load_file(PROCESSED_DATA_PATH / unit_id / "manifold_openfield.npz")
sleep_states = nap.load_file(PROCESSED_DATA_PATH / unit_id / "sleep.npz")
classified_hd = nap.load_file(PROCESSED_DATA_PATH / unit_id / "post_ttx_decoded.npz").restrict(window)
sweep_epochs = nap.load_file(PROCESSED_DATA_PATH / unit_id/ "nrem_sweep_epochs.npz")

# HD angle → RGBA colors using HUSL palette
# Register HUSL as a matplotlib colormap
husl_cmap = ListedColormap(sns.color_palette("husl", 256), name="husl")
mpl.colormaps.register(husl_cmap)
n_colors = 256
husl_palette = sns.color_palette("husl", n_colors)
angle_indices = ((manifold_openfield['angle'] / 360.0) * (n_colors - 1)).astype(int)
rgb_colors = np.array([husl_palette[i] for i in angle_indices])
colors_rgba = np.column_stack((rgb_colors, np.ones(len(rgb_colors)))).astype("float32")

print(f"Loaded {len(hd_spikes)} units")
print(f"Loaded manifold: {len(manifold_post)} points")
print(f"Time range: [{manifold_post.times()[0]:.1f}, {manifold_post.times()[-1]:.1f}] s")
print(f"Sleep states: {len(sleep_states)} intervals")

# === PYNAVIZ PLOTS ===
spike_plot = viz.PlotTsGroup(hd_spikes, index=0)
spike_plot.sort_by(metadata_name='preferred_angle', mode='ascending')
spike_plot.color_by(metadata_name='preferred_angle', cmap_name='husl')

sleep_plot = viz.PlotIntervalSet(sleep_states, index=1)
sleep_plot.color_by(metadata_name='state', cmap_name='Set2')
add_legend(sleep_plot, metadata_name='state', cmap_name='Set2')

decoded_hd_plot = viz.PlotTsd(classified_hd['position'], index=2)

tsdf = classified_hd[['continuous', 'fragmented', 'stationary']]
tsdf.set_info(color_id=[0, 1, 2])
state_prob_plot = viz.PlotTsdFrame(tsdf, index=3)
state_prob_plot.color_by(metadata_name='color_id', cmap_name='Set2')
state_prob_plot.add_interval_sets(sweep_epochs, alpha=0.4)

# # # === RELATIVE TIME FORMATTING ===
# for _plot in [spike_plot, sleep_plot, decoded_hd_plot, state_prob_plot]:
#     apply_relative_time(_plot)

# === MANIFOLD VIEWER (pygfx scene + Trail + PlaybackController) ===
canvas = RenderCanvas(max_fps=60, title="Manifold Viewer")
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()
scene.add(gfx.Background.from_color("#0a0a0a"))

camera = gfx.OrthographicCamera(maintain_aspect=True)
pan_zoom = gfx.PanZoomController(camera, register_events=renderer)

# # --- Wake scatter (openfield manifold, colored by HD angle) ---
# of_xy = manifold_openfield.values[:, :2].astype("float32")
# of_pos = np.column_stack([of_xy, np.zeros(len(of_xy), dtype="float32") - 0.1])
# wake_cloud = gfx.Points(
#     gfx.Geometry(positions=of_pos, colors=colors_rgba),
#     gfx.PointsMaterial(size=2, color_mode="vertex", opacity=0.6),
# )
# scene.add(wake_cloud)

# # --- NREM scatter/trail ---
# trail = Trail(
#     manifold_post.to_numpy().astype("float32"), scene,
#     trail_len=25,
#     cmap="plasma",
#     marker_color="#c10af3",
#     marker_size=10,
#     line_thickness=2,
#     cloud=False,
# )

# # --- PlaybackController ---
# playback = PlaybackController(
#     manifold_post.times(), register_events=renderer
# )
# overlay = TimeText(viewport=renderer, position="top-left")

# def on_playback_update(pb: PlaybackController):
#     trail.update(pb.frame_index, pb.frame_position)

# playback.add_handler(on_playback_update)
# on_playback_update(playback)

# === SYNCHRONIZE WITH CONTROLLERGROUP ===
cg = ControllerGroup(
    plots=[spike_plot, sleep_plot, decoded_hd_plot, state_prob_plot],
    interval=(window['start'].item(), window['end'].item()),
)

# PlaybackController exposes .controller and .renderer for ControllerGroup
# cg.add(playback, controller_id=4)

# === ANIMATION LOOP ===
def anim():
    renderer.render(scene, camera, flush=False)
    # overlay.update(playback)
    # overlay.render(flush=True)
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
