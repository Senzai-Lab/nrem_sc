"""
3D scatter-plot analysis of the NREM / homecage manifold using fastplotlib.

Left panel  – time-colored (viridis)
Right panel – state-colored (continuous / fragmented / stationary argmax)

classified_hd is at 2 ms bins; manifold is at 50 ms bins.
We align by finding the nearest classified_hd time for each manifold sample.
"""

import numpy as np
import pynapple as nap
import fastplotlib as fpl
import matplotlib.colors as mcolors

from nrem_sc.constants import PROCESSED_DATA_PATH

# === CONFIG ===
unit_id = "116b"
INDEX = 301863  # number of NREM / homecage samples in the combined manifold

STATE_NAMES = ["continuous", "fragmented", "stationary"]
# Set2 palette first three colours (matches classify_states convention)
STATE_COLORS = {
    "continuous":  np.array([0.40, 0.76, 0.65, 1.0], dtype="float32"),   # teal
    "fragmented":  np.array([0.99, 0.55, 0.38, 1.0], dtype="float32"),   # orange
    "stationary":  np.array([0.55, 0.63, 0.80, 1.0], dtype="float32"),   # blue
}

# === LOAD DATA ===
manifold = nap.load_file(PROCESSED_DATA_PATH / unit_id / "combined_umap_prettx.npz")
classified_hd = nap.load_file(PROCESSED_DATA_PATH / unit_id / "state_decoded.npz")

# NREM portion of the manifold (first INDEX rows)
nrem_pos = manifold.values[:INDEX, :].astype("float32")
nrem_times = manifold.times()[:INDEX]

print(f"NREM manifold : {nrem_pos.shape}  (50 ms bins)")
print(f"classified_hd : {classified_hd.shape}  (2 ms bins)")
print(f"manifold time range : [{nrem_times[0]:.1f}, {nrem_times[-1]:.1f}] s")
print(f"classified_hd range : [{classified_hd.times()[0]:.1f}, {classified_hd.times()[-1]:.1f}] s")

# === ALIGN STATE DATA TO MANIFOLD TIMESTAMPS (50 ms ← 2 ms) ===
state_probs = classified_hd[STATE_NAMES]           # TsdFrame (2 ms bins)
state_times = state_probs.times()
state_vals  = state_probs.values                     # (N, 3)

# For each manifold time, find nearest classified_hd index
aligned_idx = np.searchsorted(state_times, nrem_times, side="left")
aligned_idx = np.clip(aligned_idx, 0, len(state_times) - 1)

# Argmax across the three state columns → dominant state per 50 ms bin
aligned_probs = state_vals[aligned_idx]              # (INDEX, 3)
dominant_state = np.argmax(aligned_probs, axis=1)    # 0 / 1 / 2

# Build RGBA colour array based on dominant state
state_colors_rgba = np.zeros((INDEX, 4), dtype="float32")
for i, name in enumerate(STATE_NAMES):
    mask = dominant_state == i
    state_colors_rgba[mask] = STATE_COLORS[name]
    print(f"  {name:>12s}: {mask.sum():>8,d} points ({mask.mean()*100:.1f}%)")

# === VISUALISE ===
fig = fpl.Figure(
    shape=(1, 1),
    size=(1600, 700),
    cameras=[["3d"]],
    # names=[["Time (viridis)", "Dominant state"]],
)

# # --- Left: time-coloured scatter ---
# time_cvals = np.linspace(0, 1, INDEX, dtype="float32")

# fig[0, 0].add_scatter(
#     data=nrem_pos,
#     cmap="viridis",
#     cmap_transform=time_cvals,
#     sizes=2,
# )
# --- Right: state-coloured scatter ---

fig[0, 0].add_scatter(
    data=nrem_pos,
    colors=state_colors_rgba,
    sizes=2,
)


fig.show()

if __name__ == "__main__":
    print("\nViewer ready – drag to rotate, scroll to zoom.")
    fpl.loop.run()
