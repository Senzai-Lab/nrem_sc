from src.constants import INTERIM_DATA_PATH
import pynapple as nap
import pynaviz as viz

# === CONFIG ===
unit_id = "116b"

# === LOAD TIME-SHIFTED DATA ===
data_dir = INTERIM_DATA_PATH / unit_id / "tmp"

sleep_shifted = nap.load_file(data_dir / "sleep_shifted.npz")
hd_spikes_shifted = nap.load_file(data_dir / "hd_spikes_shifted.npz")
manifold_shifted = nap.load_file(data_dir / "manifold_shifted.npz")

print(f"Loaded data from: {data_dir}")
print(f"  - sleep: {len(sleep_shifted)} intervals")
print(f"  - units: {len(hd_spikes_shifted)} units")
print(f"  - manifold: {len(manifold_shifted)} samples")
print(f"  - time range: [{manifold_shifted.times()[0]:.1f}, {manifold_shifted.times()[-1]:.1f}] s")

# === VISUALIZE ===
# Use TsdFrameWidget for the manifold and plot as scatter (x vs y)
# manifold_widget = viz.TsdFrameWidget(manifold_shifted, interval_sets={'sleep': sleep_shifted})
# manifold_widget.plot.plot_x_vs_y('x', 'y', color='white', thickness=0, markersize=5)
# manifold_widget.show()

# Optionally, also show spikes in a separate widget synced together
spike_widget = viz.TsGroupWidget(hd_spikes_shifted)
spike_widget.show()
