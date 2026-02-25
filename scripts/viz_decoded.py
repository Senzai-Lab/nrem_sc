from nrem_sc.constants import PROCESSED_DATA_PATH
from trajectory_viewer import add_legend
from nrem_sc.utils import circ_bin_average

import pynapple as nap
print(nap.__version__)

import pynaviz as viz
from pynaviz.controller_group import ControllerGroup

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib.colors import ListedColormap

# === CONFIG ===
unit_id = "116b"
window = nap.IntervalSet(start=30000, end=40000)
husl_cmap = ListedColormap(sns.color_palette("husl", 256), name="husl")
mpl.colormaps.register(husl_cmap)

# === LOAD DATA ===
hd_spikes = nap.load_file(PROCESSED_DATA_PATH / unit_id / "hd_spikes_filtered.npz").restrict(window)
sleep_states = nap.load_file(PROCESSED_DATA_PATH / unit_id / "sleep.npz").intersect(window)
classified_hd = nap.TsdFrame(pd.read_csv(PROCESSED_DATA_PATH / unit_id / "post_ttx_decoded_states.csv", index_col=0)).restrict(window)
sweep_epochs = nap.load_file(PROCESSED_DATA_PATH / unit_id / "post_ttx_nrem_epochs.npz").intersect(window)

print(f"Loaded {len(hd_spikes)} units")
print(f"Sleep states: {len(sleep_states)} intervals")
print(f"Classified HD: {classified_hd.shape}")
print(f"Sweep epochs: {len(sweep_epochs)} intervals")

# === CONFIGURE DATA ===
tsdf = classified_hd[['continuous', 'fragmented', 'stationary']]
tsdf.bin_average(bin_size=0.1, time_units='s')
tsdf.set_info(color_id=[0, 1, 2])

# === PYNAVIZ PLOTS ===
spike_plot = viz.PlotTsGroup(hd_spikes, index=0)
spike_plot.sort_by(metadata_name='preferred_angle', mode='ascending')
spike_plot.color_by(metadata_name='preferred_angle', cmap_name='husl')
spike_plot.add_interval_sets(sweep_epochs, alpha=0.7)

sleep_plot = viz.PlotIntervalSet(sleep_states, index=1)
sleep_plot.color_by(metadata_name='state', cmap_name='Set2')
# add_legend(sleep_plot, metadata_name='state', cmap_name='Set2')

decoded_hd_plot = viz.PlotTsd(
    circ_bin_average(classified_hd['position'], bin_size=0.1, time_units='s'),
    index=2
    )
decoded_hd_plot.add_interval_sets(sweep_epochs, alpha=0.7)

state_prob_plot = viz.PlotTsdFrame(tsdf, index=3)
state_prob_plot.color_by(metadata_name='color_id', cmap_name='Set2')
state_prob_plot.add_interval_sets(sweep_epochs, alpha=0.7)

# === SYNCHRONIZE WITH CONTROLLERGROUP ===
cg = ControllerGroup(
    plots=[spike_plot, sleep_plot, decoded_hd_plot, state_prob_plot],
    interval=(window['start'][0], window['end'][0]),
)


if __name__ == "__main__":
    spike_plot.show()
