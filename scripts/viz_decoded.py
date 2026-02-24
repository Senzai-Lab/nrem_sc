from nrem_sc.constants import PROCESSED_DATA_PATH
from nrem_sc.legend import add_legend

import pynapple as nap
import pynaviz as viz
from pynaviz.controller_group import ControllerGroup

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib.colors import ListedColormap

# === CONFIG ===
unit_id = "116b"
window = nap.IntervalSet(start=[30000], end=[40000])
husl_cmap = ListedColormap(sns.color_palette("husl", 256), name="husl")
mpl.colormaps.register(husl_cmap)

# === LOAD DATA ===
hd_spikes = nap.load_file(PROCESSED_DATA_PATH / unit_id / "hd_spikes_filtered.npz").restrict(window)
sleep_states = nap.load_file(PROCESSED_DATA_PATH / unit_id / "sleep.npz").intersect(window)
classified_hd = nap.load_file(PROCESSED_DATA_PATH / unit_id / "post_ttx_decoded.npz").restrict(window)
sweep_epochs = nap.load_file(PROCESSED_DATA_PATH / unit_id / "nrem_sweep_epochs.npz").intersect(window)

print(f"Loaded {len(hd_spikes)} units")
print(f"Sleep states: {len(sleep_states)} intervals")

# === CONFIGURE DATA ===
tsdf = classified_hd[['continuous', 'fragmented', 'stationary']]
tsdf.set_info(color_id=[0, 1, 2])

# --- Work around pynaviz / pandas StringDtype incompatibility ----------
# Pynapple's .metadata property returns a DataFrame where string columns
# use pandas StringDtype.  pynaviz's MetadataMappingThread calls
# np.issubdtype(values.dtype, np.str_) which *raises* TypeError for
# StringDtype instead of returning False, crashing the background mapping
# thread.  Patch as_dataframe on the affected _Metadata instances so
# the returned DataFrame uses plain object dtype for string columns.
# def _patch_metadata_strings(pynapple_obj):
#     """Replace as_dataframe on the _Metadata so strings stay object dtype."""
#     _orig = pynapple_obj._metadata.as_dataframe

#     def _as_dataframe_fixed():
#         df = _orig()
#         for col in df.columns:
#             if pd.api.types.is_string_dtype(df[col]):
#                 df[col] = df[col].astype(object)
#         return df

#     pynapple_obj._metadata.as_dataframe = _as_dataframe_fixed

# _patch_metadata_strings(sleep_states)

# === PYNAVIZ PLOTS ===
spike_plot = viz.PlotTsGroup(hd_spikes, index=0)
spike_plot.sort_by(metadata_name='preferred_angle', mode='ascending')
spike_plot.color_by(metadata_name='preferred_angle', cmap_name='husl')

sleep_plot = viz.PlotIntervalSet(sleep_states, index=1)
sleep_plot.color_by(metadata_name='state', cmap_name='Set2')
add_legend(sleep_plot, metadata_name='state', cmap_name='Set2')

decoded_hd_plot = viz.PlotTsd(classified_hd['position'], index=2)
decoded_hd_plot.add_interval_sets(sweep_epochs, alpha=0.2)

state_prob_plot = viz.PlotTsdFrame(tsdf, index=3)
state_prob_plot.color_by(metadata_name='color_id', cmap_name='Set2')
state_prob_plot.add_interval_sets(sweep_epochs, alpha=0.2)

# === SYNCHRONIZE WITH CONTROLLERGROUP ===
cg = ControllerGroup(
    plots=[spike_plot, sleep_plot, decoded_hd_plot, state_prob_plot],
    interval=(window['start'][0], window['end'][0]),
)


if __name__ == "__main__":
    spike_plot.show()
