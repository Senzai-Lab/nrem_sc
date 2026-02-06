from src.constants import PROCESSED_DATA_PATH, INTERIM_DATA_PATH
import numpy as np
import pynapple as nap
import pynaviz as viz
from pynaviz.controller_group import ControllerGroup
from pynaviz import scope

unit_id = '107b'
# Load data
z_rate          = nap.load_file(INTERIM_DATA_PATH / unit_id / "hd_pop_zrate.npz")
burst_epochs    = nap.load_file(INTERIM_DATA_PATH / unit_id / "hd_burst_epochs.npz")
pupil_data      = nap.load_file(PROCESSED_DATA_PATH / unit_id / "pupil_nrem_normalized.npz")
# sleep_states    = nap.load_file(PROCESSED_DATA_PATH / unit_id / "sleep.npz")
hd_spikes       = nap.load_file(PROCESSED_DATA_PATH / unit_id / "hd_spikes_total.npz")
turn_spikes     = nap.load_file(PROCESSED_DATA_PATH / unit_id / "turn_spikes.npz")
# Preprocess
pupil_data = pupil_data['pupil_chord']

# Visualization
scope(globals())