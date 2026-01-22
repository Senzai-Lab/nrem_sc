import pynaviz as viz
import pynapple as nap
import numpy as np
from pynaviz import scope

from src.constants import RAW_DATA_PATH, PROCESSED_DATA_PATH

unit_id = '116b'

manifold = nap.load_file(PROCESSED_DATA_PATH / unit_id / "manifold_wake_pre_ttx.npz")
hd_spikes = nap.load_file(PROCESSED_DATA_PATH / unit_id / "hd_spikes_wake_pre_ttx.npz")
pulses = np.load(PROCESSED_DATA_PATH / unit_id / "pulses.npy")
outliers = nap.load_file(PROCESSED_DATA_PATH / unit_id / "manifold_outliers_wake_pre_ttx.npz")

if __name__ == "__main__":    
    vid = viz.VideoHandler(RAW_DATA_PATH / unit_id / "Basler_acA1300-200um__23157472__20230607_205412188.avi",
                        time=pulses)

    scope({'units': hd_spikes, 'manifold': manifold, 'video': vid, 'outlier_epochs': outliers})