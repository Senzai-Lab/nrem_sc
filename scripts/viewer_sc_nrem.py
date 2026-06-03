from nrem_sc.constants import PROCESSED_DATA_PATH, INTERIM_DATA_PATH
from viewer import EventBars, TimeSeries, Spikes, RasterView, TraceView, show

import numpy as np
import pynapple as nap

unit_id = '107b'

if __name__ == "__main__":
    sleep_states    = nap.load_file(PROCESSED_DATA_PATH / unit_id / "sleep.npz")
    sc_pop = nap.load_file(PROCESSED_DATA_PATH / unit_id / 'sc_summed.npz')

    sc = nap.load_file(PROCESSED_DATA_PATH / unit_id / 'turn_spikes.npz')
    hd = nap.load_file(PROCESSED_DATA_PATH / unit_id / 'hd_spikes.npz')
    
    turn_spike_clusters = np.load(PROCESSED_DATA_PATH / unit_id / 'turn_spike_clusters.npy', mmap_mode='r')
    turn_spike_times = np.load(PROCESSED_DATA_PATH / unit_id / 'turn_spike_times.npy', mmap_mode='r')

    hd_spike_clusters = np.load(PROCESSED_DATA_PATH / unit_id / 'hd_spike_clusters.npy', mmap_mode='r')
    hd_spike_times = np.load(PROCESSED_DATA_PATH / unit_id / 'hd_spike_times.npy', mmap_mode='r')

    events = EventBars(
        starts=sleep_states.start,
        ends=sleep_states.end,
        labels=sleep_states.state
    )

    trace = TimeSeries(
        name='summed',
        values=sc_pop.values,
        ts=sc_pop.t,
        fs=sc_pop.rate,
        chunk_samples=1000
    )

    sc_spikes = Spikes(
        name='Turn units',
        ts=turn_spike_times,
        spike_units=turn_spike_clusters,
    )

    hd_spikes = Spikes(
        name='HD units',
        ts=hd_spike_times,
        spike_units=hd_spike_clusters,
    )

    show(
        streams=[
            (trace, TraceView()),
            (sc_spikes, RasterView(
                metadata=dict(sc.metadata), unit_ids=sc.index,
                cmap='cmasher:iceburn', sort_by='cw_modulation', color_by='cw_modulation'
                )),
            (hd_spikes, RasterView(
                metadata=dict(hd.metadata), unit_ids=hd.index,
                sort_by='pref_ang', color_by='pref_ang')),
        ],
        event_bars=events,
        max_workers=2,
        )
