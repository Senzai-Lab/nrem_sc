from nrem_sc.constants import PROCESSED_DATA_PATH
from nrem_sc.utils import circ_bin_average

import numpy as np
import pynapple as nap

from replay_trajectory_classification import (
    SortedSpikesClassifier,
    Environment,
    RandomWalk,
    Uniform,
    Identity,
    DiagonalDiscrete,
    make_track_graph,
)

unit_id = '116b'
STATE_PROB = 0.9975
STATE_NAMES = ["continuous", "fragmented", "stationary"]

def get_environment(num_nodes: int = 360, place_bin_size: float = 1.0):
    radius = 180 / np.pi
    angle = np.linspace(2 * np.pi, 0, num=num_nodes, endpoint=False)
    node_positions = np.stack((radius * np.cos(angle), radius * np.sin(angle)), axis=1)

    node_ids = np.arange(node_positions.shape[0])
    edges = np.stack((node_ids, np.roll(node_ids, shift=1)), axis=1)

    track_graph = make_track_graph(node_positions, edges)

    n_nodes = len(track_graph.nodes)
    edge_order = np.stack(
        (np.roll(np.arange(n_nodes - 1, -1, -1), 1),
         np.arange(n_nodes - 1, -1, -1)),
        axis=1,
    )

    return Environment(
        place_bin_size=place_bin_size,
        track_graph=track_graph,
        edge_order=edge_order,
        edge_spacing=0,
    )

def fit_classifier(
    hd_spikes,
    hd_angle,
    train_ep: nap.IntervalSet,
    bin_size_ms: int = 1,
    place_bin_size: float = 1.0,
):
    """Fit the classifier on wake training data and return it."""
    spikes = (
        hd_spikes.count(bin_size=bin_size_ms, ep=train_ep, time_units="ms")
        .astype(np.bool_)
    )
    angle = circ_bin_average(tsd=hd_angle, bin_size=bin_size_ms, ep=train_ep, time_units="ms").to_numpy()

    # Build classifier
    environment = get_environment(place_bin_size=place_bin_size)
    continuous_transition_types = [
        [RandomWalk(movement_var=2.0), Uniform(), Identity()],
        [Uniform(),                    Uniform(), Uniform()],
        [RandomWalk(movement_var=2.0), Uniform(), Identity()],
    ]
    classifier = SortedSpikesClassifier(
        environments=environment,
        continuous_transition_types=continuous_transition_types,
        discrete_transition_type=DiagonalDiscrete(STATE_PROB),
    )
    
    # Fit classifier
    classifier.fit(angle, spikes.to_numpy())
    return classifier

if __name__ == "__main__":
    # Load data
    sleep_states    = nap.load_file(PROCESSED_DATA_PATH / unit_id / "sleep.npz")
    hd_spikes       = nap.load_file(PROCESSED_DATA_PATH / unit_id / "hd_spikes_filtered.npz")
    hd_angle        = nap.load_file(PROCESSED_DATA_PATH / unit_id / "angle_openfield.npz")
    sessions        = nap.load_file(PROCESSED_DATA_PATH / unit_id / "sessions_labeled.npz")


    classifier = fit_classifier(hd_spikes, hd_angle, hd_angle.time_support)
 
    t_window = 1000
    
    # --- Pre-TTX decoding ---
    SAVE_DIR = PROCESSED_DATA_PATH / unit_id / "pre_ttx"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    session = sessions[sessions["label"] == "homecage"][0]
    # ---
    start, end = session["start"].item(), session["end"].item()
    print(f"Homecage session: {start} - {end}")
    starts = np.arange(start, end, t_window)
    for t_a in starts:
        t_b = t_a + t_window
        t_b = np.min([t_b, end])
        print(f"Decoding interval: {t_a} - {t_b}")
        print(f"Duration: {t_b - t_a:.2f} s")
        
        # Decode states in this interval
        epoch = nap.IntervalSet(start=[t_a], end=[t_b])
        bin_size_ms = 1

        spike_counts = (
            hd_spikes.restrict(epoch)
            .count(bin_size=bin_size_ms, time_units="ms")
            .astype(np.bool_)
        )

        decoded = classifier.predict(
            spike_counts.values,
            time=spike_counts.times(),
            state_names=STATE_NAMES,
        )

        prob_state = decoded.acausal_posterior.sum(dim="position").to_numpy()
        posterior  = decoded.acausal_posterior.sum(dim="state")
        pos_max    = posterior.idxmax(dim="position").to_numpy()

        combined = np.column_stack([prob_state, pos_max])

        result_df = nap.TsdFrame(t=decoded["time"].to_numpy(), d=combined, columns=STATE_NAMES + ["position"],)
        result_df.save(SAVE_DIR / f"decoded{int(t_a)}_{int(t_b)}.npz")        
        