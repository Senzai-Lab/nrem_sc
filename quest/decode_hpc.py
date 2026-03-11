import sys
from pathlib import Path

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

STATE_PROB = 0.99
STATE_NAMES = ["continuous", "fragmented", "stationary"]
BIN_SIZE_MS = 1
DECODING_WINDOW = 500
UNIT_IDS = ['83b', '85b', '116b', '119b']
TOTAL_TASKS = 4

def circ_bin_average(tsd, is_degrees=True, **bin_kwargs):
    """
    Computes the binned average of circular data using Cartesian coordinates.
    """
    angles = np.deg2rad(tsd.values) if is_degrees else tsd.values
    x = np.cos(angles)
    y = np.sin(angles)

    x_tsd = nap.Tsd(t=tsd.times(), d=x, time_support=tsd.time_support)
    y_tsd = nap.Tsd(t=tsd.times(), d=y, time_support=tsd.time_support)
    
    x_binned = x_tsd.bin_average(**bin_kwargs)
    y_binned = y_tsd.bin_average(**bin_kwargs)
    
    mean_angles = np.arctan2(y_binned.values, x_binned.values)
    
    if is_degrees:
        mean_angles = np.rad2deg(mean_angles) % 360
    else:
        mean_angles = mean_angles % (2 * np.pi)
        
    return nap.Tsd(t=x_binned.times(), d=mean_angles)

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


def analyze(data_path: str, save_path:str, task_id:int = None):
    print(f"Data path: {data_path}")
    print(f"Save path: {save_path}")
    print(f"Task ID:   {task_id}")
    data_path = Path(data_path)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Load data
    hd_spikes       = nap.load_file(data_path / "hd_spikes_filtered.npz")
    hd_angle        = nap.load_file(data_path / "angle_openfield.npz")
    sessions        = nap.load_file(data_path / "sessions_labeled.npz")

    # Fit classifier on openfield pre-ttx recording
    classifier = fit_classifier(hd_spikes, hd_angle, hd_angle.time_support)

    for i, condition in enumerate(["pre_ttx", "post_ttx"]):
        print(f"Decoding {condition}...")
        session = sessions[sessions["label"] == 'homecage'][i] # 0->pre_ttx, 1->post_ttx
        start, end = session["start"].item(), session["end"].item()
        print(f"Homecage session: {start} - {end}")

        session_epoch = nap.IntervalSet(start=start, end=end)
        session_spike_counts = (
            hd_spikes.restrict(session_epoch)
            .count(bin_size=BIN_SIZE_MS, time_units="ms")
            .astype(np.bool_)
        )

        starts = np.arange(start, end, DECODING_WINDOW)
        for current_task, t_a in enumerate(starts):
            if (current_task % TOTAL_TASKS) != task_id:
                print("Not my task. Skipping...")
                continue

            t_b = min(t_a + DECODING_WINDOW, end)
            
            out_path = save_path / f"{condition}_{int(t_a)}_{int(t_b)}"
            out_path.mkdir(parents=True, exist_ok=True)
            print(f"Decoding interval: {t_a} - {t_b} ({t_b - t_a:.2f} s)")
            
            if len(list(out_path.glob("*.npz"))) == 2:
                print("Skipping...")
                continue            
            
            # Decode states in this interval
            epoch = nap.IntervalSet(start=t_a, end=t_b)
            spike_counts = session_spike_counts.restrict(epoch)

            decoded = classifier.predict(
                spike_counts.values,
                time=spike_counts.times(),
                state_names=STATE_NAMES,
            )
            # to_netcsdf gives int32 overflow errors for some reason
            # decoded.acausal_posterior.to_netcdf(out_file)

            # Save as Pynapple objects
            states = nap.TsdFrame(t=decoded['time'].to_numpy(),
                                  d=decoded.acausal_posterior.sum(dim='position'), # (N, 3) array
                                  columns=decoded.state)

            position = nap.TsdFrame(t=decoded['time'].to_numpy(),
                                    d=decoded.acausal_posterior.sum(dim='state').to_numpy(),
                                    metadata={'position_bins': decoded.position})

            for data, fname in zip([states, position], ['states', 'position']):
                out_file = out_path / f'{fname}.npz'
                data.save(out_file)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python decode_hpc.py input_path output_path task_id")
        sys.exit(1)

    analyze(sys.argv[1], sys.argv[2], int(sys.argv[3]))
