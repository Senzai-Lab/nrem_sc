import argparse
from pathlib import Path

import numpy as np
import pynapple as nap

from nrem_sc.constants import PROCESSED_DATA_PATH, INTERIM_DATA_PATH
from nrem_sc.utils import circ_bin_average

from replay_trajectory_classification import (
    SortedSpikesClassifier,
    Environment,
    RandomWalk,
    Uniform,
    Identity,
    DiagonalDiscrete,
    make_track_graph,
)



def get_environment(num_nodes: int = 360, place_bin_size: float = 1.0):
    radius = 180 / np.pi
    angle = np.linspace(2 * np.pi, 0, num=num_nodes, endpoint=False)
    node_positions = np.stack(
        (radius * np.cos(angle), radius * np.sin(angle)), axis=1
    )

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


def build_classifier(environment: Environment) -> SortedSpikesClassifier:
    continuous_transition_types = [
        [RandomWalk(movement_var=2.0), Uniform(), Identity()],
        [Uniform(),                    Uniform(), Uniform()],
        [RandomWalk(movement_var=2.0), Uniform(), Identity()],
    ]
    return SortedSpikesClassifier(
        environments=environment,
        continuous_transition_types=continuous_transition_types,
        discrete_transition_type=DiagonalDiscrete(0.9),
    )


# ── data loading ─────────────────────────────────────────────────────────────

def load_data(unit_id: str):
    base = PROCESSED_DATA_PATH / unit_id
    hd_spikes = nap.load_file(base / "hd_spikes_filtered.npz")
    hd_angle  = nap.load_file(base / "angle_openfield.npz")
    return hd_spikes, hd_angle


# ── training ─────────────────────────────────────────────────────────────────

def fit_classifier(
    hd_spikes,
    hd_angle,
    train_ep: nap.IntervalSet,
    bin_size_ms: int = 2,
    place_bin_size: float = 1.0,
):
    """Fit the classifier on wake training data and return it."""
    spikes = (
        hd_spikes.count(bin_size=bin_size_ms, ep=train_ep, time_units="ms")
        .astype(np.bool_)
    )
    angle = circ_bin_average(tsd=hd_angle, bin_size=bin_size_ms, ep=train_ep, time_units="ms").to_numpy()

    environment = get_environment(place_bin_size=place_bin_size)
    classifier = build_classifier(environment)
    classifier.fit(angle, spikes.to_numpy())
    return classifier


# ── decoding ─────────────────────────────────────────────────────────────────

STATE_NAMES = ["continuous", "fragmented", "stationary"]


def decode_epoch(
    classifier: SortedSpikesClassifier,
    hd_spikes,
    epoch: nap.IntervalSet,
    bin_size_ms: int = 2,
    sigma: float = 1.0,
):
    """Decode a single epoch and return a summary TsdFrame."""
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

    result = nap.TsdFrame(
        t=decoded["time"].to_numpy(),
        d=combined,
        columns=STATE_NAMES + ["position"],
    )
    return result, decoded


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Replay trajectory decoding")
    parser.add_argument("--unit", default="116b", help="Unit ID (default: 116b)")
    parser.add_argument("--bin-size", type=int, default=2, help="Bin size in ms")
    args = parser.parse_args()

    unit_id = args.unit
    bin_size_ms = args.bin_size

    # ── load ──
    print(f"Loading data for unit {unit_id} ...")
    hd_spikes, hd_angle = load_data(unit_id)

    # ── fit on wake epoch ──
    train_ep = nap.IntervalSet([16800], [18000])
    print("Fitting classifier on wake training epoch ...")
    classifier = fit_classifier(
        hd_spikes, hd_angle, train_ep,
        bin_size_ms=bin_size_ms,
    )

    # ── decode over epochs ──
    # Add / modify epochs here as needed:
    decode_epochs = {
        "sleep_34000_36000": nap.IntervalSet(start=[34000], end=[36000]),
        # "sleep_36000_38000": nap.IntervalSet(start=[36000], end=[38000]),
    }

    out_dir = INTERIM_DATA_PATH / unit_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, epoch in decode_epochs.items():
        print(f"\nDecoding epoch '{label}' ...")
        result_tdf, decoded = decode_epoch(
            classifier, hd_spikes, epoch, bin_size_ms=bin_size_ms,
        )

        # save TsdFrame
        tdf_path = out_dir / f"decoded_{label}.npz"
        result_tdf.save(tdf_path)
        print(f"  Saved TsdFrame  -> {tdf_path}")

        # save full posterior (xarray NetCDF)
        nc_path = out_dir / f"posterior_{label}.nc"
        decoded.acausal_posterior.to_netcdf(nc_path)
        print(f"  Saved posterior  -> {nc_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
