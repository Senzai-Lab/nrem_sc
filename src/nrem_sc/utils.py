from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import xarray as xr
import pynapple as nap
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.axes
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit


def group_by_ids(values: np.ndarray, ids: np.ndarray, select_ids=None) -> dict:
    """
    Groups values by their corresponding ids.
    
    Parameters
    ----------
    values : np.ndarray
        1D array of values.
    ids : np.ndarray
        1D array of ids corresponding to each value in values.
    select_ids : array-like, optional
        Specific ids to group by. If None, all unique ids in ids are used.
    """
    assert values.ndim == 1, "values must be a 1D array"
    assert ids.ndim == 1, "ids must be a 1D array"
    assert values.shape[0] == ids.shape[0], "values and ids must have the same length"

    if select_ids is None:
        select_ids = np.unique(ids)
    return {uid: values[ids == uid] for uid in select_ids}


def hd_tuning(
    spikes: nap.TsGroup,
    angles: nap.Tsd,
    n_bins: int = 61,
    sigma: float = 3
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Compute head direction tuning curves and preferred angles.
    
    Parameters
    ----------
    spikes : pynapple.TsGroup
        Spike times for each unit.
    angles : pynapple.Tsd
        Head direction angles (in degrees or radians).
    n_bins : int, optional
        Number of bins for the tuning curve (default: 61).
    sigma : float, optional
        Gaussian smoothing sigma (default: 3).
    
    Returns
    -------
    tcs : xarray.DataArray
        Tuning curves for each unit.
    pref_ang : xarray.DataArray
        Preferred angle for each unit.
    """
    angles_rad = np.deg2rad(angles) if angles.max() > 2 * np.pi else angles
    spikes_r = spikes.restrict(angles_rad.time_support)
    tcs = nap.compute_tuning_curves(
        data=spikes_r, features=angles_rad, bins=n_bins,
        epochs=angles_rad.time_support, range=(0.0, 2*np.pi),
        feature_names=['head_direction']
    )
    tcs.values = gaussian_filter1d(tcs.values, sigma=sigma, axis=1, mode="wrap")
    pref_ang = tcs.idxmax(dim="head_direction")
    return tcs, pref_ang


def circ_colors(values: np.ndarray, cmap: str = 'hsv') -> np.ndarray:
    """
    Get colors for circular values (0 to 2π).
    
    Parameters
    ----------
    values : array-like
        Circular values in the range [0, 2π].
    cmap : str, optional
        Colormap name (default: 'hsv').
    
    Returns
    -------
    colors : ndarray
        RGBA colors for each value.
    """
    norm = plt.Normalize(0, 2*np.pi)
    return plt.get_cmap(cmap)(norm(values))


def plot_tuning_grid(
    tcs: xr.DataArray,
    pref_ang: xr.DataArray,
    colors: xr.DataArray,
    nrows: int = 9,
    ncols: int = 9,
    figsize: Tuple[float, float] = (12, 9)
) -> matplotlib.figure.Figure:
    """
    Plot sorted tuning curves in a grid.
    
    Parameters
    ----------
    tcs : xarray.DataArray
        Tuning curves for each unit.
    pref_ang : xarray.DataArray
        Preferred angle for each unit.
    colors : xarray.DataArray
        Colors for each unit.
    nrows : int, optional
        Number of rows in the grid (default: 9).
    ncols : int, optional
        Number of columns in the grid (default: 9).
    figsize : tuple, optional
        Figure size (default: (12, 9)).
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    sorted_tcs = tcs.sortby(pref_ang)
    fig = plt.figure(figsize=figsize)
    for i, n in enumerate(sorted_tcs.coords["unit"]):
        plt.subplot(nrows, ncols, i + 1, projection='polar')
        plt.plot(sorted_tcs.coords["head_direction"], sorted_tcs.sel(unit=n).values,
                 color=colors.sel(unit=n).values)
        plt.xticks([])
    plt.show()
    return fig


def plot_intervals(
    intervals: nap.IntervalSet,
    column: str,
    min_dur: float = 2,
    palette: str = 'deep',
    figsize: Tuple[float, float] = (14, 2.5),
    ax: Optional[matplotlib.axes.Axes] = None
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Plot IntervalSet as horizontal bars.
    
    Parameters
    ----------
    intervals : pynapple.IntervalSet
        Interval set with a metadata column.
    column : str
        Name of the column to use for state labels.
    min_dur : float, optional
        Minimum duration to include (default: 2).
    palette : str, optional
        Color palette for states (default: 'deep').
    figsize : tuple, optional
        Figure size (default: (14, 2.5)).
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates a new figure.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    intervals = intervals.drop_short_intervals(min_dur)
    states = np.unique(intervals[column])
    n_states = len(states)
    colors = dict(zip(states, sns.color_palette(palette, n_states)))
    
    fig_height = figsize[1]
    bar_height = min(0.8, fig_height / (n_states + 1))  # Leave some padding
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for i, state in enumerate(states):
        epochs = intervals[intervals[column] == state]
        xranges = np.column_stack([epochs.start, epochs.end - epochs.start])
        ax.broken_barh(xranges, (i - bar_height/2, bar_height), facecolors=colors[state], edgecolor='none')
    
    ax.set_yticks(range(n_states), labels=states)
    ax.set_ylim(-0.5, n_states - 0.5)
    ax.set_xlabel('Time (s)')
    ax.set_xlim(intervals.start.min(), intervals.end.max())
    ax.spines[['top', 'right']].set_visible(False)
    
    if ax is None:
        plt.tight_layout()
        
    return fig, ax


def find_outliers(
    manifold: nap.TsdFrame,
    r_min: float = 2,
    r_max: float = 6
) -> Tuple[nap.IntervalSet, np.ndarray, np.ndarray]:
    """
    Find manifold outliers by radial distance from center.
    
    Parameters
    ----------
    manifold : pynapple.TsdFrame
        Manifold data with 'x' and 'y' columns.
    r_min : float, optional
        Minimum radial distance threshold (default: 2).
    r_max : float, optional
        Maximum radial distance threshold (default: 6).
    
    Returns
    -------
    outlier_intervals : pynapple.IntervalSet
        Intervals containing outlier points.
    radii : np.ndarray
        Radial distances from center for each point.
    center : np.ndarray
        Center coordinates of the manifold.
    """
    center = manifold.mean(axis=0)
    radii = np.sqrt((manifold['x'] - center[0]).to_numpy()**2 + 
                    (manifold['y'] - center[1]).to_numpy()**2)
    idx = np.where((radii < r_min) | (radii > r_max))[0]
    jumps = np.where(np.diff(idx) > 1)[0] + 1
    periods = np.split(idx, jumps)
    start = [manifold.t[p[0]].item() for p in periods if len(p) > 0]
    end = [manifold.t[p[-1]].item() for p in periods if len(p) > 0]
    return nap.IntervalSet(start=start, end=end, time_units='s'), radii, center


# =============================================================================
# Von Mises Tuning Curve Fitting
# =============================================================================

def von_mises(x: np.ndarray, kappa: float, loc: float, amp: float, offset: float) -> np.ndarray:
    """
    Von Mises distribution function for tuning curve fitting.
    
    Parameters
    ----------
    x : np.ndarray
        Angular positions (in radians).
    kappa : float
        Concentration parameter (higher = sharper tuning).
    loc : float
        Location of the peak (preferred direction).
    amp : float
        Amplitude of the tuning curve.
    offset : float
        Baseline offset.
    
    Returns
    -------
    y : np.ndarray
        Von Mises function values.
    """
    return amp * np.exp(kappa * np.cos(x - loc)) + offset


def fit_von_mises(
    x_data: np.ndarray,
    y_data: np.ndarray,
    p0: Optional[list] = None,
    bounds: Optional[Tuple] = None,
    maxfev: int = 10000
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Fit a von Mises distribution to tuning curve data.
    
    Parameters
    ----------
    x_data : np.ndarray
        Angular positions (in radians).
    y_data : np.ndarray
        Firing rate or response values.
    p0 : list, optional
        Initial parameter guesses [kappa, loc, amp, offset].
        If None, will be estimated from data.
    bounds : tuple, optional
        Parameter bounds ((lower,), (upper,)).
        Default: ([0, -inf, 0, 0], [700, inf, inf, inf]).
    maxfev : int, optional
        Maximum function evaluations (default: 10000).
    
    Returns
    -------
    popt : np.ndarray or None
        Fitted parameters [kappa, loc, amp, offset], or None if fit failed.
    pcov : np.ndarray or None
        Covariance matrix, or None if fit failed.
    """
    if not np.isfinite(y_data).all():
        return None, None
    
    if p0 is None:
        p0 = [
            1.0,
            x_data[np.argmax(y_data)],
            np.max(y_data) - np.min(y_data) + 1e-6,
            np.min(y_data)
        ]
    
    if bounds is None:
        bounds = ([0, -np.inf, 0, 0], [700, np.inf, np.inf, np.inf])
    
    try:
        popt, pcov = curve_fit(von_mises, x_data, y_data, p0=p0, bounds=bounds, maxfev=maxfev)
        return popt, pcov
    except (RuntimeError, ValueError):
        return None, None


def fit_all_tuning_curves(
    tuning_curves: xr.DataArray,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Fit von Mises distribution to all tuning curves.
    
    Parameters
    ----------
    tuning_curves : xr.DataArray
        Tuning curves with dimensions (unit, head_direction).
    verbose : bool, optional
        Print progress messages (default: False).
    
    Returns
    -------
    results : pd.DataFrame
        DataFrame with columns [kappa, loc, amp, offset] indexed by unit.
    """
    results = {}
    x_data = tuning_curves.coords["head_direction"].values
    
    for unit_id in tuning_curves.coords["unit"].values:
        y_data = tuning_curves.sel(unit=unit_id).values
        popt, _ = fit_von_mises(x_data, y_data)
        
        if popt is not None:
            results[unit_id] = {
                'kappa': popt[0],
                'loc': popt[1],
                'amp': popt[2],
                'offset': popt[3]
            }
        else:
            if verbose:
                print(f"Fit failed for unit {unit_id}")
            results[unit_id] = {
                'kappa': np.nan,
                'loc': np.nan,
                'amp': np.nan,
                'offset': np.nan
            }
    
    return pd.DataFrame.from_dict(results, orient='index')


def plot_von_mises_fit(
    x_data: np.ndarray,
    y_data: np.ndarray,
    popt: np.ndarray,
    ax: Optional[matplotlib.axes.Axes] = None,
    title: Optional[str] = None,
    data_color: str = 'C0',
    fit_color: str = 'red',
    data_label: str = 'Data',
    show_legend: bool = True
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Plot tuning curve data with von Mises fit on a polar axis.
    
    Parameters
    ----------
    x_data : np.ndarray
        Angular positions (in radians).
    y_data : np.ndarray
        Firing rate or response values.
    popt : np.ndarray
        Fitted parameters [kappa, loc, amp, offset].
    ax : matplotlib.axes.Axes, optional
        Polar axes to plot on. If None, creates new figure.
    title : str, optional
        Plot title.
    data_color : str, optional
        Color for data line (default: 'C0').
    fit_color : str, optional
        Color for fit line (default: 'red').
    data_label : str, optional
        Label for data (default: 'Data').
    show_legend : bool, optional
        Whether to show legend (default: True).
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The polar axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(5, 5))
    else:
        fig = ax.get_figure()
    
    kappa = popt[0]
    ax.plot(x_data, y_data, label=data_label, linewidth=2, color=data_color)
    ax.plot(x_data, von_mises(x_data, *popt), 
            label=f'Fit (κ={kappa:.2f})', linestyle='--', color=fit_color)
    
    if title:
        ax.set_title(title)
    if show_legend:
        ax.legend(loc='lower left', bbox_to_anchor=(-0.2, -0.2))
    ax.grid(True)
    
    return fig, ax


def plot_all_von_mises_fits(
    tuning_curves: xr.DataArray,
    fit_results: pd.DataFrame,
    save_dir: Optional[str] = None,
    figsize: Tuple[float, float] = (5, 5)
) -> Dict[Any, Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]]:
    """
    Plot von Mises fits for all units.
    
    Parameters
    ----------
    tuning_curves : xr.DataArray
        Tuning curves with dimensions (unit, head_direction).
    fit_results : pd.DataFrame
        DataFrame with fit parameters (from fit_all_tuning_curves).
    save_dir : str or Path, optional
        Directory to save plots. If None, plots are not saved.
    figsize : tuple, optional
        Figure size (default: (5, 5)).
    
    Returns
    -------
    figures : dict
        Dictionary mapping unit_id to (fig, ax) tuples.
    """
    from pathlib import Path
    
    x_data = tuning_curves.coords["head_direction"].values
    figures = {}
    
    for unit_id in tuning_curves.coords["unit"].values:
        if pd.isna(fit_results.loc[unit_id, 'kappa']):
            continue
        
        y_data = tuning_curves.sel(unit=unit_id).values
        popt = fit_results.loc[unit_id, ['kappa', 'loc', 'amp', 'offset']].values
        
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=figsize)
        plot_von_mises_fit(x_data, y_data, popt, ax=ax, title=f"Unit {unit_id}")
        
        figures[unit_id] = (fig, ax)
        
        if save_dir is not None:
            save_path = Path(save_dir) / f"unit_{unit_id}_tuning_fit.png"
            fig.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
    
    return figures


def circ_bin_average(tsd, is_degrees=True, **bin_kwargs):
    """
    Computes the binned average of circular data using Cartesian coordinates.
    """
    # 1. Convert to radians if necessary
    angles = np.deg2rad(tsd.values) if is_degrees else tsd.values
    
    # 2. Convert to Cartesian coordinates
    x = np.cos(angles)
    y = np.sin(angles)
    
    # Create temporary Tsd objects for x and y
    x_tsd = nap.Tsd(t=tsd.times(), d=x, time_support=tsd.time_support)
    y_tsd = nap.Tsd(t=tsd.times(), d=y, time_support=tsd.time_support)
    
    # 3. Bin average the Cartesian components
    x_binned = x_tsd.bin_average(**bin_kwargs)
    y_binned = y_tsd.bin_average(**bin_kwargs)
    
    # 4. Compute mean angle using arctan2
    mean_angles = np.arctan2(y_binned.values, x_binned.values)
    
    # 5. Convert back to degrees and wrap to [0, 360)
    if is_degrees:
        mean_angles = np.rad2deg(mean_angles) % 360
    else:
        mean_angles = mean_angles % (2 * np.pi)
        
    return nap.Tsd(t=x_binned.times(), d=mean_angles, time_support=x_binned.time_support)
