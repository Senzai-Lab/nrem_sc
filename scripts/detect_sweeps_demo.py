
import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
from scipy.ndimage import uniform_filter1d

def detect_sweeps(angle_tsd, window_duration=0.5, slope_threshold=2.0, r2_threshold=0.8, sampling_rate=None):
    """
    Detects coherent sweeps in angular time series data.

    Parameters
    ----------
    angle_tsd : nap.Tsd or np.ndarray
        Time series of angles in radians. If np.ndarray, sampling_rate must be provided or
        inferred from generic assumption if not critical (but it is for slope).
    window_duration : float
        Duration of the sliding window in seconds.
    slope_threshold : float
        Minimum angular velocity (rad/s) magnitude to be considered a sweep.
    r2_threshold : float
        Minimum R-squared value of the linear fit to be considered coherent.
    sampling_rate : float, optional
        Sampling rate in Hz. Required if angle_tsd is a numpy array.

    Returns
    -------
    nap.IntervalSet
        The time intervals where sweeps were detected.
    """
    
    # 1. Unwrap the angles
    if hasattr(angle_tsd, 'values') and hasattr(angle_tsd, 'index'):
        # It's a pynapple Tsd
        angles = angle_tsd.values
        times = angle_tsd.index
        dt = np.median(np.diff(times))
    else:
        # Assume numpy array
        angles = np.asarray(angle_tsd)
        if sampling_rate is None:
             raise ValueError("sampling_rate must be provided if input is numpy array")
        dt = 1.0 / sampling_rate
        times = np.arange(len(angles)) * dt

    unwrapped = np.unwrap(angles)
    window_size = int(window_duration / dt)
    
    if window_size < 3:
        raise ValueError("Window duration is too short for the sampling rate.")

    # We will compute R^2 and Slope in a rolling window.
    # An efficient way to do rolling linear regression for just slope and r2:
    
    # y = unwrapped
    # x = times (or just indices for speed)
    
    y = unwrapped
    
    # Using stride_tricks for rolling window view could be memory intensive if data is huge, 
    # but efficient for vectorized operations.
    from numpy.lib.stride_tricks import sliding_window_view
    
    # Create windows
    # Shape: (N - window_size + 1, window_size)
    y_windows = sliding_window_view(y, window_size)
    
    # X variable is just 0, dt, 2dt ... (relative time in window)
    x_local = np.arange(window_size) * dt
    
    # Vectorized linear regression:
    # y = mx + c
    # m = (mean(xy) - mean(x)mean(y)) / (mean(x^2) - mean(x)^2)
    # But mean(x) and mean(x^2) are constant for all windows.
    
    mx = np.mean(x_local)
    var_x = np.var(x_local)
    
    # Compute means of y in each window
    my = np.mean(y_windows, axis=1)
    
    # Compute covariance
    # mean(xy)
    m_xy = np.mean(y_windows * x_local, axis=1)
    
    covariance = m_xy - mx * my
    slope = covariance / var_x
    
    # R^2 = (covariance / (std_x * std_y))^2
    var_y = np.var(y_windows, axis=1)
    
    # Avoid division by zero
    var_y[var_y == 0] = 1e-9
    
    correlation = covariance / (np.sqrt(var_x) * np.sqrt(var_y))
    r2 = correlation ** 2
    
    # Filter
    # 1. High linearity (R^2 > threshold)
    # 2. High velocity (abs(slope) > threshold)
    
    is_sweep = (r2 > r2_threshold) & (np.abs(slope) > slope_threshold)
    
    # The 'is_sweep' array corresponds to the start of the window.
    # We need to map this back to time intervals.
    
    # Construct a boolean array matching the original time series length
    # If a window starting at i is a sweep, we mark indices i to i+window_size as potential sweep candidates.
    # However, simply taking the boolean vector of windows might be easier to convert to intervals first.
    
    # Let's find contiguous potential centers
    sweep_centers_indices = np.where(is_sweep)[0]
    
    if len(sweep_centers_indices) == 0:
        return nap.IntervalSet(start=[], end=[])
    
    # Map valid windows to actual time coverages.
    # A window at index i covers times[i] to times[i+window_size-1]
    
    # We can create an IntervalSet for each valid window and merge them.
    starts = times[sweep_centers_indices]
    ends = times[sweep_centers_indices + window_size - 1]
    
    detected = nap.IntervalSet(starts, ends)
    
    # Merge overlapping intervals
    merged = detected.merge_close_intervals(dt) # Merge if they are adjacent
    
    return merged

def generate_synthetic_data(duration=60, dt=0.01):
    """
    Generates synthetic heading direction data with:
    - Coherent noise
    - Diffusive noise with jumps
    - Coherent sweeps
    """
    t = np.arange(0, duration, dt)
    n_points = len(t)
    
    # Start with random walk (diffusive)
    rw = np.cumsum(np.random.randn(n_points)) * 0.1
    
    # Add jumps to random walk
    n_jumps = 10
    jump_indices = np.random.choice(n_points, n_jumps, replace=False)
    for idx in jump_indices:
        rw[idx:] += np.random.uniform(-np.pi, np.pi)
        
    # Wrap to -pi, pi
    hd = np.angle(np.exp(1j * rw))
    
    # Add coherent sweeps
    # A sweep is a linear ramp
    n_sweeps = 5
    gt_intervals = []
    
    for _ in range(n_sweeps):
        start_idx = np.random.randint(0, n_points - 200)
        dur = np.random.randint(50, 150) # points
        end_idx = start_idx + dur
        
        # Create sweep
        sweep_slope = np.random.choice([-1, 1]) * np.random.uniform(3, 10) # rad/s
        segment_t = t[start_idx:end_idx] - t[start_idx]
        sweep_data = segment_t * sweep_slope
        
        # Replace data, smoothing edges slightly to avoid massive discontinuity at start (optional)
        # But user says fast sweeps in noisy data, so let's just overwrite
        hd[start_idx:end_idx] = np.angle(np.exp(1j * sweep_data))
        
        gt_intervals.append([t[start_idx], t[end_idx]])
        
    tsd = nap.Tsd(t=t, d=hd)
    return tsd, nap.IntervalSet(start=np.array(gt_intervals)[:,0], end=np.array(gt_intervals)[:,1])

if __name__ == "__main__":
    # 1. Generate Data
    print("Generating synthetic data...")
    hd_tsd, ground_truth = generate_synthetic_data()
    
    # 2. Detect Sweeps
    print("Detecting sweeps...")
    # Parameters need tuning based on the specific dynamics of the signal
    detected_epochs = detect_sweeps(hd_tsd, window_duration=0.5, slope_threshold=2.0)
    
    print(f"Detected {len(detected_epochs)} potential sweep epochs.")
    
    # 3. Visualization
    try:
        plt.figure(figsize=(15, 6))
        # Plot full data in grey
        plt.plot(hd_tsd.index, hd_tsd.values, '.', color='lightgrey', markersize=1, label='Raw HD')
        
        # Plot detected intervals
        for s, e in detected_epochs.values:
            mask = (hd_tsd.index >= s) & (hd_tsd.index <= e)
            idx = hd_tsd.index[mask]
            data = hd_tsd.values[mask]
            plt.plot(idx, data, '.', color='red', markersize=2)
            
        plt.title("Detected Coherent Sweeps (Red) over Diffusive/Noisy Background")
        plt.xlabel("Time (s)")
        plt.ylabel("Heading (rad)")
        plt.ylim(-np.pi, np.pi)
        
        # Indicate ground truth (if using synthetic)
        # for s, e in ground_truth.values:
        #     plt.axvspan(s, e, color='green', alpha=0.1)
            
        plt.tight_layout()
        plt.savefig("sweep_detection_demo.png")
        print("Detailed plot saved to sweep_detection_demo.png")
    except Exception as e:
        print(f"Could not plot: {e}")
