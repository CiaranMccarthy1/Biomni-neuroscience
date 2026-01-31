def analyze_spike_train_statistics(
    spike_times,
    duration=None,
    refractory_period=0.002,
    burst_threshold_ms=10,
    output_dir="spike_train_analysis_results",
    sample_name="neuron_1",
):
    """Analyzes neuronal spike train data to calculate firing statistics, burst metrics, and regularity.

    Parameters
    ----------
    spike_times : list or numpy.ndarray
        List of time stamps (in seconds) when spikes occurred
    duration : float, optional
        Total duration of the recording in seconds. If None, inferred from last spike.
    refractory_period : float, optional
        Minimum time (in seconds) between spikes to check for violations (default: 0.002)
    burst_threshold_ms : float, optional
        Max ISI (in ms) to consider two spikes part of a burst (default: 10)
    output_dir : str, optional
        Directory to save result files (default: "./spike_train_analysis_results")
    sample_name : str, optional
        Identifier for the sample to name output files (default: "neuron_1")

    Returns
    -------
    str
        Research log summarizing the spike train analysis

    """
    import os
    from datetime import datetime

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize log
    log = f"# Spike Train Analysis Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    log += f"Analyzing sample: {sample_name}\n"

    # 1. Data Loading & Cleaning
    log += "\n## Step 1: Data Preprocessing\n"
    try:
        spikes = np.sort(np.array(spike_times, dtype=float))
        num_spikes = len(spikes)

        if duration is None:
            duration = spikes[-1] + 0.1 if num_spikes > 0 else 1.0
            log += f"- Inferred duration from last spike: {duration:.2f} s\n"

        log += f"- Loaded {num_spikes} spikes over {duration:.2f} seconds\n"

        if num_spikes < 2:
            return log + "- Error: Insufficient spikes (<2) for analysis.\n"

    except Exception as e:
        return f"Error processing spike data: {str(e)}"

    # 2. Basic Statistics
    log += "\n## Step 2: Firing Statistics\n"

    mean_firing_rate = num_spikes / duration
    log += f"- Mean Firing Rate: {mean_firing_rate:.2f} Hz\n"

    # Inter-spike intervals (ISIs)
    isi = np.diff(spikes)
    mean_isi = np.mean(isi)
    std_isi = np.std(isi)
    cv_isi = std_isi / mean_isi if mean_isi > 0 else 0

    log += f"- Mean ISI: {mean_isi * 1000:.2f} ms\n"
    log += f"- Coefficient of Variation (CV): {cv_isi:.2f} "

    if cv_isi < 0.2:
        log += "(Regular/Pacemaker firing)\n"
    elif 0.8 < cv_isi < 1.2:
        log += "(Poisson-like/Irregular firing)\n"
    elif cv_isi > 1.2:
        log += "(Bursting firing)\n"
    else:
        log += "\n"

    # Refractory period violations
    violations = np.sum(isi < refractory_period)
    percent_violations = (violations / len(isi)) * 100
    log += f"- Refractory Period Violations (<{refractory_period * 1000}ms): {violations} ({percent_violations:.2f}%)\n"

    # 3. Burst Analysis
    log += "\n## Step 3: Burst Analysis\n"
    # Simple Max-Interval Burst Detection
    burst_threshold_s = burst_threshold_ms / 1000.0
    is_burst = isi < burst_threshold_s

    # Count bursts (groups of consecutive short ISIs)
    burst_count = 0
    in_burst = False
    spikes_in_bursts = 0

    for flag in is_burst:
        if flag and not in_burst:
            burst_count += 1
            in_burst = True
            spikes_in_bursts += 1  # First spike in pair
        elif flag and in_burst:
            spikes_in_bursts += 1  # Subsequent spikes
        elif not flag:
            in_burst = False

    burst_index = spikes_in_bursts / num_spikes if num_spikes > 0 else 0

    log += f"- Burst Threshold: {burst_threshold_ms} ms\n"
    log += f"- Number of Bursts Detected: {burst_count}\n"
    log += f"- Burst Index (% spikes in bursts): {burst_index * 100:.1f}%\n"

    # 4. Generate Plots
    log += "\n## Step 4: Visualization\n"

    # Plot 1: ISI Histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram restricted to 95th percentile to see structure
    isi_ms = isi * 1000
    cutoff = np.percentile(isi_ms, 95)
    ax1.hist(isi_ms[isi_ms < cutoff], bins=50, color="darkblue", alpha=0.7)
    ax1.set_title(f"ISI Distribution (CV={cv_isi:.2f})")
    ax1.set_xlabel("Inter-spike Interval (ms)")
    ax1.set_ylabel("Count")

    # Plot 2: Raster / Spike Train (first 5 seconds)
    display_dur = min(duration, 5.0)
    display_spikes = spikes[spikes < display_dur]
    ax2.vlines(display_spikes, 0, 1, color="black", linewidth=1)
    ax2.set_title(f"Spike Raster (First {display_dur}s)")
    ax2.set_xlabel("Time (s)")
    ax2.set_yticks([])
    ax2.set_xlim(0, display_dur)

    plot_path = os.path.join(output_dir, f"{sample_name}_analysis_plot.png")
    plt.savefig(plot_path)
    plt.close()
    log += f"- Summary plot saved to: {plot_path}\n"

    # 5. Save Data
    log += "\n## Step 5: Data Export\n"

    # Save stats summary
    stats_data = {
        "mean_rate_hz": mean_firing_rate,
        "mean_isi_s": mean_isi,
        "cv_isi": cv_isi,
        "burst_index": burst_index,
        "refractory_violations": violations,
    }
    stats_df = pd.DataFrame([stats_data])
    stats_path = os.path.join(output_dir, f"{sample_name}_stats.csv")
    stats_df.to_csv(stats_path, index=False)

    log += f"- Numerical stats saved to: {stats_path}\n"
    log += "\nAnalysis Complete."

    return log


def analyze_calcium_imaging_activity(
    traces, frame_rate=30.0, threshold_sigma=3.0, output_dir="./results", experiment_id="calcium_exp_1"
):
    """Analyzes calcium imaging fluorescence traces to detect neuronal activity events.

    Performs dF/F0 normalization, peak detection using a robust Z-score threshold,
    and generates raster plots of population activity.

    Parameters
    ----------
    traces : numpy.ndarray or str
        Raw fluorescence data. Can be a path to a CSV file (rows=frames, cols=cells)
        or a numpy array of shape (n_frames, n_cells).
    frame_rate : float, optional
        Sampling rate of the imaging in Hz (default: 30.0)
    threshold_sigma : float, optional
        Z-score threshold for defining a calcium event (default: 3.0)
    output_dir : str, optional
        Directory to save results (default: "./results")
    experiment_id : str, optional
        Identifier for the experiment (default: "calcium_exp_1")

    Returns
    -------
    str
        Research log summarizing the calcium activity analysis.

    """
    import os
    from datetime import datetime

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.signal import find_peaks
    from scipy.stats import zscore

    os.makedirs(output_dir, exist_ok=True)

    log = f"# Calcium Imaging Analysis Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    log += f"Experiment: {experiment_id}\n"

    # 1. Load Data
    log += "\n## Step 1: Loading Data\n"
    try:
        if isinstance(traces, str):
            if traces.endswith(".csv"):
                data = pd.read_csv(traces).values
            elif traces.endswith(".npy"):
                data = np.load(traces)
            else:
                return "Error: Unsupported file format. Use CSV or NPY."
            log += f"- Loaded file: {traces}\n"
        else:
            data = np.array(traces)

        # Ensure shape is (frames, cells)
        if data.shape[0] < data.shape[1]:
            log += "- Warning: Data transposed. Assuming shape (n_cells, n_frames). Transposing...\n"
            data = data.T

        n_frames, n_cells = data.shape
        duration = n_frames / frame_rate

        log += f"- Dimensions: {n_cells} cells, {n_frames} frames\n"
        log += f"- Frame Rate: {frame_rate} Hz (Duration: {duration:.2f} s)\n"

    except Exception as e:
        return f"Error loading traces: {str(e)}"

    # 2. Signal Processing (dF/F)
    log += "\n## Step 2: Signal Processing (dF/F0)\n"

    # Calculate F0 (baseline) as sliding window percentile or simple median
    # Here we use global median for simplicity, assuming infrequent spiking
    f0 = np.percentile(data, 20, axis=0)  # 20th percentile to estimate baseline noise

    # Avoid division by zero
    f0[f0 == 0] = 1.0

    # Calculate dF/F
    dff = (data - f0) / f0

    log += "- Calculated dF/F0 using 20th percentile baseline.\n"

    # 3. Event Detection
    log += "\n## Step 3: Event Detection\n"

    events_raster = np.zeros_like(dff)
    total_events = 0
    active_cells = 0

    event_rates = []

    for i in range(n_cells):
        cell_trace = dff[:, i]

        # Robust noise estimation (mad)
        median = np.median(cell_trace)
        mad = np.median(np.abs(cell_trace - median))
        sigma = 1.4826 * mad

        if sigma == 0:
            sigma = 1e-6

        # Detect peaks
        peaks, _ = find_peaks(cell_trace, height=threshold_sigma * sigma, distance=int(frame_rate / 2))

        events_raster[peaks, i] = 1
        num_peaks = len(peaks)
        total_events += num_peaks

        rate = num_peaks / duration
        event_rates.append(rate)

        if num_peaks > 0:
            active_cells += 1

    mean_population_rate = np.mean(event_rates)
    percent_active = (active_cells / n_cells) * 100

    log += f"- Detection Threshold: {threshold_sigma} sigma\n"
    log += f"- Total Events Detected: {total_events}\n"
    log += f"- Active Cells: {active_cells}/{n_cells} ({percent_active:.1f}%)\n"
    log += f"- Mean Population Event Rate: {mean_population_rate * 60:.2f} events/min\n"

    # 4. Visualization
    log += "\n## Step 4: Visualization\n"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Time axis
    time = np.arange(n_frames) / frame_rate

    # Plot 1: Heatmap of dF/F
    im = ax1.imshow(
        dff.T, aspect="auto", cmap="viridis", vmin=0, vmax=np.percentile(dff, 99), extent=[0, duration, n_cells, 0]
    )
    ax1.set_ylabel("Cell ID")
    ax1.set_title("Population Activity (dF/F0)")
    plt.colorbar(im, ax=ax1, label="dF/F0")

    # Plot 2: Raster Plot of Events
    # Get coordinates of events
    rows, cols = np.where(events_raster.T)
    # Convert rows (cell index) to y, cols (time index) to time
    event_times = cols / frame_rate

    ax2.scatter(event_times, rows, s=2, c="black", alpha=0.6)
    ax2.set_ylabel("Cell ID")
    ax2.set_xlabel("Time (s)")
    ax2.set_title(f"Detected Events (> {threshold_sigma}$\\sigma$)")
    ax2.set_xlim(0, duration)
    ax2.set_ylim(n_cells, 0)  # Invert y to match heatmap

    viz_path = os.path.join(output_dir, f"{experiment_id}_activity_map.png")
    plt.savefig(viz_path, dpi=150)
    plt.close()

    log += f"- Activity map saved to: {viz_path}\n"

    # 5. Save Results
    log += "\n## Step 5: Saving Results\n"

    # Save dF/F traces
    dff_path = os.path.join(output_dir, f"{experiment_id}_dff_traces.csv")
    dff_df = pd.DataFrame(dff, columns=[f"Cell_{i}" for i in range(n_cells)])
    dff_df.insert(0, "Time_s", time)
    dff_df.to_csv(dff_path, index=False)

    # Save events binary matrix
    events_path = os.path.join(output_dir, f"{experiment_id}_events_binary.csv")
    np.savetxt(events_path, events_raster, delimiter=",", fmt="%d")

    log += f"- Processed dF/F traces saved to: {dff_path}\n"
    log += f"- Binary event matrix saved to: {events_path}\n"

    return log


def generate_synthetic_spike_train(duration=5.0, firing_rate_hz=20.0, pattern="regular", noise_level=0.1):
    """Generates synthetic spike times for testing.

    Parameters
    ----------
    duration : float, optional
        Duration of the spike train in seconds (default: 5.0)
    firing_rate_hz : float, optional
        Mean firing rate in Hz (default: 20.0)
    pattern : str, optional
        Firing pattern: 'regular', 'poisson', or 'bursty' (default: "regular")
    noise_level : float, optional
        Jitter noise level for regular pattern (default: 0.1)

    Returns
    -------
    numpy.ndarray
        Array of spike times in seconds.
    """
    import numpy as np

    if pattern == "regular":
        # Pacemaker: evenly spaced intervals with gaussian jitter
        isi = 1.0 / firing_rate_hz
        intervals = np.random.normal(isi, isi * noise_level, int(duration * firing_rate_hz * 2))
        intervals = np.abs(intervals)
    elif pattern == "poisson":
        # Random: Exponential intervals
        isi = 1.0 / firing_rate_hz
        intervals = np.random.exponential(isi, int(duration * firing_rate_hz * 2))
    elif pattern == "bursty":
        # Bursting: Clusters of fast spikes followed by silence
        num_bursts = int(duration * (firing_rate_hz / 5))
        intervals = []
        for _ in range(num_bursts):
            intervals.extend(np.random.normal(0.005, 0.001, 5))  # 5 spikes @ 200Hz
            intervals.append(np.random.exponential(0.2))  # 200ms pause
        intervals = np.array(intervals)
    else:
        return np.array([])

    spikes = np.cumsum(intervals)
    return spikes[spikes < duration]


def get_brain_region_metadata(region_name):
    """Retrieves metadata, IDs, and hierarchy information for a specific brain region from Allen Brain Atlas.

    Parameters
    ----------
    region_name : str
         Common name or acronym of the brain region (e.g., 'Hippocampus', 'VISp').

    Returns
    -------
    dict
        Dictionary containing ID, name, acronym, and structure set information.
    """
    import requests

    url = "http://api.brain-map.org/api/v2/data/query.json"

    # Query structure by name or acronym (case-insensitive)
    query = (
        "criteria=model::Structure,"
        "rma::criteria,"
        f"[name$il'*{region_name}*']"
    )

    try:
        response = requests.get(url, params={"q": query}, timeout=10)
        data = response.json()

        if data["success"] and len(data["msg"]) > 0:
            # Return the first match
            structure = data["msg"][0]
            return {
                "id": structure.get("id"),
                "name": structure.get("name"),
                "acronym": structure.get("acronym"),
                "atlas_id": structure.get("atlas_id"),
                "structure_id_path": structure.get("structure_id_path"),
                "color_hex_triplet": structure.get("color_hex_triplet")
            }
        else:
            return {"error": f"Region '{region_name}' not found."}

    except Exception as e:
        return {"error": f"Failed to retrieve metadata: {str(e)}"}
