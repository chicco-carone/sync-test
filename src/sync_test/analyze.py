#!/usr/bin/env python3
"""
Analyze sync measurement CSV files and generate statistics.

Usage:
    python analyze.py <csv_file> [--plot]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_csv(filepath: Path) -> dict:
    """Load measurement CSV file."""
    data = {
        'timestamp': [],
        'offset_us': [],
        'delta_us': [],
        'correlation': [],
        'is_snap': []
    }

    with open(filepath, 'r') as f:
        header = f.readline().strip()
        expected = "timestamp_unix,offset_us,delta_us,correlation_peak,is_snap"
        if header != expected:
            print(f"Warning: Unexpected header format")
            print(f"  Expected: {expected}")
            print(f"  Got: {header}")

        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 5:
                data['timestamp'].append(float(parts[0]))
                data['offset_us'].append(float(parts[1]))
                data['delta_us'].append(float(parts[2]))
                data['correlation'].append(float(parts[3]))
                data['is_snap'].append(int(parts[4]))

    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key])

    return data


def compute_statistics(data: dict, min_correlation: float = 0.5) -> dict:
    """Compute all statistics from measurement data."""
    # Filter out low-correlation measurements
    valid_mask = data['correlation'] >= min_correlation
    num_total = len(data['offset_us'])
    num_valid = np.sum(valid_mask)
    num_skipped = num_total - num_valid

    # Round all offsets to whole microseconds for analysis
    offsets = np.round(data['offset_us'][valid_mask])
    abs_offsets = np.abs(offsets)
    deltas = data['delta_us'][valid_mask]
    is_snap = data['is_snap'][valid_mask]
    correlations = data['correlation'][valid_mask]
    timestamps = data['timestamp'][valid_mask]

    if len(offsets) == 0:
        raise ValueError(f"No valid measurements (all {num_total} samples below correlation threshold {min_correlation})")

    # Non-snap deltas for drift stability
    non_snap_deltas = deltas[is_snap == 0]

    # Snap analysis
    snap_indices = np.where(is_snap == 1)[0]
    snap_deltas = deltas[is_snap == 1]

    # Duration
    duration_seconds = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0

    stats = {
        # Basic info
        'num_measurements': num_valid,
        'num_skipped': num_skipped,
        'num_total': num_total,
        'duration_seconds': duration_seconds,
        'duration_hours': duration_seconds / 3600,

        # Signed offset stats (for bias detection)
        'signed': {
            'mean': np.mean(offsets),
            'std': np.std(offsets),
            'min': np.min(offsets),
            'max': np.max(offsets),
        },

        # Absolute offset stats (primary metrics for sync quality)
        'absolute': {
            'mean': np.mean(abs_offsets),
            'std': np.std(abs_offsets),
            'max': np.max(abs_offsets),
            'percentile_50': np.percentile(abs_offsets, 50),
            'percentile_90': np.percentile(abs_offsets, 90),
            'percentile_95': np.percentile(abs_offsets, 95),
            'percentile_99': np.percentile(abs_offsets, 99),
        },

        # Bounded range
        'bounded_range': np.max(offsets) - np.min(offsets),

        # Drift stability
        'drift_stability': {
            'std': np.std(non_snap_deltas) if len(non_snap_deltas) > 1 else 0,
            'mean_abs': np.mean(np.abs(non_snap_deltas)) if len(non_snap_deltas) > 0 else 0,
        },

        # Snap analysis
        'snaps': {
            'count': len(snap_indices),
            'frequency_per_hour': len(snap_indices) / (duration_seconds / 3600) if duration_seconds > 0 else 0,
            'mean_interval_seconds': duration_seconds / len(snap_indices) if len(snap_indices) > 0 else float('inf'),
            'mean_magnitude': np.mean(np.abs(snap_deltas)) if len(snap_deltas) > 0 else 0,
            'max_magnitude': np.max(np.abs(snap_deltas)) if len(snap_deltas) > 0 else 0,
        },

        # Correlation stats
        'correlation': {
            'mean': np.mean(correlations),
            'min': np.min(correlations),
            'std': np.std(correlations),
        },
    }

    return stats


def print_report(stats: dict, filepath: Path):
    """Print formatted statistics report."""
    print()
    print("=" * 70)
    print("SYNC MEASUREMENT ANALYSIS REPORT")
    print("=" * 70)
    print(f"  File: {filepath.name}")
    print(f"  Valid measurements: {stats['num_measurements']:,}")
    if stats['num_skipped'] > 0:
        print(f"  Skipped (low correlation): {stats['num_skipped']:,}")

    # Format duration nicely
    duration_secs = stats['duration_seconds']
    hours = int(duration_secs // 3600)
    minutes = int((duration_secs % 3600) // 60)
    seconds = int(duration_secs % 60)

    if hours > 0:
        duration_str = f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        duration_str = f"{minutes}m {seconds}s"
    else:
        duration_str = f"{seconds}s"

    print(f"  Duration: {duration_str} ({stats['duration_hours']:.2f} hours)")

    # Add cadence information
    if stats['duration_seconds'] > 0:
        samples_per_minute = stats['num_measurements'] / (stats['duration_seconds'] / 60)
        samples_per_second = stats['num_measurements'] / stats['duration_seconds']
        mean_interval = stats['duration_seconds'] / max(stats['num_measurements'] - 1, 1)

        print(f"  Sample rate: {samples_per_second:.2f} samples/sec ({samples_per_minute:.1f} samples/min)")
        print(f"  Mean interval: {mean_interval:.2f} seconds between samples")

    print()
    print("-" * 70)
    print("SYNC QUALITY (absolute offset)")
    print("-" * 70)
    print(f"  Mean absolute offset:    {stats['absolute']['mean']:8.2f} μs")
    print(f"  Std dev:                 {stats['absolute']['std']:8.2f} μs")
    print(f"  Median (50th pct):       {stats['absolute']['percentile_50']:8.2f} μs")
    print(f"  90th percentile:         {stats['absolute']['percentile_90']:8.2f} μs")
    print(f"  95th percentile:         {stats['absolute']['percentile_95']:8.2f} μs")
    print(f"  99th percentile:         {stats['absolute']['percentile_99']:8.2f} μs")
    print(f"  Max absolute offset:     {stats['absolute']['max']:8.2f} μs")
    print(f"  Bounded range:           {stats['bounded_range']:8.2f} μs")

    print()
    print("-" * 70)
    print("OFFSET BIAS (signed)")
    print("-" * 70)
    print(f"  Mean offset:             {stats['signed']['mean']:+8.2f} μs")
    print(f"  Std dev:                 {stats['signed']['std']:8.2f} μs")
    print(f"  Range:                   [{stats['signed']['min']:+.1f}, {stats['signed']['max']:+.1f}] μs")

    bias_magnitude = abs(stats['signed']['mean'])
    if bias_magnitude < 10:
        bias_note = "(negligible)"
    elif bias_magnitude < 50:
        bias_note = "(small constant offset)"
    else:
        bias_note = "(significant - check device latencies)"
    print(f"  Bias assessment:         {bias_note}")

    print()
    print("-" * 70)
    print("DRIFT STABILITY")
    print("-" * 70)
    print(f"  Std dev of deltas:       {stats['drift_stability']['std']:8.3f} μs")
    print(f"  Mean |delta|:            {stats['drift_stability']['mean_abs']:8.3f} μs")

    if stats['drift_stability']['std'] < 0.1:
        stability_note = "(excellent - sub-100ns stability)"
    elif stats['drift_stability']['std'] < 0.5:
        stability_note = "(very good - sub-500ns stability)"
    elif stats['drift_stability']['std'] < 1.0:
        stability_note = "(good - sub-μs stability)"
    else:
        stability_note = "(moderate)"
    print(f"  Assessment:              {stability_note}")

    print()
    print("-" * 70)
    print("SIGNAL QUALITY")
    print("-" * 70)
    print(f"  Mean correlation:        {stats['correlation']['mean']:.4f}")
    print(f"  Min correlation:         {stats['correlation']['min']:.4f}")

    if stats['correlation']['mean'] > 0.95:
        corr_note = "(excellent signal quality)"
    elif stats['correlation']['mean'] > 0.85:
        corr_note = "(good signal quality)"
    elif stats['correlation']['mean'] > 0.70:
        corr_note = "(acceptable)"
    else:
        corr_note = "(poor - check signal levels)"
    print(f"  Assessment:              {corr_note}")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    # Helper function to format distance
    def format_distance(us_value):
        dist_mm = us_value * 0.343
        if dist_mm < 10:
            return f"{dist_mm:.2f} mm"
        else:
            return f"{dist_mm/10:.2f} cm"

    median = stats['absolute']['percentile_50']
    p95 = stats['absolute']['percentile_95']
    p99 = stats['absolute']['percentile_99']
    worst = stats['absolute']['max']

    print(f"  - Median sync error:     {median:.1f} μs ({format_distance(median)}*)")
    print(f"  - 95th percentile:       {p95:.1f} μs ({format_distance(p95)}*)")
    print(f"  - 99th percentile:       {p99:.1f} μs ({format_distance(p99)}*)")
    print(f"  - Worst case:            {worst:.1f} μs ({format_distance(worst)}*)")
    print()
    print("  * Distance calculated using speed of sound in air at 20°C (343 m/s)")
    print()
    print("=" * 70)


def compute_rolling_statistics(timestamps, abs_offsets, window_minutes=5):
    """
    Compute rolling window statistics.

    Args:
        timestamps: Array of unix timestamps
        abs_offsets: Array of absolute offset values
        window_minutes: Size of rolling window in minutes

    Returns:
        Dict with rolling statistics arrays
    """
    window_seconds = window_minutes * 60

    # Initialize arrays
    window_centers = []
    rolling_mean = []
    rolling_std = []
    rolling_p95 = []
    rolling_median = []

    # Start from the first timestamp
    start_time = timestamps[0]
    end_time = timestamps[-1]

    # Step through time with overlapping windows (50% overlap)
    step = window_seconds / 2
    current_time = start_time + window_seconds / 2

    while current_time < end_time - window_seconds / 2:
        # Find samples in this window
        window_start = current_time - window_seconds / 2
        window_end = current_time + window_seconds / 2

        mask = (timestamps >= window_start) & (timestamps < window_end)
        window_data = abs_offsets[mask]

        if len(window_data) >= 10:  # Require at least 10 samples
            window_centers.append((current_time - start_time) / 60)  # Convert to minutes
            rolling_mean.append(np.mean(window_data))
            rolling_std.append(np.std(window_data))
            rolling_p95.append(np.percentile(window_data, 95))
            rolling_median.append(np.percentile(window_data, 50))

        current_time += step

    return {
        'time_minutes': np.array(window_centers),
        'mean': np.array(rolling_mean),
        'std': np.array(rolling_std),
        'p95': np.array(rolling_p95),
        'median': np.array(rolling_median),
    }


def plot_results(data: dict, stats: dict, filepath: Path, min_correlation: float = 0.5,
                 rolling_window_minutes: int | None = None):
    """Generate visualization plots.

    Args:
        data: Measurement data dictionary
        stats: Computed statistics dictionary
        filepath: Path to the CSV file
        min_correlation: Minimum correlation threshold for filtering
        rolling_window_minutes: If set, creates additional rolling window analysis plots
    """
    if not HAS_MATPLOTLIB:
        print("\nMatplotlib not installed. Skipping plots.")
        print("Install with: uv add matplotlib")
        return

    # Filter out low-correlation measurements for offset plots
    valid_mask = data['correlation'] >= min_correlation
    # Round all offsets to whole microseconds for plotting
    offsets = np.round(data['offset_us'][valid_mask])
    abs_offsets = np.abs(offsets)
    timestamps = data['timestamp'][valid_mask]
    deltas = data['delta_us'][valid_mask]

    # Convert timestamps to relative time in minutes
    time_minutes = (timestamps - data['timestamp'][0]) / 60

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # Add title with different font sizes
    fig.text(0.5, 0.98, 'Sendspin Synchronization Accuracy', ha='center', fontsize=16, fontweight='bold')
    fig.text(0.5, 0.955, 'Audio playback synchronization between two Home Assistant Voice Preview Edition devices',
             ha='center', fontsize=11)

    # Plot 1: Offset over time (signed)
    ax1 = axes[0, 0]
    ax1.plot(time_minutes, offsets, 'b-', linewidth=0.5, alpha=0.7)
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Offset (μs)')
    ax1.set_title('Signed Offset Over Time')
    ax1.grid(True, alpha=0.3)
    # Make y-axis symmetric about 0
    max_abs_offset = np.max(np.abs(offsets))
    ax1.set_ylim(-max_abs_offset, max_abs_offset)
    # Add secondary y-axis for distance (speed of sound: 343 m/s = 0.343 mm/μs)
    max_distance_mm = max_abs_offset * 0.343
    if max_distance_mm < 10:
        # Use mm for small distances
        ax1_dist = ax1.secondary_yaxis('right', functions=(lambda x: x * 0.343, lambda x: x / 0.343))
        ax1_dist.set_ylabel('Distance* (mm)')
    else:
        # Use cm for larger distances
        ax1_dist = ax1.secondary_yaxis('right', functions=(lambda x: x * 0.0343, lambda x: x / 0.0343))
        ax1_dist.set_ylabel('Distance* (cm)')

    # Plot 2: Absolute offset histogram
    ax2 = axes[0, 1]
    # Use whole-number bin widths (1, 2, 5, 10, etc.)
    max_offset = np.max(abs_offsets)
    # Aim for roughly 30-50 bins, but use whole number widths
    target_bins = 40
    bin_width = max(1, int(np.ceil(max_offset / target_bins)))
    bins = np.arange(0, max_offset + bin_width + 1, bin_width)
    ax2.hist(abs_offsets, bins=bins, edgecolor='black', alpha=0.7)

    # Add distance to legend labels
    median_us = stats['absolute']['percentile_50']
    p95_us = stats['absolute']['percentile_95']
    median_dist_mm = median_us * 0.343
    p95_dist_mm = p95_us * 0.343

    # Format distance based on magnitude
    if median_dist_mm < 10:
        median_dist_str = f'{median_dist_mm:.2f} mm'
    else:
        median_dist_str = f'{median_dist_mm/10:.2f} cm'

    if p95_dist_mm < 10:
        p95_dist_str = f'{p95_dist_mm:.2f} mm'
    else:
        p95_dist_str = f'{p95_dist_mm/10:.2f} cm'

    ax2.axvline(x=median_us, color='red', linestyle='-', linewidth=2,
                label=f'Median: {median_us:.1f}μs ({median_dist_str}*)')
    ax2.axvline(x=p95_us, color='orange', linestyle='--', linewidth=2,
                label=f'95th pct: {p95_us:.1f}μs ({p95_dist_str}*)')
    ax2.set_xlabel('Absolute Offset (μs)')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Absolute Offset')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Cumulative distribution function
    ax3 = axes[1, 0]
    # Sort absolute offsets for CDF
    sorted_abs_offsets = np.sort(abs_offsets)
    cumulative_prob = np.arange(1, len(sorted_abs_offsets) + 1) / len(sorted_abs_offsets) * 100
    ax3.plot(sorted_abs_offsets, cumulative_prob, 'b-', linewidth=1.5)

    # Add key percentile markers with distance
    for percentile, color, style in [(50, 'red', '-'), (95, 'orange', '--'), (99, 'purple', ':')]:
        value = np.percentile(abs_offsets, percentile)
        dist_mm = value * 0.343

        # Format distance based on magnitude
        if dist_mm < 10:
            dist_str = f'{dist_mm:.2f} mm'
        else:
            dist_str = f'{dist_mm/10:.2f} cm'

        ax3.axvline(x=value, color=color, linestyle=style, linewidth=1.5, alpha=0.7,
                   label=f'{percentile}th: {value:.1f}μs ({dist_str}*)')
        ax3.axhline(y=percentile, color=color, linestyle=style, linewidth=1, alpha=0.3)

    ax3.set_xlabel('Absolute Offset (μs)')
    ax3.set_ylabel('Cumulative Probability (%)')
    ax3.set_title('Cumulative Distribution of Absolute Offset')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)

    # Add secondary x-axis for distance
    max_distance_mm_cdf = np.max(abs_offsets) * 0.343
    if max_distance_mm_cdf < 10:
        ax3_dist = ax3.secondary_xaxis('top', functions=(lambda x: x * 0.343, lambda x: x / 0.343))
        ax3_dist.set_xlabel('Distance* (mm)')
    else:
        ax3_dist = ax3.secondary_xaxis('top', functions=(lambda x: x * 0.0343, lambda x: x / 0.0343))
        ax3_dist.set_xlabel('Distance* (cm)')

    # Plot 4: Drift rate over time (delta between measurements)
    ax4 = axes[1, 1]

    ax4.plot(time_minutes, deltas, 'g-', linewidth=0.5, alpha=0.7)
    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

    # Add drift stability metric
    drift_std = np.std(deltas) if len(deltas) > 0 else 0
    ax4.axhline(y=drift_std, color='orange', linestyle='--', alpha=0.7, label=f'±1σ: {drift_std:.3f}μs')
    ax4.axhline(y=-drift_std, color='orange', linestyle='--', alpha=0.7)

    ax4.set_xlabel('Time (minutes)')
    ax4.set_ylabel('Drift Rate (μs/sample)')
    ax4.set_title('Drift Rate Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Make y-axis symmetric about 0
    if len(deltas) > 0:
        max_abs_drift = np.max(np.abs(deltas))
        ax4.set_ylim(-max_abs_drift, max_abs_drift)

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])

    # Add note about distance calculation (bottom left)
    fig.text(0.02, 0.01, '* Distance calculated using speed of sound in air at 20°C (343 m/s)',
             ha='left', fontsize=9, style='italic')
    # Add filename at bottom right
    fig.text(0.98, 0.01, f'{filepath.name}', ha='right', fontsize=9, style='italic', color='gray')

    # Save plot
    plot_path = filepath.with_suffix('.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to: {plot_path}")

    plt.show()

    # Create rolling window analysis if requested
    if rolling_window_minutes is not None:
        print(f"\nGenerating rolling window analysis ({rolling_window_minutes}-minute windows)...")
        plot_rolling_window_analysis(timestamps, abs_offsets, filepath, rolling_window_minutes)


def plot_rolling_window_analysis(timestamps, abs_offsets, filepath: Path, window_minutes: int):
    """Create rolling window analysis visualization."""
    if not HAS_MATPLOTLIB:
        return

    rolling_stats = compute_rolling_statistics(timestamps, abs_offsets, window_minutes)

    if len(rolling_stats['time_minutes']) == 0:
        print(f"Warning: Not enough data for {window_minutes}-minute rolling windows. Try a smaller window size.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f'Rolling Window Analysis ({window_minutes}-minute windows)', fontsize=14, fontweight='bold')

    # Plot 1: Rolling statistics over time
    ax1 = axes[0]
    ax1.plot(rolling_stats['time_minutes'], rolling_stats['median'], 'b-', linewidth=2, label='Median', alpha=0.8)
    ax1.plot(rolling_stats['time_minutes'], rolling_stats['mean'], 'g--', linewidth=1.5, label='Mean', alpha=0.7)
    ax1.plot(rolling_stats['time_minutes'], rolling_stats['p95'], 'r:', linewidth=2, label='95th percentile', alpha=0.8)

    # Add trend line for median
    if len(rolling_stats['time_minutes']) > 1:
        z = np.polyfit(rolling_stats['time_minutes'], rolling_stats['median'], 1)
        p = np.poly1d(z)
        ax1.plot(rolling_stats['time_minutes'], p(rolling_stats['time_minutes']), 'k--',
                linewidth=1, alpha=0.5, label=f'Trend: {z[0]:.3f}μs/min')

    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Absolute Offset (μs)')
    ax1.set_title('Rolling Window Statistics')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Rolling standard deviation (stability indicator)
    ax2 = axes[1]
    ax2.plot(rolling_stats['time_minutes'], rolling_stats['std'], 'purple', linewidth=2)
    ax2.fill_between(rolling_stats['time_minutes'], 0, rolling_stats['std'], alpha=0.3, color='purple')

    # Color background based on trend (improving = green, degrading = red)
    if len(rolling_stats['std']) > 10:
        # Compute trend of std over time
        z_std = np.polyfit(rolling_stats['time_minutes'], rolling_stats['std'], 1)
        if z_std[0] < -0.001:  # Improving
            ax2.set_facecolor('#e8f5e9')  # Light green
            trend_text = 'IMPROVING: Variability decreasing over time'
            trend_color = 'green'
        elif z_std[0] > 0.001:  # Degrading
            ax2.set_facecolor('#ffebee')  # Light red
            trend_text = 'DEGRADING: Variability increasing over time'
            trend_color = 'red'
        else:  # Stable
            ax2.set_facecolor('#f5f5f5')  # Light gray
            trend_text = 'STABLE: Consistent variability over time'
            trend_color = 'gray'

        ax2.text(0.5, 0.95, trend_text, transform=ax2.transAxes,
                ha='center', va='top', fontsize=11, fontweight='bold',
                color=trend_color, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Standard Deviation (μs)')
    ax2.set_title('Sync Stability Over Time')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save rolling window plot
    rolling_plot_path = filepath.with_stem(filepath.stem + '_rolling').with_suffix('.png')
    plt.savefig(rolling_plot_path, dpi=150)
    print(f"Rolling window plot saved to: {rolling_plot_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze sync measurement CSV files"
    )
    parser.add_argument('csv_file', type=str,
                        help='Path to the CSV file to analyze')
    parser.add_argument('--plot', '-p', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--min-correlation', '-c', type=float, default=0.5,
                        help='Minimum correlation threshold (default: 0.5)')
    parser.add_argument('--last-seconds', '-l', type=float, default=None,
                        help='Only analyze the last N seconds of data')
    parser.add_argument('--rolling-window', '-r', type=int, default=None,
                        help='Enable rolling window analysis with specified window size in minutes (e.g., 5)')

    args = parser.parse_args()

    filepath = Path(args.csv_file)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    print(f"Loading {filepath}...")
    data = load_csv(filepath)

    if len(data['offset_us']) == 0:
        print("Error: No data found in file")
        sys.exit(1)

    print(f"Loaded {len(data['offset_us']):,} measurements")

    # Filter to last N seconds if requested
    if args.last_seconds is not None:
        last_timestamp = data['timestamp'][-1]
        cutoff_time = last_timestamp - args.last_seconds
        time_mask = data['timestamp'] >= cutoff_time

        # Filter all data arrays
        for key in data:
            data[key] = data[key][time_mask]

        print(f"Filtered to last {args.last_seconds} seconds ({len(data['offset_us']):,} measurements)")

    stats = compute_statistics(data, min_correlation=args.min_correlation)
    print_report(stats, filepath)

    if args.plot:
        plot_results(data, stats, filepath, min_correlation=args.min_correlation,
                    rolling_window_minutes=args.rolling_window)


if __name__ == "__main__":
    main()
