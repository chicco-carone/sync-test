#!/usr/bin/env python3
"""
Live sync measurement monitor.

Polls a CSV file being written by measure_sync.py and displays live updating plots.

Usage:
    python monitor.py <csv_file> [--mode full|simple] [--interval SECONDS]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class LiveMonitor:
    """Live monitoring dashboard for sync measurements."""

    def __init__(self, csv_file: Path, mode: str = 'simple', poll_interval: float = 1.0,
                 min_correlation: float = 0.5, window_minutes: float = 10.0):
        """
        Initialize the live monitor.

        Args:
            csv_file: Path to the CSV file being written
            mode: 'simple' (2-panel) or 'full' (4-panel)
            poll_interval: How often to poll the file (seconds)
            min_correlation: Minimum correlation threshold
            window_minutes: Time window to display (minutes)
        """
        self.csv_file = csv_file
        self.mode = mode
        self.poll_interval = poll_interval
        self.min_correlation = min_correlation
        self.window_minutes = window_minutes

        # Data storage
        self.timestamps = []
        self.offsets = []
        self.deltas = []
        self.correlations = []
        self.last_read_lines = 0

        # Track start time for relative time display
        self.start_time = None

    def load_new_data(self):
        """Load new data from the CSV file since last read."""
        if not self.csv_file.exists():
            return False

        try:
            with open(self.csv_file, 'r') as f:
                lines = f.readlines()

            # Skip header and already-read lines
            new_lines = lines[max(1, self.last_read_lines):]
            if not new_lines:
                return False

            # Parse new data
            for line in new_lines:
                parts = line.strip().split(',')
                if len(parts) >= 5:
                    try:
                        timestamp = float(parts[0])
                        offset = float(parts[1])
                        delta = float(parts[2])
                        correlation = float(parts[3])

                        self.timestamps.append(timestamp)
                        self.offsets.append(offset)
                        self.deltas.append(delta)
                        self.correlations.append(correlation)
                    except ValueError:
                        continue

            self.last_read_lines = len(lines)

            # Set start time on first data
            if self.start_time is None and self.timestamps:
                self.start_time = self.timestamps[0]

            return True

        except (IOError, OSError):
            return False

    def get_filtered_data(self):
        """Get filtered and windowed data for plotting."""
        if not self.timestamps:
            return {
                'time_minutes': np.array([]),
                'offsets': np.array([]),
                'abs_offsets': np.array([]),
                'deltas': np.array([]),
                'time_minutes_all': np.array([]),
                'offsets_all': np.array([]),
                'abs_offsets_all': np.array([]),
                'correlations': np.array([]),
                'all_time_abs_offsets': np.array([]),  # All-time data for histogram
            }

        # Convert to numpy arrays
        timestamps = np.array(self.timestamps)
        offsets = np.array(self.offsets)
        deltas = np.array(self.deltas)
        correlations = np.array(self.correlations)

        # Filter by correlation
        valid_mask = correlations >= self.min_correlation
        timestamps_filtered = timestamps[valid_mask]
        offsets_filtered = offsets[valid_mask]
        deltas_filtered = deltas[valid_mask]

        if len(timestamps_filtered) == 0:
            return {
                'time_minutes': np.array([]),
                'offsets': np.array([]),
                'abs_offsets': np.array([]),
                'deltas': np.array([]),
                'time_minutes_all': np.array([]),
                'offsets_all': np.array([]),
                'abs_offsets_all': np.array([]),
                'correlations': np.array([]),
                'all_time_abs_offsets': np.array([]),
            }

        # Round offsets to whole microseconds
        offsets_filtered = np.round(offsets_filtered)

        # Save all-time absolute offsets for histogram
        all_time_abs_offsets = np.abs(offsets_filtered)

        # Convert to relative time in minutes
        time_minutes_all = (timestamps_filtered - self.start_time) / 60

        # Window to last N minutes (for windowed time-series plots)
        if self.window_minutes > 0:
            current_time = time_minutes_all[-1] if len(time_minutes_all) > 0 else 0
            window_start = current_time - self.window_minutes
            window_mask = time_minutes_all >= window_start

            time_minutes_windowed = time_minutes_all[window_mask]
            offsets_windowed = offsets_filtered[window_mask]
            deltas_windowed = deltas_filtered[window_mask]
        else:
            time_minutes_windowed = time_minutes_all
            offsets_windowed = offsets_filtered
            deltas_windowed = deltas_filtered

        return {
            # Windowed data (for windowed plots)
            'time_minutes': time_minutes_windowed,
            'offsets': offsets_windowed,
            'abs_offsets': np.abs(offsets_windowed),
            'deltas': deltas_windowed,
            # All-time data
            'time_minutes_all': time_minutes_all,
            'offsets_all': offsets_filtered,
            'abs_offsets_all': all_time_abs_offsets,
            'correlations': correlations,  # Keep all for stats
            'all_time_abs_offsets': all_time_abs_offsets,  # All-time data for histogram
        }

    def create_simple_view(self):
        """Create simplified 2-panel view."""
        self.fig, self.axes = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.suptitle(f'Live Sync Monitor: {self.csv_file.name}', fontsize=14, fontweight='bold')

        # Panel 1: Offset over time
        self.ax_offset = self.axes[0]
        self.line_offset, = self.ax_offset.plot([], [], 'b-', linewidth=0.8)
        self.ax_offset.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        self.ax_offset.set_xlabel('Time (minutes)')
        self.ax_offset.set_ylabel('Offset (μs)')
        self.ax_offset.set_title('Signed Offset Over Time')
        self.ax_offset.grid(True, alpha=0.3)

        # Panel 2: Rolling statistics
        self.ax_stats = self.axes[1]
        self.text_stats = self.ax_stats.text(0.5, 0.5, 'Waiting for data...',
                                             ha='center', va='center',
                                             fontsize=12, transform=self.ax_stats.transAxes)
        self.ax_stats.set_xlim(0, 1)
        self.ax_stats.set_ylim(0, 1)
        self.ax_stats.axis('off')

        plt.tight_layout()

    def create_full_view(self):
        """Create full 4-panel analysis view."""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle(f'Live Sync Monitor: {self.csv_file.name}', fontsize=14, fontweight='bold')

        # Panel 1: All-time offset over time (signed)
        self.ax_offset_all = self.axes[0, 0]
        self.line_offset_all, = self.ax_offset_all.plot([], [], 'b-', linewidth=0.5, alpha=0.7)
        self.ax_offset_all.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        self.ax_offset_all.set_xlabel('Time (minutes)')
        self.ax_offset_all.set_ylabel('Offset (μs)')
        self.ax_offset_all.set_title('Signed Offset Over Time (all-time)')
        self.ax_offset_all.grid(True, alpha=0.3)

        # Panel 2: Absolute offset histogram
        self.ax_hist = self.axes[0, 1]
        self.ax_hist.set_xlabel('Absolute Offset (μs)')
        self.ax_hist.set_ylabel('Count')
        self.ax_hist.set_title('Distribution of Absolute Offset')
        self.ax_hist.grid(True, alpha=0.3)

        # Panel 3: Statistics display
        self.ax_stats = self.axes[1, 0]
        self.text_stats = self.ax_stats.text(0.05, 0.95, 'Waiting for data...',
                                             ha='left', va='top',
                                             fontsize=9, transform=self.ax_stats.transAxes,
                                             family='monospace')
        self.ax_stats.set_xlim(0, 1)
        self.ax_stats.set_ylim(0, 1)
        self.ax_stats.axis('off')

        # Panel 4: Windowed offset over time (signed)
        self.ax_offset_window = self.axes[1, 1]
        self.line_offset_window, = self.ax_offset_window.plot([], [], 'b-', linewidth=0.5, alpha=0.7)
        self.ax_offset_window.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        self.ax_offset_window.set_xlabel('Time (minutes)')
        self.ax_offset_window.set_ylabel('Offset (μs)')
        window_title = 'all' if self.window_minutes == 0 else f'last {self.window_minutes:.0f} min'
        self.ax_offset_window.set_title(f'Signed Offset Over Time ({window_title})')
        self.ax_offset_window.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

        # Add note about distance calculation (bottom left)
        self.fig.text(0.02, 0.01, '* Distance calculated using speed of sound in air at 20°C (343 m/s)',
                     ha='left', fontsize=9, style='italic')

    def update_simple_view(self, data):
        """Update the simple 2-panel view."""
        if len(data['time_minutes']) == 0:
            return

        # Update offset plot
        self.line_offset.set_data(data['time_minutes'], data['offsets'])
        self.ax_offset.relim()
        self.ax_offset.autoscale_view(scalex=True, scaley=False)

        # Make y-axis symmetric about 0, based on actual data in window
        max_abs = np.max(np.abs(data['offsets'])) if len(data['offsets']) > 0 else 1
        self.ax_offset.set_ylim(-max_abs, max_abs)

        # Update statistics text (using windowed data for simple view)
        if len(data['abs_offsets']) > 0:
            median = np.percentile(data['abs_offsets'], 50)
            p95 = np.percentile(data['abs_offsets'], 95)
            p99 = np.percentile(data['abs_offsets'], 99)
            max_offset = np.max(data['abs_offsets'])
            mean_offset = np.mean(data['abs_offsets'])

            # Calculate distance for median
            median_dist_mm = median * 0.343
            if median_dist_mm < 10:
                median_dist = f"{median_dist_mm:.2f} mm"
            else:
                median_dist = f"{median_dist_mm/10:.2f} cm"

            stats_text = (
                f"Live Statistics (last {self.window_minutes:.0f} minutes)\n\n"
                f"  Samples: {len(data['abs_offsets']):,}\n\n"
                f"  Median:  {median:.1f} μs  ({median_dist})\n"
                f"  Mean:    {mean_offset:.1f} μs\n"
                f"  95th:    {p95:.1f} μs\n"
                f"  99th:    {p99:.1f} μs\n"
                f"  Max:     {max_offset:.1f} μs\n"
            )
            self.text_stats.set_text(stats_text)

    def update_full_view(self, data):
        """Update the full 4-panel view."""
        if len(data['time_minutes']) == 0:
            return

        # Update all-time offset plot (top-left)
        if len(data['time_minutes_all']) > 0:
            self.line_offset_all.set_data(data['time_minutes_all'], data['offsets_all'])
            self.ax_offset_all.relim()
            self.ax_offset_all.autoscale_view(scalex=True, scaley=False)

            # Make y-axis symmetric about 0, based on actual data
            max_abs = np.max(np.abs(data['offsets_all'])) if len(data['offsets_all']) > 0 else 1
            self.ax_offset_all.set_ylim(-max_abs, max_abs)

        # Update histogram (using all-time data) - top-right
        self.ax_hist.clear()
        if len(data['all_time_abs_offsets']) > 10:
            abs_offsets = data['all_time_abs_offsets']
            max_offset = np.max(abs_offsets)
            bin_width = max(1, int(np.ceil(max_offset / 40)))
            bins = np.arange(0, max_offset + bin_width + 1, bin_width)
            self.ax_hist.hist(abs_offsets, bins=bins, edgecolor='black', alpha=0.7)

            # Add percentile markers with distance labels
            median_us = np.percentile(abs_offsets, 50)
            p95_us = np.percentile(abs_offsets, 95)

            # Calculate distances
            median_dist_mm = median_us * 0.343
            p95_dist_mm = p95_us * 0.343

            # Format distance strings
            if median_dist_mm < 10:
                median_dist_str = f'{median_dist_mm:.2f} mm'
            else:
                median_dist_str = f'{median_dist_mm/10:.2f} cm'

            if p95_dist_mm < 10:
                p95_dist_str = f'{p95_dist_mm:.2f} mm'
            else:
                p95_dist_str = f'{p95_dist_mm/10:.2f} cm'

            # Add vertical lines with labels
            self.ax_hist.axvline(x=median_us, color='red', linestyle='-', linewidth=2,
                                label=f'Median: {median_us:.1f}μs ({median_dist_str}*)')
            self.ax_hist.axvline(x=p95_us, color='orange', linestyle='--', linewidth=2,
                                label=f'95th pct: {p95_us:.1f}μs ({p95_dist_str}*)')

        self.ax_hist.set_xlabel('Absolute Offset (μs)')
        self.ax_hist.set_ylabel('Count')
        self.ax_hist.set_title('Distribution of Absolute Offset (all-time)')
        self.ax_hist.legend()
        self.ax_hist.grid(True, alpha=0.3)

        # Update statistics text (bottom-left) - two columns: all-time and windowed
        if len(data['abs_offsets_all']) > 0:
            # Helper function to format distance
            def format_distance(us_value):
                dist_mm = us_value * 0.343
                if dist_mm < 10:
                    return f"{dist_mm:.2f} mm"
                else:
                    return f"{dist_mm/10:.2f} cm"

            # All-time statistics
            all_median = np.percentile(data['abs_offsets_all'], 50)
            all_p90 = np.percentile(data['abs_offsets_all'], 90)
            all_p95 = np.percentile(data['abs_offsets_all'], 95)
            all_p99 = np.percentile(data['abs_offsets_all'], 99)
            all_max = np.max(data['abs_offsets_all'])
            all_range = np.max(data['offsets_all']) - np.min(data['offsets_all'])

            # Windowed statistics
            win_samples = len(data['abs_offsets'])
            if win_samples > 0:
                win_median = np.percentile(data['abs_offsets'], 50)
                win_p90 = np.percentile(data['abs_offsets'], 90)
                win_p95 = np.percentile(data['abs_offsets'], 95)
                win_p99 = np.percentile(data['abs_offsets'], 99)
                win_max = np.max(data['abs_offsets'])
                win_range = np.max(data['offsets']) - np.min(data['offsets'])

                # Window label
                if self.window_minutes == 0:
                    window_label = "All-time"
                else:
                    window_label = f"Last {self.window_minutes:.0f} min"

                stats_text = (
                    f"           All-time         {window_label}\n"
                    f"\n"
                    f"Samples:   {len(data['abs_offsets_all']):<7,}        {win_samples}\n"
                    f"\n"
                    f"Median:    {all_median:.1f} μs          {win_median:.1f} μs\n"
                    f"           ({format_distance(all_median):<10})  ({format_distance(win_median)})\n"
                    f"\n"
                    f"90th:      {all_p90:.1f} μs          {win_p90:.1f} μs\n"
                    f"           ({format_distance(all_p90):<10})  ({format_distance(win_p90)})\n"
                    f"\n"
                    f"95th:      {all_p95:.1f} μs          {win_p95:.1f} μs\n"
                    f"           ({format_distance(all_p95):<10})  ({format_distance(win_p95)})\n"
                    f"\n"
                    f"99th:      {all_p99:.1f} μs          {win_p99:.1f} μs\n"
                    f"           ({format_distance(all_p99):<10})  ({format_distance(win_p99)})\n"
                    f"\n"
                    f"Max:       {all_max:.1f} μs          {win_max:.1f} μs\n"
                    f"           ({format_distance(all_max):<10})  ({format_distance(win_max)})\n"
                    f"\n"
                    f"Range:     {all_range:.1f} μs          {win_range:.1f} μs"
                )
            else:
                # If no windowed data, only show all-time
                stats_text = (
                    f"All-time Statistics\n\n"
                    f"Samples:   {len(data['abs_offsets_all']):,}\n\n"
                    f"Median:    {all_median:.1f} μs\n"
                    f"           ({format_distance(all_median)})\n\n"
                    f"90th:      {all_p90:.1f} μs\n"
                    f"           ({format_distance(all_p90)})\n\n"
                    f"95th:      {all_p95:.1f} μs\n"
                    f"           ({format_distance(all_p95)})\n\n"
                    f"99th:      {all_p99:.1f} μs\n"
                    f"           ({format_distance(all_p99)})\n\n"
                    f"Max:       {all_max:.1f} μs\n"
                    f"           ({format_distance(all_max)})\n\n"
                    f"Range:     {all_range:.1f} μs"
                )

            self.text_stats.set_text(stats_text)

        # Update windowed offset plot (bottom-right)
        if len(data['time_minutes']) > 0:
            self.line_offset_window.set_data(data['time_minutes'], data['offsets'])
            self.ax_offset_window.relim()
            self.ax_offset_window.autoscale_view(scalex=True, scaley=False)

            # Make y-axis symmetric about 0, based on actual data in window
            max_abs = np.max(np.abs(data['offsets'])) if len(data['offsets']) > 0 else 1
            self.ax_offset_window.set_ylim(-max_abs, max_abs)

    def update(self, frame):
        """Animation update callback."""
        # Load new data from file
        self.load_new_data()

        # Get filtered and windowed data
        data = self.get_filtered_data()

        # Update appropriate view
        if self.mode == 'simple':
            self.update_simple_view(data)
        else:
            self.update_full_view(data)

        return []

    def run(self):
        """Start the live monitor."""
        if not HAS_MATPLOTLIB:
            print("Error: matplotlib not installed.")
            print("Install with: uv add matplotlib")
            sys.exit(1)

        # Wait for file to exist
        if not self.csv_file.exists():
            print(f"Waiting for {self.csv_file} to be created...")
            while not self.csv_file.exists():
                time.sleep(0.5)
            print(f"File detected! Starting monitor...")
            time.sleep(0.5)  # Give it a moment to write header

        # Create the view
        if self.mode == 'simple':
            self.create_simple_view()
        else:
            self.create_full_view()

        # Set up animation with the specified poll interval
        interval_ms = int(self.poll_interval * 1000)
        self.anim = FuncAnimation(self.fig, self.update, interval=interval_ms,
                                 blit=False, cache_frame_data=False)

        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Live monitor for sync measurements"
    )
    parser.add_argument('csv_file', type=str,
                        help='Path to the CSV file being written by measure_sync')
    parser.add_argument('--mode', '-m', type=str, default='simple',
                        choices=['simple', 'full'],
                        help='Display mode: simple (2-panel) or full (4-panel, default: simple)')
    parser.add_argument('--interval', '-i', type=float, default=1.0,
                        help='Polling interval in seconds (default: 1.0)')
    parser.add_argument('--window', '-w', type=float, default=10.0,
                        help='Time window to display in minutes (default: 10.0, use 0 for all data)')
    parser.add_argument('--min-correlation', '-c', type=float, default=0.5,
                        help='Minimum correlation threshold (default: 0.5)')

    args = parser.parse_args()

    csv_file = Path(args.csv_file)

    monitor = LiveMonitor(
        csv_file=csv_file,
        mode=args.mode,
        poll_interval=args.interval,
        min_correlation=args.min_correlation,
        window_minutes=args.window
    )

    print(f"Live Sync Monitor")
    print(f"  File: {csv_file}")
    print(f"  Mode: {args.mode}")
    print(f"  Poll interval: {args.interval}s")
    print(f"  Window: {args.window} minutes" + (" (all data)" if args.window == 0 else ""))
    print(f"  Min correlation: {args.min_correlation}")
    print()

    try:
        monitor.run()
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
