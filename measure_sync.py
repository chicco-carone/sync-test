#!/usr/bin/env python3
"""
Real-time audio sync measurement tool.

Captures stereo audio from a USB audio interface, cross-correlates the two channels,
and logs the time offset between them. Designed for measuring sync accuracy between
two audio devices playing the same content.

Usage:
    python measure_sync.py [--device DEVICE_ID] [--duration SECONDS] [--output FILE]

The tool will:
1. Capture audio in 500ms windows (matching the chirp period)
2. Cross-correlate left and right channels
3. Find the correlation peak with sub-sample interpolation
4. Log timestamp, offset (μs), and correlation strength to CSV
"""

import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy import signal as sig

# Configuration
SAMPLE_RATE = 96000  # Capture at 96kHz for better timing resolution
WINDOW_DURATION_MS = 500  # Match chirp period
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION_MS / 1000)

# Correlation search range: ±5ms should be plenty for a synced system
MAX_LAG_MS = 5
MAX_LAG_SAMPLES = int(SAMPLE_RATE * MAX_LAG_MS / 1000)

# Microseconds per sample at 96kHz
US_PER_SAMPLE = 1_000_000 / SAMPLE_RATE  # ~10.42 μs


class SyncMeasurer:
    def __init__(self, output_file: str, device: int | None = None):
        self.output_file = Path(output_file)
        self.device = device
        self.running = False
        self.measurements = []
        self.start_time = None

        # Pre-allocate buffer for efficiency
        self.buffer = np.zeros((WINDOW_SAMPLES, 2), dtype=np.float32)

        # Stats for live display
        self.recent_offsets = []
        self.max_recent = 100  # Keep last 100 measurements for rolling stats

    def parabolic_interpolation(self, correlation: np.ndarray, peak_idx: int) -> float:
        """
        Sub-sample interpolation using parabolic fit.
        Returns the interpolated peak position relative to peak_idx.
        """
        if peak_idx <= 0 or peak_idx >= len(correlation) - 1:
            return 0.0

        y0 = correlation[peak_idx - 1]
        y1 = correlation[peak_idx]
        y2 = correlation[peak_idx + 1]

        # Parabolic interpolation formula
        denominator = y0 - 2 * y1 + y2
        if abs(denominator) < 1e-10:
            return 0.0

        offset = 0.5 * (y0 - y2) / denominator
        return offset

    def measure_offset(self, left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
        """
        Measure the time offset between two channels using cross-correlation.

        Returns:
            offset_us: Time offset in microseconds (positive = left ahead of right)
            correlation_strength: Normalized correlation peak (0-1)
        """
        # Normalize signals to avoid amplitude bias
        left_norm = left - np.mean(left)
        right_norm = right - np.mean(right)

        left_std = np.std(left_norm)
        right_std = np.std(right_norm)

        if left_std < 1e-10 or right_std < 1e-10:
            # Silence or near-silence
            return 0.0, 0.0

        left_norm /= left_std
        right_norm /= right_std

        # Compute cross-correlation using FFT for efficiency
        # We only need correlation for lags in [-MAX_LAG_SAMPLES, +MAX_LAG_SAMPLES]
        # Full correlation is overkill, but scipy's fftconvolve is still fast

        # Use correlate with 'full' mode, then extract the region we care about
        correlation = sig.correlate(left_norm, right_norm, mode='full', method='fft')

        # The center of the correlation output corresponds to lag=0
        center = len(right_norm) - 1

        # Extract only the lags we care about
        start_idx = center - MAX_LAG_SAMPLES
        end_idx = center + MAX_LAG_SAMPLES + 1
        correlation_window = correlation[start_idx:end_idx]

        # Find peak in the windowed correlation
        peak_idx_in_window = np.argmax(correlation_window)
        peak_value = correlation_window[peak_idx_in_window]

        # Normalize correlation strength (divide by length for normalized cross-correlation)
        correlation_strength = peak_value / len(left_norm)

        # Sub-sample interpolation
        sub_sample_offset = self.parabolic_interpolation(correlation_window, peak_idx_in_window)

        # Calculate lag in samples (relative to center of window)
        # scipy.correlate(a, b) peak at positive lag means b is delayed relative to a
        # So if peak is at +5 samples, right is 5 samples behind left (left is ahead)
        # We want: positive = left ahead, so we negate the lag
        lag_samples = (peak_idx_in_window - MAX_LAG_SAMPLES) + sub_sample_offset

        # Convert to microseconds
        # Negate because correlate convention: positive peak index = right is delayed = left is ahead
        # But our lag calculation gives positive when peak is at positive index
        # Actually, let's reconsider: correlate(left, right)[k] is sum of left[n]*right[n-k]
        # Peak at k>0 means right needs to be shifted back (right is ahead)
        # Peak at k<0 means right needs to be shifted forward (right is behind, left is ahead)
        # So: positive lag_samples from our calc = right is ahead = left is behind
        # We want positive to mean "left ahead", so we negate
        offset_us = -lag_samples * US_PER_SAMPLE

        return offset_us, correlation_strength

    def audio_callback(self, indata: np.ndarray, frames: int,
                       time_info: dict, status: sd.CallbackFlags):
        """Called by sounddevice for each audio block."""
        if status:
            print(f"Audio status: {status}", file=sys.stderr)

        # Copy data to our buffer (sounddevice may reuse the array)
        self.buffer[:frames] = indata[:frames]

    def write_header(self):
        """Write CSV header to output file."""
        with open(self.output_file, 'w') as f:
            f.write("timestamp_unix,offset_us,correlation_peak\n")

    def append_measurement(self, timestamp: float, offset_us: float, correlation: float):
        """Append a measurement to the CSV file."""
        with open(self.output_file, 'a') as f:
            f.write(f"{timestamp:.6f},{offset_us:.3f},{correlation:.6f}\n")

    def update_live_display(self, offset_us: float, correlation: float, elapsed: float):
        """Update the live terminal display."""
        self.recent_offsets.append(offset_us)
        if len(self.recent_offsets) > self.max_recent:
            self.recent_offsets.pop(0)

        # Calculate rolling stats
        if len(self.recent_offsets) >= 2:
            mean = np.mean(self.recent_offsets)
            std = np.std(self.recent_offsets)
            min_off = np.min(self.recent_offsets)
            max_off = np.max(self.recent_offsets)
        else:
            mean = offset_us
            std = 0
            min_off = offset_us
            max_off = offset_us

        # Format elapsed time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        print(f"\r[{hours:02d}:{minutes:02d}:{seconds:02d}] "
              f"Offset: {offset_us:+8.2f}μs | "
              f"Mean: {mean:+8.2f}μs | "
              f"Std: {std:6.2f}μs | "
              f"Range: [{min_off:+.1f}, {max_off:+.1f}]μs | "
              f"Corr: {correlation:.4f}",
              end='', flush=True)

    def run(self, duration: float | None = None):
        """
        Run the measurement loop.

        Args:
            duration: Duration in seconds, or None for indefinite
        """
        self.running = True
        self.start_time = time.time()
        self.write_header()

        print(f"Starting sync measurement...")
        print(f"  Sample rate: {SAMPLE_RATE} Hz")
        print(f"  Window: {WINDOW_DURATION_MS} ms ({WINDOW_SAMPLES} samples)")
        print(f"  Max lag search: ±{MAX_LAG_MS} ms (±{MAX_LAG_SAMPLES} samples)")
        print(f"  Resolution: ~{US_PER_SAMPLE:.2f} μs/sample (sub-sample interpolation enabled)")
        print(f"  Output: {self.output_file}")
        if duration:
            print(f"  Duration: {duration} seconds")
        else:
            print(f"  Duration: indefinite (Ctrl+C to stop)")
        print()

        # List devices if needed
        if self.device is None:
            print("Available audio devices:")
            print(sd.query_devices())
            print()
            print("Specify device with --device ID")
            return

        device_info = sd.query_devices(self.device)
        print(f"Using device: {device_info['name']}")
        print()

        measurement_count = 0

        try:
            with sd.InputStream(device=self.device,
                               channels=2,
                               samplerate=SAMPLE_RATE,
                               blocksize=WINDOW_SAMPLES,
                               dtype=np.float32,
                               callback=self.audio_callback):

                while self.running:
                    # Wait for a full window of samples
                    sd.sleep(int(WINDOW_DURATION_MS))

                    # Process the buffer
                    left = self.buffer[:, 0].copy()
                    right = self.buffer[:, 1].copy()

                    # Measure offset
                    offset_us, correlation = self.measure_offset(left, right)

                    # Record timestamp
                    timestamp = time.time()
                    elapsed = timestamp - self.start_time

                    # Log to file
                    self.append_measurement(timestamp, offset_us, correlation)
                    measurement_count += 1

                    # Update display
                    self.update_live_display(offset_us, correlation, elapsed)

                    # Check duration limit
                    if duration and elapsed >= duration:
                        self.running = False

        except KeyboardInterrupt:
            pass

        print(f"\n\nMeasurement complete!")
        print(f"  Total measurements: {measurement_count}")
        print(f"  Total duration: {time.time() - self.start_time:.1f} seconds")
        print(f"  Output saved to: {self.output_file}")

    def stop(self):
        """Stop the measurement loop."""
        self.running = False


def list_devices():
    """List all available audio devices."""
    print("Available audio input devices:")
    print("-" * 60)
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] >= 2:
            print(f"  [{i}] {dev['name']}")
            print(f"      Inputs: {dev['max_input_channels']}, "
                  f"Sample rate: {dev['default_samplerate']}")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Measure audio sync between two channels in real-time"
    )
    parser.add_argument('--device', '-d', type=int, default=None,
                        help='Audio input device ID (use --list to see devices)')
    parser.add_argument('--duration', '-t', type=float, default=None,
                        help='Duration in seconds (default: run until Ctrl+C)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output CSV file (default: sync_measurements_TIMESTAMP.csv)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List available audio devices and exit')

    args = parser.parse_args()

    if args.list:
        list_devices()
        return

    # Generate default output filename with timestamp
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"sync_measurements_{timestamp}.csv"

    # Create measurer and set up signal handling
    measurer = SyncMeasurer(args.output, args.device)

    def signal_handler(signum, frame):
        print("\nStopping...")
        measurer.stop()

    signal.signal(signal.SIGINT, signal_handler)

    # Run measurement
    measurer.run(args.duration)


if __name__ == "__main__":
    main()
