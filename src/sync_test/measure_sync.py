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
3. Find the correlation peak with FFT-based sinc interpolation (100x upsampling)
4. Log timestamp, offset (μs), delta, and correlation strength to CSV
5. Detect and track correction snaps
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
WINDOW_DURATION_MS = 750  # Longer than chirp period to guarantee full chirp capture
PROCESS_INTERVAL_MS = 500  # Process every 500ms (matching chirp period)
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION_MS / 1000)
PROCESS_INTERVAL_SAMPLES = int(SAMPLE_RATE * PROCESS_INTERVAL_MS / 1000)

# Correlation search range: ±5ms should be plenty for a synced system
MAX_LAG_MS = 5
MAX_LAG_SAMPLES = int(SAMPLE_RATE * MAX_LAG_MS / 1000)

# Microseconds per sample at 96kHz
US_PER_SAMPLE = 1_000_000 / SAMPLE_RATE  # ~10.42 μs

# Snap detection threshold (μs) - changes larger than this are considered corrections
SNAP_THRESHOLD_US = 8.0

# Minimum correlation to consider a measurement valid
MIN_CORRELATION = 0.5


class SyncMeasurer:
    def __init__(self, output_file: str, device: int | None = None,
                 snap_threshold: float = SNAP_THRESHOLD_US):
        self.output_file = Path(output_file)
        self.device = device
        self.snap_threshold = snap_threshold
        self.running = False
        self.start_time = None

        # Ring buffer for continuous capture (holds 750ms of audio)
        self.buffer = np.zeros((WINDOW_SAMPLES, 2), dtype=np.float32)
        self.buffer_write_pos = 0

        # Stats tracking
        self.previous_offset: float | None = None
        self.recent_offsets: list[float] = []
        self.recent_deltas: list[float] = []  # Non-snap deltas only (for jitter calc)
        self.all_offsets: list[float] = []  # All offsets (for bounded range)
        self.max_recent = 100  # Keep last 100 measurements for rolling stats

        # Snap tracking
        self.snap_count = 0
        self.snap_magnitudes: list[float] = []

        # Low correlation tracking
        self.skipped_low_correlation = 0

    def sinc_interpolation(self, correlation: np.ndarray, peak_idx: int,
                          upsample_factor: int = 100, window_size: int = 50) -> float:
        """
        Sub-sample interpolation using FFT-based sinc interpolation.

        Uses zero-padding in the frequency domain to upsample the correlation
        peak region, then finds the refined peak location.

        Args:
            correlation: The correlation array
            peak_idx: Integer index of the correlation peak
            upsample_factor: Factor by which to upsample (default: 100x)
            window_size: Number of samples on each side of peak to use (default: 50)

        Returns:
            The interpolated peak position offset relative to peak_idx (in original samples)
        """
        # Extract window around the peak
        start = max(0, peak_idx - window_size)
        end = min(len(correlation), peak_idx + window_size + 1)
        window = correlation[start:end]

        if len(window) < 3:
            return 0.0

        # FFT-based upsampling via zero-padding in frequency domain
        n = len(window)
        spectrum = np.fft.fft(window)

        # Zero-pad in frequency domain
        # For real signals, we need to handle the spectrum symmetry carefully
        n_upsampled = n * upsample_factor
        spectrum_padded = np.zeros(n_upsampled, dtype=complex)

        # Copy positive frequencies
        n_pos = (n + 1) // 2
        spectrum_padded[:n_pos] = spectrum[:n_pos]

        # Copy negative frequencies
        if n % 2 == 0:
            spectrum_padded[-n//2:] = spectrum[-n//2:]
        else:
            spectrum_padded[-(n//2):] = spectrum[-(n//2):]

        # Inverse FFT to get upsampled signal (scale by upsample factor)
        upsampled = np.fft.ifft(spectrum_padded).real * upsample_factor

        # Find peak in upsampled data
        upsampled_peak_idx = np.argmax(upsampled)

        # Convert back to original sample coordinates
        # The peak in the original window was at (peak_idx - start)
        original_peak_in_window = peak_idx - start

        # Offset in upsampled coordinates
        upsampled_offset = upsampled_peak_idx - (original_peak_in_window * upsample_factor)

        # Convert to original sample coordinates
        offset = upsampled_offset / upsample_factor

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

        # Sub-sample interpolation using FFT-based sinc interpolation
        sub_sample_offset = self.sinc_interpolation(correlation_window, peak_idx_in_window)

        # Calculate lag in samples (relative to center of window)
        lag_samples = (peak_idx_in_window - MAX_LAG_SAMPLES) + sub_sample_offset

        # Convert to microseconds (negate for our sign convention)
        offset_us = -lag_samples * US_PER_SAMPLE

        return offset_us, correlation_strength

    def audio_callback(self, indata: np.ndarray, frames: int,
                       time_info: dict, status: sd.CallbackFlags):
        """Called by sounddevice for each audio block."""
        if status:
            print(f"Audio status: {status}", file=sys.stderr)

        # Write to ring buffer
        for i in range(frames):
            self.buffer[self.buffer_write_pos] = indata[i]
            self.buffer_write_pos = (self.buffer_write_pos + 1) % WINDOW_SAMPLES

    def get_buffer_snapshot(self) -> np.ndarray:
        """Get a linearized copy of the ring buffer (oldest to newest)."""
        pos = self.buffer_write_pos
        # Roll so that oldest sample is at index 0
        return np.roll(self.buffer, -pos, axis=0).copy()

    def write_header(self):
        """Write CSV header to output file."""
        with open(self.output_file, 'w') as f:
            f.write("timestamp_unix,offset_us,delta_us,correlation_peak,is_snap\n")

    def append_measurement(self, timestamp: float, offset_us: float,
                           delta_us: float, correlation: float, is_snap: bool):
        """Append a measurement to the CSV file."""
        with open(self.output_file, 'a') as f:
            f.write(f"{timestamp:.6f},{offset_us:.3f},{delta_us:.3f},"
                    f"{correlation:.6f},{1 if is_snap else 0}\n")

    def process_measurement(self, offset_us: float) -> tuple[float, bool]:
        """
        Process a new offset measurement.

        Returns:
            delta_us: Change from previous measurement
            is_snap: Whether this is detected as a correction snap
        """
        # Calculate delta
        if self.previous_offset is None:
            delta_us = 0.0
            is_snap = False
        else:
            delta_us = offset_us - self.previous_offset
            is_snap = abs(delta_us) > self.snap_threshold

        # Update tracking
        self.previous_offset = offset_us
        self.all_offsets.append(offset_us)
        self.recent_offsets.append(offset_us)
        if len(self.recent_offsets) > self.max_recent:
            self.recent_offsets.pop(0)

        # Track snaps separately
        if is_snap:
            self.snap_count += 1
            self.snap_magnitudes.append(delta_us)
        else:
            # Only non-snap deltas contribute to jitter calculation
            self.recent_deltas.append(delta_us)
            if len(self.recent_deltas) > self.max_recent:
                self.recent_deltas.pop(0)

        return delta_us, is_snap

    def update_live_display(self, offset_us: float, delta_us: float,
                            correlation: float, is_snap: bool, elapsed: float):
        """Update the live terminal display."""
        # Calculate drift stability (std dev of non-snap deltas)
        if len(self.recent_deltas) >= 2:
            drift_stability = np.std(self.recent_deltas)
        else:
            drift_stability = 0.0

        # Calculate offset jitter (std dev of recent offsets - classical jitter)
        if len(self.recent_offsets) >= 2:
            offset_jitter = np.std(self.recent_offsets)
        else:
            offset_jitter = 0.0

        # Bounded range (all-time min/max)
        if self.all_offsets:
            bounded_min = np.min(self.all_offsets)
            bounded_max = np.max(self.all_offsets)
            bounded_range = bounded_max - bounded_min
        else:
            bounded_min = bounded_max = offset_us
            bounded_range = 0.0

        # Format elapsed time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        print(f"\r[{hours:02d}:{minutes:02d}:{seconds:02d}] "
              f"Offset: {offset_us:+8.1f}μs | "
              f"Jitter: {offset_jitter:5.1f}μs | "
              f"Drift: {drift_stability:5.2f}μs | "
              f"Range: [{bounded_min:+.1f}, {bounded_max:+.1f}]μs | "
              f"Corr: {correlation:.2f}  ",
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
        print(f"  Process interval: {PROCESS_INTERVAL_MS} ms")
        print(f"  Max lag search: ±{MAX_LAG_MS} ms (±{MAX_LAG_SAMPLES} samples)")
        print(f"  Resolution: ~{US_PER_SAMPLE:.2f} μs/sample (100x FFT-based sinc interpolation)")
        print(f"  Snap threshold: {self.snap_threshold} μs")
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
            # Use smaller blocksize for smoother ring buffer filling
            with sd.InputStream(device=self.device,
                               channels=2,
                               samplerate=SAMPLE_RATE,
                               blocksize=PROCESS_INTERVAL_SAMPLES,
                               dtype=np.float32,
                               callback=self.audio_callback):

                # Wait for initial buffer fill (750ms)
                sd.sleep(int(WINDOW_DURATION_MS))

                while self.running:
                    # Get snapshot of the 750ms ring buffer
                    snapshot = self.get_buffer_snapshot()
                    left = snapshot[:, 0]
                    right = snapshot[:, 1]

                    # Measure offset
                    offset_us, correlation = self.measure_offset(left, right)

                    # Record timestamp
                    timestamp = time.time()
                    elapsed = timestamp - self.start_time

                    # Skip low correlation measurements (unreliable)
                    if correlation < MIN_CORRELATION:
                        self.skipped_low_correlation += 1
                        # Still log to file for transparency, but don't include in stats
                        self.append_measurement(timestamp, offset_us, 0.0,
                                                correlation, False)
                        sd.sleep(int(PROCESS_INTERVAL_MS))
                        continue

                    # Process and detect snaps
                    delta_us, is_snap = self.process_measurement(offset_us)

                    # Log to file
                    self.append_measurement(timestamp, offset_us, delta_us,
                                            correlation, is_snap)
                    measurement_count += 1

                    # Update display
                    self.update_live_display(offset_us, delta_us, correlation,
                                             is_snap, elapsed)

                    # Check duration limit
                    if duration and elapsed >= duration:
                        self.running = False

                    # Wait for next processing interval
                    sd.sleep(int(PROCESS_INTERVAL_MS))

        except KeyboardInterrupt:
            pass

        # Print final summary
        self.print_summary(measurement_count)

    def print_summary(self, measurement_count: int):
        """Print final measurement summary."""
        print(f"\n\n{'='*60}")
        print("Measurement Summary")
        print('='*60)
        print(f"  Total measurements: {measurement_count}")
        if self.skipped_low_correlation > 0:
            print(f"  Skipped (low correlation): {self.skipped_low_correlation}")
        print(f"  Total duration: {time.time() - self.start_time:.1f} seconds")
        print(f"  Output saved to: {self.output_file}")

        if self.all_offsets:
            abs_offsets = np.abs(self.all_offsets)
            print(f"\nOffset Statistics (absolute):")
            print(f"  Mean: {np.mean(abs_offsets):.2f} μs")
            print(f"  Std Dev: {np.std(abs_offsets):.2f} μs")
            print(f"  Median: {np.median(abs_offsets):.2f} μs")
            print(f"  95th percentile: {np.percentile(abs_offsets, 95):.2f} μs")
            print(f"  Max: {np.max(abs_offsets):.2f} μs")
            print(f"  Bounded Range: {np.max(self.all_offsets) - np.min(self.all_offsets):.2f} μs")

        if self.recent_deltas:
            print(f"\nDrift Stability (non-snap deltas):")
            print(f"  Std Dev: {np.std(self.recent_deltas):.2f} μs")
            print(f"  Mean |delta|: {np.mean(np.abs(self.recent_deltas)):.2f} μs")

        if self.recent_offsets:
            print(f"\nOffset Jitter (rolling window):")
            print(f"  Std Dev: {np.std(self.recent_offsets):.2f} μs")

        print('='*60)

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
    parser.add_argument('--snap-threshold', '-s', type=float, default=SNAP_THRESHOLD_US,
                        help=f'Snap detection threshold in μs (default: {SNAP_THRESHOLD_US})')
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
    measurer = SyncMeasurer(args.output, args.device, args.snap_threshold)

    def signal_handler(signum, frame):
        print("\nStopping...")
        measurer.stop()

    signal.signal(signal.SIGINT, signal_handler)

    # Run measurement
    measurer.run(args.duration)


if __name__ == "__main__":
    main()
