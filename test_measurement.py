#!/usr/bin/env python3
"""
Test the sync measurement logic with synthetic signals.
Verifies that offset measurement and sub-sample interpolation work correctly.
"""

import numpy as np
from scipy import signal as sig
from measure_sync import SyncMeasurer, SAMPLE_RATE, US_PER_SAMPLE


def create_test_chirp(duration_ms: float = 50, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Create a test chirp signal."""
    num_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)
    return sig.chirp(t, f0=100, f1=10000, t1=t[-1], method='linear').astype(np.float32)


def create_test_window(chirp: np.ndarray, offset_samples: float,
                       window_samples: int = 48000) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a test window with left and right channels,
    where right is delayed by offset_samples relative to left.

    Positive offset = left is ahead of right.
    """
    # Create window with chirp in the middle
    left = np.zeros(window_samples, dtype=np.float32)
    right = np.zeros(window_samples, dtype=np.float32)

    # Place chirp at 1/4 into the window
    chirp_start = window_samples // 4

    # Left channel: chirp at chirp_start
    left[chirp_start:chirp_start + len(chirp)] = chirp

    # Right channel: chirp delayed by offset_samples
    # For sub-sample offsets, we use linear interpolation to shift
    int_offset = int(np.floor(offset_samples))
    frac_offset = offset_samples - int_offset

    right_start = chirp_start + int_offset

    if frac_offset == 0:
        right[right_start:right_start + len(chirp)] = chirp
    else:
        # Interpolate for fractional sample offset
        # Simple linear interpolation
        for i in range(len(chirp)):
            idx = right_start + i
            if 0 <= idx < window_samples - 1:
                right[idx] += chirp[i] * (1 - frac_offset)
                right[idx + 1] += chirp[i] * frac_offset

    return left, right


def test_offset_measurement():
    """Test offset measurement accuracy at various offsets."""
    measurer = SyncMeasurer("test_output.csv", device=None)
    chirp = create_test_chirp()

    print("Testing offset measurement accuracy")
    print("=" * 70)
    print(f"{'True Offset (μs)':>20} | {'Measured (μs)':>15} | {'Error (μs)':>12} | {'Corr':>8}")
    print("-" * 70)

    # Test various offsets in microseconds
    test_offsets_us = [0, 10, 25, 50, 100, -50, -100, 500, -500, 1000, -1000]

    max_error = 0
    for true_offset_us in test_offsets_us:
        # Convert to samples
        true_offset_samples = true_offset_us / US_PER_SAMPLE

        left, right = create_test_window(chirp, true_offset_samples)

        measured_offset_us, correlation = measurer.measure_offset(left, right)

        error_us = measured_offset_us - true_offset_us
        max_error = max(max_error, abs(error_us))

        status = "✓" if abs(error_us) < 2 else "✗"  # Allow 2μs error

        print(f"{true_offset_us:>20.2f} | {measured_offset_us:>15.2f} | {error_us:>12.3f} | {correlation:>8.4f} {status}")

    print("-" * 70)
    print(f"Maximum error: {max_error:.3f} μs")

    # Test sub-sample accuracy more thoroughly
    print("\n\nSub-sample interpolation test (offsets from 0-100 μs in 5μs steps)")
    print("=" * 70)

    errors = []
    for true_offset_us in np.arange(0, 105, 5):
        true_offset_samples = true_offset_us / US_PER_SAMPLE
        left, right = create_test_window(chirp, true_offset_samples)
        measured_offset_us, _ = measurer.measure_offset(left, right)
        error = measured_offset_us - true_offset_us
        errors.append(abs(error))

    print(f"Mean absolute error: {np.mean(errors):.3f} μs")
    print(f"Max absolute error: {np.max(errors):.3f} μs")
    print(f"Std of errors: {np.std(errors):.3f} μs")

    if np.max(errors) < 3:
        print("\n✓ Sub-sample interpolation working well!")
    else:
        print("\n✗ Sub-sample interpolation needs improvement")

    return max_error < 5  # Pass if max error < 5μs


def test_noise_robustness():
    """Test measurement accuracy with added noise."""
    measurer = SyncMeasurer("test_output.csv", device=None)
    chirp = create_test_chirp()

    print("\n\nNoise robustness test")
    print("=" * 70)
    print(f"{'SNR (dB)':>10} | {'True (μs)':>12} | {'Measured (μs)':>15} | {'Error (μs)':>12} | {'Corr':>8}")
    print("-" * 70)

    true_offset_us = 50
    true_offset_samples = true_offset_us / US_PER_SAMPLE

    for snr_db in [40, 30, 20, 10, 6]:
        left, right = create_test_window(chirp, true_offset_samples)

        # Add noise
        signal_power = np.mean(chirp ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise_std = np.sqrt(noise_power)

        left += np.random.randn(len(left)).astype(np.float32) * noise_std
        right += np.random.randn(len(right)).astype(np.float32) * noise_std

        measured_offset_us, correlation = measurer.measure_offset(left, right)
        error_us = measured_offset_us - true_offset_us

        status = "✓" if abs(error_us) < 5 else "✗"
        print(f"{snr_db:>10} | {true_offset_us:>12.2f} | {measured_offset_us:>15.2f} | {error_us:>12.3f} | {correlation:>8.4f} {status}")

    print("-" * 70)


def test_performance():
    """Test processing speed."""
    import time

    measurer = SyncMeasurer("test_output.csv", device=None)
    chirp = create_test_chirp()
    left, right = create_test_window(chirp, 5)

    print("\n\nPerformance test")
    print("=" * 70)

    # Warm up
    for _ in range(10):
        measurer.measure_offset(left, right)

    # Time it
    iterations = 1000
    start = time.perf_counter()
    for _ in range(iterations):
        measurer.measure_offset(left, right)
    elapsed = time.perf_counter() - start

    time_per_measurement_ms = elapsed / iterations * 1000
    measurements_per_second = iterations / elapsed

    print(f"Time per measurement: {time_per_measurement_ms:.3f} ms")
    print(f"Measurements per second: {measurements_per_second:.1f}")
    print(f"Window duration: 500 ms")

    if time_per_measurement_ms < 100:  # Should process in << 500ms
        print(f"\n✓ Fast enough for real-time (need < 500ms, got {time_per_measurement_ms:.1f}ms)")
    else:
        print(f"\n✗ Too slow for real-time!")


if __name__ == "__main__":
    test_offset_measurement()
    test_noise_robustness()
    test_performance()

    # Clean up test file
    import os
    if os.path.exists("test_output.csv"):
        os.remove("test_output.csv")
