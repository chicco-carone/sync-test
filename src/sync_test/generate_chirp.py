#!/usr/bin/env python3
"""
Generate a 1-hour FLAC file with periodic chirps for sync testing.

Chirp parameters:
- 50ms linear chirp from 100Hz to 10kHz
- 5ms fade in/out to avoid clicks
- Repeated every 500ms (so 450ms silence between chirps)
- 48kHz sample rate, mono
- 1 hour duration
"""

import numpy as np
import soundfile as sf
from scipy.signal import chirp

# Parameters
SAMPLE_RATE = 48000
DURATION_HOURS = 1
DURATION_SECONDS = DURATION_HOURS * 3600

CHIRP_DURATION_MS = 50
CHIRP_PERIOD_MS = 500
FADE_MS = 5

CHIRP_START_FREQ = 100  # Hz
CHIRP_END_FREQ = 18000  # Hz

OUTPUT_FILE = "sync_test_chirps.flac"


def generate_single_chirp(sample_rate: int, duration_ms: int, fade_ms: int,
                          start_freq: int, end_freq: int) -> np.ndarray:
    """Generate a single chirp with fade in/out."""
    num_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)

    # Generate linear chirp
    signal = chirp(t, f0=start_freq, f1=end_freq, t1=t[-1], method='linear')

    # Apply fade in/out
    fade_samples = int(sample_rate * fade_ms / 1000)
    if fade_samples > 0:
        # Fade in (raised cosine)
        fade_in = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_samples)))
        signal[:fade_samples] *= fade_in

        # Fade out (raised cosine)
        fade_out = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_samples)))
        signal[-fade_samples:] *= fade_out

    return signal.astype(np.float32)


def generate_chirp_period(sample_rate: int, period_ms: int, chirp_signal: np.ndarray) -> np.ndarray:
    """Generate one period: chirp followed by silence."""
    period_samples = int(sample_rate * period_ms / 1000)
    period = np.zeros(period_samples, dtype=np.float32)
    period[:len(chirp_signal)] = chirp_signal
    return period


def main():
    print(f"Generating {DURATION_HOURS}-hour sync test chirp file...")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  Chirp: {CHIRP_DURATION_MS}ms, {CHIRP_START_FREQ}Hz -> {CHIRP_END_FREQ}Hz")
    print(f"  Period: {CHIRP_PERIOD_MS}ms (chirp every {CHIRP_PERIOD_MS}ms)")
    print(f"  Fade: {FADE_MS}ms in/out")

    # Generate single chirp
    single_chirp = generate_single_chirp(
        SAMPLE_RATE, CHIRP_DURATION_MS, FADE_MS, CHIRP_START_FREQ, CHIRP_END_FREQ
    )
    print(f"  Single chirp: {len(single_chirp)} samples")

    # Generate one period
    one_period = generate_chirp_period(SAMPLE_RATE, CHIRP_PERIOD_MS, single_chirp)
    print(f"  One period: {len(one_period)} samples")

    # Calculate total periods needed
    total_samples = SAMPLE_RATE * DURATION_SECONDS
    num_periods = int(np.ceil(total_samples / len(one_period)))
    print(f"  Total periods: {num_periods}")
    print(f"  Total duration: {num_periods * len(one_period) / SAMPLE_RATE:.2f} seconds")

    # Generate full audio by tiling
    # We'll generate in chunks to avoid memory issues
    print(f"\nWriting to {OUTPUT_FILE}...")

    periods_per_chunk = 1000  # ~500 seconds per chunk
    samples_per_chunk = periods_per_chunk * len(one_period)

    with sf.SoundFile(OUTPUT_FILE, mode='w', samplerate=SAMPLE_RATE,
                      channels=1, format='FLAC') as f:
        periods_written = 0
        while periods_written < num_periods:
            periods_this_chunk = min(periods_per_chunk, num_periods - periods_written)
            chunk = np.tile(one_period, periods_this_chunk)
            f.write(chunk)
            periods_written += periods_this_chunk
            progress = periods_written / num_periods * 100
            print(f"  Progress: {progress:.1f}% ({periods_written}/{num_periods} periods)", end='\r')

    print(f"\n\nDone! Created {OUTPUT_FILE}")

    # Verify file
    info = sf.info(OUTPUT_FILE)
    print(f"\nFile info:")
    print(f"  Duration: {info.duration:.2f} seconds ({info.duration/3600:.2f} hours)")
    print(f"  Sample rate: {info.samplerate} Hz")
    print(f"  Channels: {info.channels}")
    print(f"  Format: {info.format}")


if __name__ == "__main__":
    main()
