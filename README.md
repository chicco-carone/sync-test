# Sync Test Tools

Tools for measuring audio synchronization accuracy between two devices playing the same content.

## What This Does

Measures the time offset between two audio channels captured from a USB audio interface. Used to test how accurately two clients synchronize audio.

## Usage

```bash
# Install dependencies
uv sync

# Generate 1-hour test chirp file (48kHz, chirps every 500ms)
uv run sync-generate-chirp

# List audio devices
uv run sync-measure --list

# Run measurement (captures at 96kHz for ~10μs resolution)
uv run sync-measure --device <ID>

# Analyze results
uv run sync-analyze <csv_file> --plot
```

## Key Files

- `src/sync_test/measure_sync.py` - Real-time capture and cross-correlation
- `src/sync_test/analyze.py` - Post-run statistics and plotting
- `src/sync_test/generate_chirp.py` - Test signal generator

## How It Works

1. Both devices play the same chirp pattern
2. Left channel of each device feeds into USB capture card
3. Cross-correlation finds time offset between channels
4. FFT-based sinc interpolation (100x upsampling) gives sub-sample accuracy (~0.1μs)
5. Logs offset, delta, and correlation to CSV
