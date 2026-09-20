# Snapcast Player Offset Calibrator

Measure a Snapcast target player against a known-good reference player and obtain a
practical latency adjustment for the target.

## What This Does

The local PC runs the known-good Snapclient. Music Assistant sends the supplied
chirp track to the Snapcast group. Capture the local Snapclient monitor as the
reference and the target player with a USB microphone or line feed. The tool
cross-correlates both captures and reports (or applies) a target latency change.

> A direct line-level feed is most accurate. When using a microphone, keep it
> fixed and account for the speaker-to-microphone distance, which is included in
> the measured delay.

## Usage

```bash
# Install dependencies
uv sync

# List audio devices
uv run sync-measure --list

# Play sync_test_chirps_18khz.flac from Music Assistant to the Snapcast group.
# Capture the local Snapclient monitor and target microphone separately.
uv run sync-measure \
  --reference-device <local-snapclient-monitor> \
  --target-device <target-microphone> \
  --sample-rate 48000 --min-correlation 0.1 --duration 60

# Generate a fresh 1-hour test chirp file (48kHz, chirps every 500ms)
uv run sync-generate-chirp

# Analyze results
uv run sync-analyze <csv_file> --plot
```

## Applying the result

At the end of a run, the program uses the median valid measurement and prints a
target-latency adjustment. For example, `Median target offset: +2400 us` means
the target is 2.4 ms behind the local reference: **increase the target's
Snapcast latency by 2.4 ms**. A negative value means decrease it. This direction
matches Snapcast: a higher configured latency advances that client's playback.

The sign convention is fixed: a positive offset means the reference capture
(channel 1) arrives before the target capture (channel 2). If your interface is
wired in the opposite order, use `--reference-channel 2 --target-channel 1`.
The default search range is ±100 ms, suitable for an initial calibration. Once
the player is close, `--max-lag-ms 5` rejects unrelated peaks more aggressively.

## PipeWire or ALSA local reference capture

The reference does not need a physical second microphone. Select the local
Snapclient monitor and target microphone directly with `--reference-device`
and `--target-device`; this avoids creating an aggregate source. For PipeWire,
these devices are scheduled in PipeWire's graph; the tool discards its initial
three-second buffer warm-up before measuring:

```bash
uv run sync-measure --reference-device 'Snapclient/Snapcast Audio Stream' \
  --target-device 'USB Audio Microphone' --sample-rate 48000 --duration 60
```

For hardware devices with independent clocks, measure a short run and watch
the reported drift; use a single stereo USB interface instead if it is large.
Keep the target mic fixed while tuning, and account for its acoustic distance
from the target speaker (a direct target line feed is more accurate).
Acoustic capture can produce lower waveform correlation than direct line feeds;
start with `--min-correlation 0.1`, verify that the offsets are stable, then
use the same setting for an automatic update.

## Automatic Snapserver update and verification

The tool uses Snapserver's built-in JSON-RPC API (TCP port 1705), so no
third-party Python Snapcast package is required. Find the exact target client
ID in Snapweb or with Snapserver's `Server.GetStatus`, then run:

```bash
uv run sync-measure \
  --reference-device <local-snapclient-monitor> \
  --target-device <target-microphone> \
  --sample-rate 48000 --min-correlation 0.1 --duration 60 \
  --snapserver snapserver.local --target-client '<client-id>' \
  --apply --resync --retest-duration 30
```

`--apply` reads the target's current latency, applies the signed median
correction, and writes the new integer-millisecond latency through
`Client.SetLatency`. `--resync` temporarily removes and restores the target's
group membership; Snapserver does not expose a dedicated force-resync RPC, and
this API-only operation makes it receive its group/stream configuration again.
The verification capture starts after a two-second settle period and is written
as `sync_verification_*.csv`.

## Key Files

- `src/sync_test/measure_sync.py` - Real-time capture and cross-correlation
- `src/sync_test/analyze.py` - Post-run statistics and plotting
- `src/sync_test/generate_chirp.py` - Test signal generator

## How It Works

1. Both players play the same chirp pattern through the Snapcast group.
2. The USB interface records the reference and target separately.
3. Cross-correlation finds their signed time offset.
4. FFT-based sinc interpolation refines the peak below one sample.
5. The median valid offset becomes the target latency recommendation.
