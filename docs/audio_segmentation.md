# Audio Segmentation and Voice Activity Detection (VAD)

This guide documents the **Voice Activity Detection (VAD) audio segmentation engine** and **audio extraction pipeline** in `transcription.audio`.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Key Functions & Classes](#2-key-functions--classes)
3. [Segmentation Algorithm & Smart Splitting](#3-segmentation-algorithm--smart-splitting)
4. [CLI Tools](#4-cli-tools)
   - [Hyperparameter Sweep Evaluation](#hyperparameter-sweep-evaluation)
   - [Audio Segment Extraction](#audio-segment-extraction)
5. [Programmatic Python Usage](#5-programmatic-python-usage)

---

## 1. Overview

Long continuous Cherokee audio recordings (e.g., historical interviews, conversational workshops, multi-minute New Testament Bible chapters) must be segmented into shorter, coherent speech segments ($\le 10$ seconds) before feeding into:
- Acoustic model training (`transcription.training.train`)
- Character and word timestamping (`transcription.alignment`)
- Active learning and Praat TextGrid generation

The segmentation engine in [`transcription.audio.segment`](file:///Users/julietmcvicker/code/workshop-transcription/transcription/audio/segment.py) employs a **two-phase hybrid VAD algorithm**:
1. **Vectorized Energy Profiling:** Computes exact decibels relative to full scale (dBFS) energy curves over 10ms windows.
2. **Dynamic Parameter Selection & Smart Recursive Splitting:** Automatically selects optimal silence thresholds to maximize speech coverage without exceeding target duration limits, using quiet-window fallbacks to avoid cutting in the middle of words.

---

## 2. Key Functions & Classes

The primary module is [`transcription.audio.segment`](file:///Users/julietmcvicker/code/workshop-transcription/transcription/audio/segment.py).

### `AudioChunk`

A lightweight dataclass representing a discrete segmented audio chunk with global timestamp boundaries:

```python
@dataclass
class AudioChunk:
    chunk_index: int        # 0-indexed position in sequence
    audio: AudioSegment     # Sliced pydub AudioSegment (16kHz, normalized)
    start_sec: float        # Global start offset in seconds
    end_sec: float          # Global end offset in seconds

    @property
    def duration_sec(self) -> float:
        """Returns segment duration in seconds rounded to 3 decimal places."""
        return round(self.end_sec - self.start_sec, 3)
```

---

### `get_energy_profile(audio, step_ms=10)`

Computes a continuous dBFS energy profile for an `AudioSegment` in `step_ms` intervals (default: `10ms`).

```python
def get_energy_profile(audio: AudioSegment, step_ms: int = 10) -> NDArray[np.float32]:
    ...
```

- **Vectorized NumPy RMS:** Reshapes samples into $(N, \text{samples\_per\_step})$ windows and computes root-mean-square energy vectorized across all frames.
- **Stereo Downmixing:** Multi-channel audio is averaged across channels to mono.
- **dBFS Scaling:** Computes $20 \log_{10}(\text{RMS} / \text{max\_amp})$ and clips the range to $[-120.0, 0.0]$ dBFS.

---

### `segment_audio_from_profile(dbfs_profile, total_duration_ms, step_ms=10, min_silence_len=500, silence_thresh=-40, keep_silence=100)`

Extracts non-silent interval boundaries from a precomputed dBFS energy profile:
- Identifies contiguous silent runs where energy is strictly below `silence_thresh` for at least `min_silence_len` milliseconds.
- Inverts silent runs to determine active speech ranges.
- Adds `keep_silence` padding (in ms) to the start and end of segments while guaranteeing adjacent segments do not overlap.

---

### `get_best_parameters(dbfs_profile, total_len_ms)`

Evaluates an exhaustive 2D parameter grid across the energy profile to select optimal hyperparameters for an audio file:
- **Thresholds:** $-55, -50, -45, -40, -35, -30, -25, -20$ dBFS
- **Min Silence Lengths:** $100, 200, 300, 500, 800, 1000$ ms
- **Keep Silence Paddings:** $0, 50, 100, 150, 200$ ms

#### Scoring Heuristic

Configurations are filtered to ensure $\text{max\_segment\_length} \le 10.0\text{s}$ and ranked using:
$$\text{Score} = (\text{Net Coverage} - \text{Overlap \%}) - \text{Silence Penalty} - \text{Fragment Penalty}$$
where:
- $\text{Silence Penalty} = (\text{keep\_silence} / 100.0) \times 0.5$
- $\text{Fragment Penalty} = 5.0$ if $\text{avg\_len} < 1.0\text{s}$ and $\text{count} > 5$

---

### `split_long_segments_smart(segments, audio, max_duration_ms=10000, overlap_ms=250)`

Recursively subdivides any segment exceeding `max_duration_ms`:
1. **Internal Pause Scan:** Scans the sub-audio segment across a finer grid of silence thresholds ($-45$ to $-15$ dBFS, min silence $50$ to $500$ ms) to detect subtle internal breath pauses.
2. **Quiet Window Fallback:** If no clear pause exists, searches the middle 40% ($30\% \text{ to } 70\%$) of the waveform for the 100ms window with the lowest RMS energy.
3. **Boundary Overlap Padding:** Applies `overlap_ms` (default: 250ms) padding across the split point so boundary words are not abruptly clipped.

---

### `segment_long_audio(audio_or_path, max_duration_ms=10000, overlap_ms=250)`

The top-level orchestrator for audio chunking:
1. Loads audio (from file path or existing `AudioSegment`), applies peak volume normalization, and resamples to 16,000 Hz.
2. Computes the dBFS energy profile and runs [`get_best_parameters()`](file:///Users/julietmcvicker/code/workshop-transcription/transcription/audio/segment.py#L305-L385).
3. Applies [`split_long_segments_smart()`](file:///Users/julietmcvicker/code/workshop-transcription/transcription/audio/segment.py#L387-L472) to partition segments into $\le 10\text{s}$ slices.
4. Returns a list of [`AudioChunk`](file:///Users/julietmcvicker/code/workshop-transcription/transcription/audio/segment.py#L14-L23) objects with start and end timestamps.

---

## 3. Segmentation Algorithm & Smart Splitting

```
               +--------------------------------------+
               | Input Audio File (or AudioSegment)   |
               +-------------------+------------------+
                                   |
                                   v
               +--------------------------------------+
               | Peak Normalization & 16kHz Resample  |
               +-------------------+------------------+
                                   |
                                   v
               +--------------------------------------+
               | get_energy_profile (Vectorized dBFS) |
               +-------------------+------------------+
                                   |
                                   v
               +--------------------------------------+
               | get_best_parameters (Grid Search)    |
               +-------------------+------------------+
                                   |
                                   v
               +--------------------------------------+
               | Initial Segments (Threshold-based)   |
               +-------------------+------------------+
                                   |
                        Are any segments > 10s?
                        /                    \
                     Yes                      No
                     /                          \
                    v                            \
   +------------------------------------+         \
   | split_long_segments_smart          |          \
   | - Internal fine pause scan (50ms+) |          |
   | - Quietest 100ms window fallback   |          |
   | - 250ms overlap boundary padding   |          |
   +----------------+-------------------+          |
                    \                             /
                     \                           /
                      v                         v
               +--------------------------------------+
               | List[AudioChunk]                     |
               | (start_sec, end_sec, duration_sec)   |
               +--------------------------------------+
```

---

## 4. CLI Tools

### Hyperparameter Sweep Evaluation

Run [`transcription.audio.segment`](file:///Users/julietmcvicker/code/workshop-transcription/transcription/audio/segment.py) directly from the command line to evaluate VAD thresholds or run a hyperparameter sweep over a target audio file.

```bash
# Perform an exhaustive hyperparameter sweep across threshold combinations
python -m transcription.audio.segment path/to/interview.wav --sweep

# Evaluate segmentation with specific manual parameters
python -m transcription.audio.segment path/to/interview.wav \
    --thresh -35 \
    --min-silence 400 \
    --keep-silence 150
```

#### Sweep Table Output Example

```
### Hyperparameter Sweep Results (Completed 30 configurations in 0.0421s)

| Thresh (dBFS) | Min Sil (ms) | Keep Sil (ms) | Seg Count | % Segmented | Avg Len (s) | Min Len (s) | Median Len (s) | Max Len (s) |
|---------------|--------------|---------------|-----------|-------------|-------------|-------------|----------------|-------------|
| -50           | 300          | 100           | 42        | 88.50%      | 4.12        | 0.85        | 3.80           | 8.95        |
| -50           | 500          | 100           | 28        | 91.20%      | 6.45        | 1.20        | 5.90           | 12.40       |
| -40           | 500          | 100           | 35        | 84.10%      | 4.75        | 0.90        | 4.20           | 9.80        |
| -30           | 300          | 200           | 55        | 72.30%      | 2.60        | 0.45        | 2.10           | 6.10        |
```

---

### Audio Segment Extraction

The extraction script [`transcription.audio.extract`](file:///Users/julietmcvicker/code/workshop-transcription/transcription/audio/extract.py) slices audio files on silence boundaries, saves individual `.wav` files to an output directory, and generates a `manifest.json`.

```bash
python -m transcription.audio.extract path/to/recording.wav \
    --out-dir data/processed/segments \
    --thresh -30 \
    --min-silence 500 \
    --keep-silence 200
```

#### Generated Output Structure

```
data/processed/segments/
├── recording_segment_0000_0_4520.wav
├── recording_segment_0001_4900_9100.wav
├── recording_segment_0002_9500_14200.wav
└── manifest.json
```

#### `manifest.json` Format

```json
[
  {
    "id": 0,
    "filename": "recording_segment_0000_0_4520.wav",
    "filepath": "data/processed/segments/recording_segment_0000_0_4520.wav",
    "start_ms": 0,
    "end_ms": 4520,
    "duration_ms": 4520,
    "start_seconds": 0.0,
    "end_seconds": 4.52,
    "duration_seconds": 4.52
  },
  {
    "id": 1,
    "filename": "recording_segment_0001_4900_9100.wav",
    "filepath": "data/processed/segments/recording_segment_0001_4900_9100.wav",
    "start_ms": 4900,
    "end_ms": 9100,
    "duration_ms": 4200,
    "start_seconds": 4.9,
    "end_seconds": 9.1,
    "duration_seconds": 4.2
  }
]
```

---

## 5. Programmatic Python Usage

### Example 1: In-Memory Long Audio Segmentation

```python
from transcription.audio.segment import segment_long_audio

audio_path = "data/raw_audio/cherokee_interview_01.wav"

# Automatically normalize, resample, find optimal parameters, and split chunks <= 10s
chunks = segment_long_audio(
    audio_or_path=audio_path,
    max_duration_ms=10000,  # 10s max duration
    overlap_ms=250,         # 250ms boundary overlap on fallback splits
)

print(f"Total Chunks Created: {len(chunks)}")
for chunk in chunks[:5]:
    print(
        f"Chunk #{chunk.chunk_index:03d} | "
        f"Time: {chunk.start_sec:6.2f}s -> {chunk.end_sec:6.2f}s | "
        f"Duration: {chunk.duration_sec:5.2f}s | "
        f"Frames: {chunk.audio.frame_count()}"
    )
```

### Example 2: Custom Energy Profiling and Dynamic Thresholding

```python
from pydub import AudioSegment
from pydub.effects import normalize
from transcription.audio.segment import (
    get_energy_profile,
    get_best_parameters,
    segment_audio_from_profile,
    compute_metrics,
)

# 1. Load and normalize audio
audio = AudioSegment.from_file("data/raw_audio/chapter_01.mp3")
audio = normalize(audio).set_frame_rate(16000)

# 2. Compute 10ms frame dBFS energy profile
dbfs_profile = get_energy_profile(audio, step_ms=10)

# 3. Find optimal parameters
best_cfg = get_best_parameters(dbfs_profile, total_len_ms=len(audio))
print(f"Optimal Threshold: {best_cfg['silence_thresh']} dBFS")
print(f"Min Silence:       {best_cfg['min_silence_len']} ms")
print(f"Keep Silence:      {best_cfg['keep_silence']} ms")

# 4. Extract segments and inspect summary metrics
segments = segment_audio_from_profile(
    dbfs_profile,
    total_duration_ms=len(audio),
    step_ms=10,
    min_silence_len=best_cfg["min_silence_len"],
    silence_thresh=best_cfg["silence_thresh"],
    keep_silence=best_cfg["keep_silence"],
)
metrics = compute_metrics(segments, total_duration_ms=len(audio))
print(f"Speech Coverage:   {metrics['percent_segmented']:.2f}%")
print(f"Average Length:    {metrics['avg_len']:.2f}s")
print(f"Max Length:        {metrics['max_len']:.2f}s")
```

### Example 3: Slicing and Exporting Chunks Directly to Disk

```python
import os
from transcription.audio.segment import segment_long_audio

output_dir = "data/processed/exported_chunks"
os.makedirs(output_dir, exist_ok=True)

chunks = segment_long_audio("data/audio.wav")

for chunk in chunks:
    out_file = os.path.join(
        output_dir,
        f"chunk_{chunk.chunk_index:04d}_{int(chunk.start_sec*1000)}_{int(chunk.end_sec*1000)}.wav"
    )
    # Export chunk as 16kHz mono WAV
    chunk.audio.export(out_file, format="wav")
print(f"Exported {len(chunks)} chunks to '{output_dir}'.")
```
