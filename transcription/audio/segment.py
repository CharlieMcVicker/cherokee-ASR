#!/usr/bin/env python3
import sys
import os
import argparse
import time
import numpy as np
from pydub import AudioSegment


def get_energy_profile(audio, step_ms=10):
    """
    Computes the dBFS energy profile for the audio segment in step_ms increments.
    Returns a numpy array of dBFS values for each step (vectorized for maximum performance).
    """
    num_steps = len(audio) // step_ms
    if num_steps <= 0:
        return np.array([], dtype=np.float32)

    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels)).mean(axis=1)

    samples_per_step = int((audio.frame_rate * step_ms) / 1000.0)
    if samples_per_step <= 0:
        samples_per_step = 1

    total_needed = num_steps * samples_per_step
    if len(samples) < total_needed:
        samples = np.pad(samples, (0, total_needed - len(samples)))
    else:
        samples = samples[:total_needed]

    windows = samples.reshape((num_steps, samples_per_step))
    rms = np.sqrt(np.mean(windows**2, axis=1))

    max_amp = (
        float(audio.max_possible_amplitude)
        if audio.max_possible_amplitude > 0
        else 32768.0
    )
    rms = np.maximum(rms, 1e-9)
    dbfs = 20.0 * np.log10(rms / max_amp)
    dbfs = np.clip(dbfs, -120.0, 0.0)

    return dbfs.astype(np.float32)


def segment_audio_from_profile(
    dbfs_profile,
    total_duration_ms,
    step_ms=10,
    min_silence_len=500,
    silence_thresh=-40,
    keep_silence=100,
):
    """
    Finds nonsilent segments directly from the precomputed dBFS energy profile.
    """
    min_silence_steps = min_silence_len // step_ms
    keep_silence_steps = keep_silence // step_ms

    # Boolean mask: True if signal is silent (energy < threshold)
    # We handle negative infinity (absolute silence) safely
    is_silent = dbfs_profile < silence_thresh

    # We want to identify continuous runs of silence that are >= min_silence_steps.
    # To do this, we can label the silent/nonsilent segments.
    # An easy way is to run a status tracker:
    # Find contiguous silent blocks:
    silent_runs = []
    in_silent_run = False
    run_start = 0

    for i, silent in enumerate(is_silent):
        if silent:
            if not in_silent_run:
                in_silent_run = True
                run_start = i
        else:
            if in_silent_run:
                in_silent_run = False
                run_len = i - run_start
                if run_len >= min_silence_steps:
                    silent_runs.append((run_start, i))
    # Handle end of array
    if in_silent_run:
        run_len = len(is_silent) - run_start
        if run_len >= min_silence_steps:
            silent_runs.append((run_start, len(is_silent)))

    # Now invert the silent runs to find speaking segments.
    # We start with the entire range [0, len(is_silent)]
    nonsilent_ranges = []
    last_end = 0
    for start, end in silent_runs:
        if start > last_end:
            nonsilent_ranges.append((last_end, start))
        last_end = end
    if last_end < len(is_silent):
        nonsilent_ranges.append((last_end, len(is_silent)))

    # Map steps back to milliseconds and apply keep_silence padding without causing adjacent segment overlap
    segments = []
    num_ranges = len(nonsilent_ranges)
    for idx, (start_step, end_step) in enumerate(nonsilent_ranges):
        start_ms = start_step * step_ms
        end_ms = end_step * step_ms

        # Calculate start padding
        if idx == 0:
            pad_start = max(0, start_ms - keep_silence)
        else:
            prev_end_ms = nonsilent_ranges[idx - 1][1] * step_ms
            gap_before = max(0, start_ms - prev_end_ms)
            pad_start = start_ms - min(keep_silence, gap_before // 2)

        # Calculate end padding
        if idx == num_ranges - 1:
            pad_end = min(total_duration_ms, end_ms + keep_silence)
        else:
            next_start_ms = nonsilent_ranges[idx + 1][0] * step_ms
            gap_after = max(0, next_start_ms - end_ms)
            pad_end = end_ms + min(keep_silence, gap_after // 2)

        segments.append(
            {"start": pad_start, "end": pad_end, "duration": pad_end - pad_start}
        )

    return segments


def compute_metrics(segments, total_duration_ms):
    if not segments:
        return {
            "percent_segmented": 0.0,
            "avg_len": 0.0,
            "min_len": 0.0,
            "max_len": 0.0,
            "median_len": 0.0,
            "count": 0,
        }

    durations = [seg["duration"] for seg in segments]

    # Calculate non-overlapping segmented duration
    intervals = sorted([(seg["start"], seg["end"]) for seg in segments])
    union_duration = 0
    if intervals:
        curr_start, curr_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= curr_end:
                curr_end = max(curr_end, end)
            else:
                union_duration += curr_end - curr_start
                curr_start, curr_end = start, end
        union_duration += curr_end - curr_start
    else:
        union_duration = 0

    percent_segmented = (union_duration / total_duration_ms) * 100.0
    avg_len = np.mean(durations) / 1000.0  # seconds
    min_len = np.min(durations) / 1000.0  # seconds
    max_len = np.max(durations) / 1000.0  # seconds
    median_len = np.median(durations) / 1000.0  # seconds

    return {
        "percent_segmented": percent_segmented,
        "avg_len": avg_len,
        "min_len": min_len,
        "max_len": max_len,
        "median_len": median_len,
        "count": len(segments),
    }


def print_table(results):
    header = "| Thresh (dBFS) | Min Sil (ms) | Keep Sil (ms) | Seg Count | % Segmented | Avg Len (s) | Min Len (s) | Median Len (s) | Max Len (s) |"
    separator = "|---------------|--------------|---------------|-----------|-------------|-------------|-------------|----------------|-------------|"
    print(header)
    print(separator)
    for r in results:
        m = r["metrics"]
        print(
            f"| {r['silence_thresh']:<13} | {r['min_silence_len']:<12} | {r['keep_silence']:<13} | "
            f"{m['count']:<9} | {m['percent_segmented']:<11.2f}% | {m['avg_len']:<11.2f} | "
            f"{m['min_len']:<11.2f} | {m['median_len']:<14.2f} | {m['max_len']:<11.2f} |"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Segment audio files and evaluate hyperparameters."
    )
    parser.add_argument("audio_path", help="Path to input audio file")
    parser.add_argument(
        "--sweep", action="store_true", help="Perform a hyperparameter sweep"
    )
    parser.add_argument(
        "--thresh",
        type=int,
        default=-40,
        help="Silence threshold in dBFS (default: -40)",
    )
    parser.add_argument(
        "--min-silence",
        type=int,
        default=500,
        help="Minimum silence length in ms (default: 500)",
    )
    parser.add_argument(
        "--keep-silence",
        type=int,
        default=100,
        help="Keep silence padding in ms (default: 100)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"Error: file {args.audio_path} does not exist.")
        sys.exit(1)

    print(f"Loading and decoding audio: {args.audio_path}")
    t0 = time.time()
    audio = AudioSegment.from_file(args.audio_path)
    total_len = len(audio)
    print(f"Loaded {total_len / 1000.0:.2f}s of audio in {time.time() - t0:.2f}s")

    print("Computing energy profile (dBFS)...")
    t0 = time.time()
    dbfs_profile = get_energy_profile(audio, step_ms=10)
    print(f"Energy profile computed in {time.time() - t0:.2f}s")

    if args.sweep:
        thresholds = [-50, -45, -40, -35, -30]
        min_silence_lens = [300, 500, 1000]
        keep_silences = [100, 200]

        results = []
        t0 = time.time()
        for thresh in thresholds:
            for min_sil in min_silence_lens:
                for keep_sil in keep_silences:
                    segments = segment_audio_from_profile(
                        dbfs_profile,
                        total_len,
                        step_ms=10,
                        min_silence_len=min_sil,
                        silence_thresh=thresh,
                        keep_silence=keep_sil,
                    )
                    metrics = compute_metrics(segments, total_len)
                    results.append(
                        {
                            "silence_thresh": thresh,
                            "min_silence_len": min_sil,
                            "keep_silence": keep_sil,
                            "metrics": metrics,
                        }
                    )
        sweep_time = time.time() - t0
        print(
            f"\n### Hyperparameter Sweep Results (Completed {len(results)} configurations in {sweep_time:.4f}s)\n"
        )
        print_table(results)
    else:
        # Single run
        segments = segment_audio_from_profile(
            dbfs_profile,
            total_len,
            step_ms=10,
            min_silence_len=args.min_silence,
            silence_thresh=args.thresh,
            keep_silence=args.keep_silence,
        )
        metrics = compute_metrics(segments, total_len)
        results = [
            {
                "silence_thresh": args.thresh,
                "min_silence_len": args.min_silence,
                "keep_silence": args.keep_silence,
                "metrics": metrics,
            }
        ]
        print("\n### Segmentation Results\n")
        print_table(results)


def get_best_parameters(dbfs_profile, total_len_ms):
    """
    Finds the best segmentation parameters matching maximum segment duration <= 10s.
    """
    thresholds = [-55, -50, -45, -40, -35, -30, -25, -20]
    min_silence_lens = [100, 200, 300, 500, 800, 1000]
    keep_silences = [0, 50, 100, 150, 200]

    results = []
    for thresh in thresholds:
        for min_sil in min_silence_lens:
            for keep_sil in keep_silences:
                segments = segment_audio_from_profile(
                    dbfs_profile,
                    total_len_ms,
                    step_ms=10,
                    min_silence_len=min_sil,
                    silence_thresh=thresh,
                    keep_silence=keep_sil,
                )
                metrics = compute_metrics(segments, total_len_ms)

                # Overlap calculation
                sorted_segs = sorted(segments, key=lambda s: s["start"])
                overlap_ms = 0
                for i in range(len(sorted_segs) - 1):
                    cur_end = sorted_segs[i]["end"]
                    nxt_start = sorted_segs[i + 1]["start"]
                    if nxt_start < cur_end:
                        overlap_ms += (
                            min(cur_end, sorted_segs[i + 1]["end"]) - nxt_start
                        )

                overlap_percent = (
                    (overlap_ms / total_len_ms * 100.0) if total_len_ms > 0 else 0.0
                )

                results.append(
                    {
                        "silence_thresh": thresh,
                        "min_silence_len": min_sil,
                        "keep_silence": keep_sil,
                        "percent_segmented": metrics["percent_segmented"],
                        "overlap_percent": overlap_percent,
                        "max_len": metrics["max_len"],
                        "avg_len": metrics["avg_len"],
                        "count": metrics["count"],
                        "segments": segments,
                    }
                )

    # Filter for max_len <= 10.0 and at least 1 segment
    valid_results = [r for r in results if r["max_len"] <= 10.0 and r["count"] > 0]

    def score_config(r):
        net_coverage = r["percent_segmented"] - r["overlap_percent"]
        silence_penalty = (r["keep_silence"] / 100.0) * 0.5
        fragment_penalty = 5.0 if (r["avg_len"] < 1.0 and r["count"] > 5) else 0.0
        return net_coverage - silence_penalty - fragment_penalty

    if valid_results:
        best = max(valid_results, key=score_config)
    else:
        # Fallback: pick the one with minimum max_len among those with > 0 segments
        with_segments = [r for r in results if r["count"] > 0]
        if with_segments:
            best = min(with_segments, key=lambda x: x["max_len"])
        else:
            best = {
                "silence_thresh": -40,
                "min_silence_len": 500,
                "keep_silence": 100,
                "segments": [],
            }

    return best


def split_long_segments_smart(segments, audio, max_duration_ms=10000, overlap_ms=250):
    """
    Splits any segments longer than max_duration_ms by scanning for internal silence/pauses,
    or falling back to the quietest point within the segment to avoid cutting in the middle of words.
    Adds overlap_ms padding at boundaries where a fallback split occurred.
    """
    final_segs = []

    def process_segment(start_ms, end_ms):
        duration = end_ms - start_ms
        if duration <= max_duration_ms:
            final_segs.append({"start": start_ms, "end": end_ms, "duration": duration})
            return

        # Extract the long sub-audio
        sub_audio = audio[start_ms:end_ms]
        dbfs_profile = get_energy_profile(sub_audio, step_ms=10)

        # Grid parameters to find brief pauses/silences inside active speech
        thresholds = [-45, -40, -35, -30, -25, -20, -15]
        min_silence_lens = [500, 400, 300, 200, 100, 50]
        keep_silences = [100, 50, 0]

        for min_sil in min_silence_lens:
            for thresh in thresholds:
                for keep_sil in keep_silences:
                    sub_segs = segment_audio_from_profile(
                        dbfs_profile,
                        duration,
                        step_ms=10,
                        min_silence_len=min_sil,
                        silence_thresh=thresh,
                        keep_silence=keep_sil,
                    )
                    # Filter out empty or trivial splits that don't subdivide the duration
                    if not sub_segs or len(sub_segs) <= 1:
                        continue

                    max_sub_len = max(s["duration"] for s in sub_segs)
                    if max_sub_len <= max_duration_ms:
                        # Found a configuration that splits all parts to <= 10s!
                        for s in sub_segs:
                            process_segment(start_ms + s["start"], start_ms + s["end"])
                        return

        # Fallback: Find the quietest 100ms window in the middle 40% (30% to 70%) of the segment to split
        samples = np.array(sub_audio.get_array_of_samples(), dtype=np.float32)
        sr = sub_audio.frame_rate
        win_size = int(sr * 0.1)  # 100ms

        start_search = int(len(samples) * 0.3)
        end_search = int(len(samples) * 0.7)

        if end_search - start_search > win_size:
            min_energy = float("inf")
            best_split_idx = (start_search + end_search) // 2
            step = int(sr * 0.05)  # 50ms step

            for idx in range(start_search, end_search - win_size, step):
                win = samples[idx : idx + win_size]
                energy = np.mean(win**2)
                if energy < min_energy:
                    min_energy = energy
                    best_split_idx = idx + (win_size // 2)

            split_ms = int((best_split_idx / sr) * 1000)
        else:
            split_ms = duration // 2

        # Recursively process the two halves with overlap padding to prevent phrase/word clipping
        left_end = min(end_ms, start_ms + split_ms + overlap_ms)
        right_start = max(start_ms, start_ms + split_ms - overlap_ms)

        process_segment(start_ms, left_end)
        process_segment(right_start, end_ms)

    for seg in segments:
        process_segment(seg["start"], seg["end"])

    return final_segs


if __name__ == "__main__":
    main()
