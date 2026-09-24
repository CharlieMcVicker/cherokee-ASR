#!/usr/bin/env python3
import os
import sys
import csv
import time
from multiprocessing import Pool, cpu_count
from pydub import AudioSegment
from pydub.effects import normalize


def process_interview(args):
    mp3_filename, segments_list, mp3_dir, out_base_dir = args
    t_start = time.time()
    mp3_path = os.path.join(mp3_dir, mp3_filename)

    if not os.path.exists(mp3_path):
        print(f"Error: {mp3_path} does not exist. Skipping.")
        return mp3_filename, 0, 0.0

    print(f"[{mp3_filename}] Loading audio (converting to 16kHz mono)...")
    try:
        audio = AudioSegment.from_file(mp3_path).set_frame_rate(16000).set_channels(1)
    except Exception as e:
        print(f"Error loading {mp3_path}: {e}")
        return mp3_filename, 0, 0.0

    duration_s = len(audio) / 1000.0
    print(
        f"[{mp3_filename}] Loaded. Duration: {duration_s:.1f}s. Extracting {len(segments_list)} segments..."
    )

    success_count = 0
    for seg in segments_list:
        start_ms = seg["start_ms"]
        end_ms = seg["end_ms"]
        segmented_filepath = seg["segmented_filepath"]

        # Determine the target output path under out_base_dir
        # segmented_filepath is like: ../data/processed/denoised_segments/Thomas Still/Thomas Still_segment_0701_2214649_2216779.wav
        # We want to place it under out_base_dir (e.g. data/processed/noisy_segments)
        # We can extract the relative part after 'denoised_segments/' or rebuild it.
        parts = segmented_filepath.split("denoised_segments/")
        if len(parts) > 1:
            rel_path = parts[1]
        else:
            # Fallback in case path doesn't contain denoised_segments/
            rel_path = os.path.join(
                os.path.basename(os.path.dirname(segmented_filepath)),
                os.path.basename(segmented_filepath),
            )

        out_filepath = os.path.join(out_base_dir, rel_path)
        os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

        try:
            # Extract segment and normalize volume
            chunk = audio[start_ms:end_ms]
            chunk = normalize(chunk)
            chunk.export(out_filepath, format="wav")
            success_count += 1
        except Exception as e:
            print(f"[{mp3_filename}] Failed to export segment {start_ms}-{end_ms}: {e}")

    elapsed = time.time() - t_start
    print(
        f"[{mp3_filename}] Completed {success_count}/{len(segments_list)} segments in {elapsed:.1f}s."
    )
    return mp3_filename, success_count, elapsed


def main():
    mp3_dir = "data/projects/cvcs"
    manifest_path = "data/projects/audiofiles-to-transcribe/segmentation_manifest.csv"
    out_base_dir = "data/processed/noisy_segments"

    if not os.path.exists(mp3_dir):
        print(f"Error: {mp3_dir} directory not found.")
        sys.exit(1)
    if not os.path.exists(manifest_path):
        print(f"Error: {manifest_path} not found.")
        sys.exit(1)

    print("Parsing manifest CSV...")
    # Group segments by original_filename
    interviews = {}
    with open(manifest_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            orig_fn = row["original_filename"]
            # Only process if the original filename is an MP3 that exists in mp3_dir
            if orig_fn.lower().endswith(".mp3"):
                # Handle case mismatch if any
                actual_fn = orig_fn
                if not os.path.exists(os.path.join(mp3_dir, actual_fn)):
                    # Check lowercase/case insensitive match
                    for f_in_dir in os.listdir(mp3_dir):
                        if f_in_dir.lower() == orig_fn.lower():
                            actual_fn = f_in_dir
                            break

                if actual_fn not in interviews:
                    interviews[actual_fn] = []

                interviews[actual_fn].append(
                    {
                        "segmented_filepath": row["segmented_filepath"],
                        "start_ms": int(row["start_ms"]),
                        "end_ms": int(row["end_ms"]),
                    }
                )

    print(f"Found {len(interviews)} interviews in manifest with segments.")

    # Prepare arguments for multiprocessing
    tasks = []
    total_segments = 0
    for mp3_filename, segs in interviews.items():
        tasks.append((mp3_filename, segs, mp3_dir, out_base_dir))
        total_segments += len(segs)

    print(f"Total segments to extract: {total_segments}")
    print(f"Starting multiprocessing pool with {cpu_count()} workers...")

    t_start = time.time()
    total_success = 0

    # We use a Pool to process interviews in parallel
    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        results = pool.map(process_interview, tasks)

    for mp3_filename, success_count, elapsed in results:
        total_success += success_count

    total_elapsed = time.time() - t_start
    print(
        f"\nAll done! Extracted {total_success}/{total_segments} segments successfully in {total_elapsed:.1f} seconds."
    )


if __name__ == "__main__":
    main()
