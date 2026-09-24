#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
praat_export.py

Reads edited Praat .TextGrid files from 'data/processed/praat/', parses the 'human' tier,
extracts the approved audio segments from the 16kHz WAV files, saves them as WAVs,
and writes/updates the active learning dataset CSV manifest.
"""

import os
import csv
import re
import glob
import soundfile as sf

PRAAT_DIR = "data/processed/praat"
OUT_DIR = "data/processed/active_learning"
MANIFEST_PATH = os.path.join(OUT_DIR, "active_learning_manifest.csv")


def parse_long_textgrid(content):
    """Parse standard long-format Praat TextGrid."""
    tiers = {}
    items = content.split("item [")
    for item in items[1:]:
        name_match = re.search(r'name\s*=\s*"([^"]*)"', item)
        class_match = re.search(r'class\s*=\s*"([^"]*)"', item)
        if not name_match or not class_match:
            continue

        tier_name = name_match.group(1)
        tier_class = class_match.group(1)

        if tier_class != "IntervalTier":
            continue

        intervals = []
        # Pattern to match: xmin, xmax, and a Praat-style string with escaped quotes
        pattern = r'xmin\s*=\s*([\d\.\-]+)\s*\n\s*xmax\s*=\s*([\d\.\-]+)\s*\n\s*text\s*=\s*"((?:[^"]|"")*)"'

        for match in re.finditer(pattern, item):
            xmin = float(match.group(1))
            xmax = float(match.group(2))
            text = match.group(3).replace('""', '"').strip()
            intervals.append((xmin, xmax, text))

        tiers[tier_name] = intervals
    return tiers


def parse_short_textgrid(content):
    """Parse standard short-format Praat TextGrid."""
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    if len(lines) < 6:
        return {}

    tiers = {}
    idx = 6
    try:
        while idx < len(lines):
            tier_class = lines[idx].strip('"')
            tier_name = lines[idx + 1].strip('"')
            xmin = float(lines[idx + 2])
            xmax = float(lines[idx + 3])
            num_intervals = int(lines[idx + 4])
            idx += 5

            intervals = []
            if tier_class == "IntervalTier":
                for _ in range(num_intervals):
                    int_xmin = float(lines[idx])
                    int_xmax = float(lines[idx + 1])
                    text = lines[idx + 2].strip('"').replace('""', '"')
                    intervals.append((int_xmin, int_xmax, text))
                    idx += 3
                tiers[tier_name] = intervals
            else:
                for _ in range(num_intervals):
                    idx += 2
    except Exception:
        pass
    return tiers


def parse_textgrid(filepath):
    """Read and parse a TextGrid file, detecting its format."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    content = content.replace("\r\n", "\n").replace("\r", "\n")

    if "item [" not in content and '"IntervalTier"' in content:
        return parse_short_textgrid(content)
    else:
        return parse_long_textgrid(content)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    textgrid_files = glob.glob(os.path.join(PRAAT_DIR, "*.TextGrid"))
    if not textgrid_files:
        print(f"No TextGrid files found in {PRAAT_DIR}.")
        return

    exported_count = 0
    manifest_rows = []

    for grid_path in textgrid_files:
        speaker = os.path.splitext(os.path.basename(grid_path))[0]
        wav_path = os.path.join(PRAAT_DIR, f"{speaker}.wav")

        if not os.path.exists(wav_path):
            print(f"Warning: WAV file not found for {grid_path}. Skipping.")
            continue

        print(f"Parsing {grid_path}...")
        try:
            tiers = parse_textgrid(grid_path)
        except Exception as e:
            print(f"Error parsing {grid_path}: {e}")
            continue

        if "human" not in tiers:
            print(f"Warning: 'human' tier not found in {grid_path}. Skipping.")
            continue

        human_intervals = tiers["human"]
        approved_intervals = [
            (xmin, xmax, text) for (xmin, xmax, text) in human_intervals if text
        ]

        if not approved_intervals:
            continue

        # Open WAV to read sample rate
        info = sf.info(wav_path)
        sr = info.samplerate

        for xmin, xmax, text in approved_intervals:
            start_ms = int(round(xmin * 1000))
            end_ms = int(round(xmax * 1000))

            # Read segment from audio file
            start_frame = int(round(xmin * sr))
            num_frames = int(round((xmax - xmin) * sr))

            try:
                data, _ = sf.read(wav_path, start=start_frame, frames=num_frames)
            except Exception as e:
                print(f"Error reading segment {xmin}-{xmax} from {wav_path}: {e}")
                continue

            # Save approved segment WAV
            seg_filename = f"{speaker}_approved_{start_ms}_{end_ms}.wav"
            seg_path = os.path.join(OUT_DIR, seg_filename)

            try:
                sf.write(seg_path, data, sr)
            except Exception as e:
                print(f"Error writing segment {seg_path}: {e}")
                continue

            manifest_rows.append(
                {
                    "file_path": os.path.join(
                        "data/processed/active_learning", seg_filename
                    ),
                    "transcription": text,
                    "original_speaker": speaker,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                }
            )
            exported_count += 1

    # Write manifest CSV
    if manifest_rows:
        print(f"Writing active learning manifest to {MANIFEST_PATH}...")
        with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "file_path",
                    "transcription",
                    "original_speaker",
                    "start_ms",
                    "end_ms",
                ],
            )
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"Successfully exported {exported_count} approved segments.")
    else:
        print("No approved/edited intervals found in the 'human' tier.")


if __name__ == "__main__":
    main()
