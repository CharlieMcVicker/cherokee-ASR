#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
praat_prepare.py

Reads 'data/results/cvcs_all_noisy.csv', converts the original MP3 source audio files
to 16kHz 16-bit mono WAV files, and generates Praat .TextGrid files with aligned tiers:
- 'machine': Pre-filled with ASR machine transcriptions.
- 'human': Pre-populated with matching boundaries but empty text.
"""

import os
import csv
import re
import subprocess
import soundfile as sf

CSV_PATH = "data/results/cvcs_all_noisy.csv"
MP3_DIR = "cvcs-mp3s"
OUT_DIR = "data/processed/praat"


def get_audio_duration(wav_path):
    """Return duration of WAV file in seconds."""
    info = sf.info(wav_path)
    return info.duration


def convert_mp3_to_wav(mp3_path, wav_path):
    """Convert MP3 file to 16kHz, 16-bit, mono WAV using ffmpeg."""
    print(f"Converting {mp3_path} to {wav_path}...")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            mp3_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            wav_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def write_textgrid(filepath, total_duration, machine_intervals, human_intervals):
    """Write standard long-format Praat TextGrid file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write('File type = "ooTextFile"\n')
        f.write('Object class = "TextGrid"\n\n')
        f.write("xmin = 0\n")
        f.write(f"xmax = {total_duration}\n")
        f.write("tiers? <exists>\n")
        f.write("size = 2\n")
        f.write("item []:\n")

        # Tier 1: machine
        f.write("    item [1]:\n")
        f.write('        class = "IntervalTier"\n')
        f.write('        name = "machine"\n')
        f.write("        xmin = 0\n")
        f.write(f"        xmax = {total_duration}\n")
        f.write(f"        intervals: size = {len(machine_intervals)}\n")
        for i, (xmin, xmax, text) in enumerate(machine_intervals, start=1):
            escaped_text = text.replace('"', '""')
            f.write(f"        intervals [{i}]:\n")
            f.write(f"            xmin = {xmin}\n")
            f.write(f"            xmax = {xmax}\n")
            f.write(f'            text = "{escaped_text}"\n')

        # Tier 2: human
        f.write("    item [2]:\n")
        f.write('        class = "IntervalTier"\n')
        f.write('        name = "human"\n')
        f.write("        xmin = 0\n")
        f.write(f"        xmax = {total_duration}\n")
        f.write(f"        intervals: size = {len(human_intervals)}\n")
        for i, (xmin, xmax, text) in enumerate(human_intervals, start=1):
            escaped_text = text.replace('"', '""')
            f.write(f"        intervals [{i}]:\n")
            f.write(f"            xmin = {xmin}\n")
            f.write(f"            xmax = {xmax}\n")
            f.write(f'            text = "{escaped_text}"\n')


def build_contiguous_intervals(segments, total_duration):
    """
    Sort segments and fill all gaps from 0 to total_duration with empty intervals.
    """
    # Sort segments by start time
    sorted_segs = sorted(segments, key=lambda x: x[0])
    intervals = []
    current_time = 0.0

    for start_sec, end_sec, text in sorted_segs:
        # Prevent boundary errors
        if start_sec < current_time:
            start_sec = current_time
        if end_sec > total_duration:
            end_sec = total_duration

        if start_sec > current_time:
            intervals.append((current_time, start_sec, ""))

        if end_sec > start_sec:
            intervals.append((start_sec, end_sec, text))
            current_time = end_sec

    if current_time < total_duration:
        intervals.append((current_time, total_duration, ""))

    return intervals


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    # Group transcription lines by speaker/recording
    recordings = {}

    print(f"Reading transcripts from {CSV_PATH}...")
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_path = row["file_path"]
            filename = row["filename"]
            text = row["greedy_transcription"]

            # Parse speaker/source name from folder structure or filename prefix
            # E.g., data/processed/noisy_segments/Rosanna Vann/...
            speaker = os.path.basename(os.path.dirname(file_path))

            # Parse start and end times in ms from the filename
            # E.g., Rosanna Vann_segment_0548_1667795_1668715.wav
            match = re.search(r"_segment_\d+_(\d+)_(\d+)\.wav$", filename)
            if not match:
                continue

            start_ms = int(match.group(1))
            end_ms = int(match.group(2))

            start_sec = start_ms / 1000.0
            end_sec = end_ms / 1000.0

            if speaker not in recordings:
                recordings[speaker] = []
            recordings[speaker].append((start_sec, end_sec, text))

    print(f"Found transcripts for {len(recordings)} unique recordings.")

    for speaker, segments in recordings.items():
        mp3_path = os.path.join(MP3_DIR, f"{speaker}.mp3")
        wav_path = os.path.join(OUT_DIR, f"{speaker}.wav")
        grid_path = os.path.join(OUT_DIR, f"{speaker}.TextGrid")

        if not os.path.exists(mp3_path):
            print(
                f"Warning: MP3 source not found for {speaker} at {mp3_path}. Skipping."
            )
            continue

        # Convert MP3 to 16kHz WAV if it doesn't exist
        if not os.path.exists(wav_path):
            try:
                convert_mp3_to_wav(mp3_path, wav_path)
            except Exception as e:
                print(f"Error converting {mp3_path}: {e}")
                continue

        # Get total duration
        try:
            total_duration = get_audio_duration(wav_path)
        except Exception as e:
            print(f"Error reading duration of {wav_path}: {e}")
            continue

        # Build contiguous intervals
        machine_intervals = build_contiguous_intervals(segments, total_duration)

        # Human intervals have identical boundaries but empty text
        human_intervals = [(xmin, xmax, "") for (xmin, xmax, _) in machine_intervals]

        print(f"Writing {grid_path}...")
        write_textgrid(grid_path, total_duration, machine_intervals, human_intervals)

    print("PRAAT preparation complete!")


if __name__ == "__main__":
    main()
