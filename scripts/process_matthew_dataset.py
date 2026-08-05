#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_matthew_dataset.py

Processes all 28 chapters of the Book of Matthew:
1. Aligns audio and transcripts with phonological reconciliation enabled.
2. Extracts verse audio segments, resamples to 16kHz mono WAV, and saves as matthew_XX_YY.wav.
3. Generates cherokee_new_testament/train_csvs/matthew.csv with columns: path, sentence.
4. Computes audio duration metrics (min, max, median, total seconds, and < 20s cutoff stats).
"""

import csv
import os
import sys
import numpy as np
from pathlib import Path
from pydub import AudioSegment

from transcription.new_testament.pipeline import align_chapter

BASE_DIR = Path("/Users/julietmcvicker/code/workshop-transcription")
NT_DIR = BASE_DIR / "cherokee_new_testament"
AUDIO_SRC_DIR = NT_DIR / "audio_source"
TRANSCRIPTS_DIR = NT_DIR / "book_transcripts"
SPLIT_AUDIO_DIR = NT_DIR / "split_audio"
TRAIN_CSVS_DIR = NT_DIR / "train_csvs"
PRAAT_OUT_DIR = BASE_DIR / "output_praat" / "new_testament"


def main():
    SPLIT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_CSVS_DIR.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    durations_sec = []

    print("Starting processing of Book of Matthew (28 chapters)...")

    for ch in range(1, 29):
        ch_str = f"{ch:02d}"
        audio_path = AUDIO_SRC_DIR / f"matthew_{ch_str}.mp3"
        transcript_path = TRANSCRIPTS_DIR / f"matthew_{ch_str}.json"

        if not audio_path.exists():
            print(f"[Warning] Audio file not found: {audio_path}, skipping.")
            continue
        if not transcript_path.exists():
            print(f"[Warning] Transcript JSON not found: {transcript_path}, skipping.")
            continue

        print(f"\n--- Aligning Matthew Chapter {ch_str} ---")
        out_praat_ch = PRAAT_OUT_DIR / f"matthew_{ch_str}"

        alignment = align_chapter(
            audio_path=audio_path,
            transcript_path=transcript_path,
            output_dir=out_praat_ch,
            export_praat=True,
            reconcile=True,
        )

        full_audio = AudioSegment.from_file(audio_path)

        for v_idx, v in enumerate(alignment.verses, 1):
            if not v.words or v.end_sec <= v.start_sec:
                continue

            verse_num_str = f"{v_idx:02d}"
            wav_filename = f"matthew_{ch_str}_{verse_num_str}.wav"
            wav_path = SPLIT_AUDIO_DIR / wav_filename

            # Extract audio clip
            start_ms = int(v.start_sec * 1000)
            end_ms = int(v.end_sec * 1000)
            clip = full_audio[start_ms:end_ms]

            # Resample to 16kHz Mono 16-bit WAV
            clip = clip.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            clip.export(wav_path, format="wav")

            # Extract reconciled sentence string across verse words
            reconciled_words = []
            for w in v.words:
                rec_w = getattr(w, "reconciled_word", "") or w.word
                if rec_w:
                    reconciled_words.append(rec_w)

            sentence = " ".join(reconciled_words).strip()
            if not sentence:
                sentence = v.raw_phonetic

            # Store relative path for training CSV
            rel_path = f"cherokee_new_testament/split_audio/{wav_filename}"
            csv_rows.append({"path": rel_path, "sentence": sentence})

            dur_sec = v.end_sec - v.start_sec
            durations_sec.append(dur_sec)

    csv_out_path = TRAIN_CSVS_DIR / "matthew.csv"
    with open(csv_out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "sentence"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nSaved training CSV to '{csv_out_path}' with {len(csv_rows)} rows.")

    if durations_sec:
        min_len = min(durations_sec)
        max_len = max(durations_sec)
        median_len = float(np.median(durations_sec))
        total_sec = sum(durations_sec)

        under_20 = [d for d in durations_sec if d < 20.0]
        under_20_sec = sum(under_20)
        under_20_count = len(under_20)

        print("\n=== Audio Statistics for Book of Matthew ===")
        print(f"Total verse segments : {len(durations_sec)}")
        print(f"Min verse length     : {min_len:.2f} seconds")
        print(f"Max verse length     : {max_len:.2f} seconds")
        print(f"Median verse length  : {median_len:.2f} seconds")
        print(
            f"Total audio duration : {total_sec:.2f} seconds ({total_sec/60:.2f} minutes / {total_sec/3600:.2f} hours)"
        )
        print("\n--- Under 20 Seconds Cutoff Stats ---")
        print(
            f"Verses < 20s count   : {under_20_count} / {len(durations_sec)} ({under_20_count/len(durations_sec)*100:.1f}%)"
        )
        print(
            f"Verses < 20s audio   : {under_20_sec:.2f} seconds ({under_20_sec/60:.2f} minutes / {under_20_sec/3600:.2f} hours)"
        )


if __name__ == "__main__":
    main()
