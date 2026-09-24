#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_mark_dataset.py

Processes all 16 chapters of the Book of Mark:
1. Aligns audio and transcripts with phonological reconciliation, cached emissions, and pre-Bible confusion matrix.
2. Extracts verse audio segments, resamples to 16kHz mono WAV, and saves as mark_XX_YY.wav.
3. Generates data/projects/cherokee_new_testament/train_csvs/mark.csv with columns: path, sentence.
4. Generates data/projects/cherokee_new_testament/alignments/mark_alignment_records.json.
5. Computes audio duration metrics (min, max, median, total seconds).
"""

from pathlib import Path
import sys

# Add repo root to python path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.realign_bible import realign_book


def main():
    print("Starting processing of Book of Mark (16 chapters)...")
    realign_book("mark")


if __name__ == "__main__":
    main()
