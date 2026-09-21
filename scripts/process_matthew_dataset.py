#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_matthew_dataset.py

Processes all 28 chapters of the Book of Matthew:
1. Aligns audio and transcripts with phonological reconciliation, cached emissions, and pre-Bible confusion matrix.
2. Extracts verse audio segments, resamples to 16kHz mono WAV, and saves as matthew_XX_YY.wav.
3. Generates cherokee_new_testament/train_csvs/matthew.csv with columns: path, sentence.
4. Generates cherokee_new_testament/alignments/matthew_alignment_records.json.
5. Computes audio duration metrics (min, max, median, total seconds, and < 20s cutoff stats).
"""

from pathlib import Path
import sys

# Add repo root to python path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.realign_bible import realign_book


def main():
    print("Starting processing of Book of Matthew (28 chapters)...")
    realign_book("matthew")


if __name__ == "__main__":
    main()
