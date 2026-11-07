#!/usr/bin/env python3
"""
Preprocessing script to filter and split Bible audio transcripts (Mark & Matthew)
into train, validation, and test CSV splits based on word-level alignment confidence
and chapter boundaries.

Pipeline stage:
  align -> confidences (bible_alignment_records.json) -> filtered list & splits
"""

import argparse
import csv
import json
import os
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List

from digohwelisgi.cherokee.orthography import normalize_standard2_cv

# Define Chapter allocations per book
# Target ~80% Train, ~10% Valid, ~10% Test without leaking chapters.
CHAPTER_SPLITS = {
    "mark": {
        "valid_chapters": [4, 11],
        "test_chapters": [7, 16],
    },
    "matthew": {
        "valid_chapters": [3, 6, 17, 19],
        "test_chapters": [2, 9, 20, 28],
    },
}

DEFAULT_RECORDS_PATH = (
    "data/projects/cherokee_new_testament/alignments/bible_alignment_records.json"
)
DEFAULT_TRAIN_CSVS_DIR = "data/projects/cherokee_new_testament/train_csvs"
DEFAULT_OUTPUT_DIR = "data/training/processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter Bible alignment records by minimum word confidence and generate train/valid/test splits."
    )
    parser.add_argument(
        "--min-word-confidence",
        type=float,
        default=0.15,
        help="Minimum confidence threshold for all words in a verse (default: 0.15).",
    )
    parser.add_argument(
        "--records-path",
        type=str,
        default=DEFAULT_RECORDS_PATH,
        help="Path to bible_alignment_records.json containing alignments and word confidences.",
    )
    parser.add_argument(
        "--train-csvs-dir",
        type=str,
        default=DEFAULT_TRAIN_CSVS_DIR,
        help="Directory to save per-book training CSVs (mark.csv, matthew.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save final training splits (bible-wav2vec2-*.csv).",
    )
    return parser.parse_args()


def is_accepted_verse(record: Dict[str, Any], min_word_conf: float) -> bool:
    """A verse is accepted if it has aligned words, emitted text, and all word confidences > min_word_conf."""
    words = record.get("words", [])
    emitted = (
        record.get("greedy_hypothesis") or record.get("emitted_text") or ""
    ).strip()
    if not words or not emitted:
        return False

    confs = [w.get("confidence") for w in words]
    if any(c is None or c <= min_word_conf for c in confs):
        return False

    return True


def main():
    args = parse_args()
    min_word_conf = args.min_word_confidence

    records_file = Path(args.records_path)
    if not records_file.exists():
        raise FileNotFoundError(f"Alignment records not found: {records_file}")

    with open(records_file, "r", encoding="utf-8") as f:
        records: List[Dict[str, Any]] = json.load(f)

    os.makedirs(args.train_csvs_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loaded {len(records)} alignment records from {records_file}")
    print(f"Filtering with minimum word confidence > {min_word_conf}...")

    # Filter records
    accepted_records = [r for r in records if is_accepted_verse(r, min_word_conf)]
    print(
        f"Accepted verses: {len(accepted_records)} / {len(records)} "
        f"({len(accepted_records) / len(records) * 100:.1f}%)"
    )

    # Group by book and partition by chapter
    book_rows: Dict[str, List[Dict[str, str]]] = {}
    train_rows: List[Dict[str, str]] = []
    valid_rows: List[Dict[str, str]] = []
    test_rows: List[Dict[str, str]] = []

    phrase_confs = []
    worst_word_confs = []

    for r in accepted_records:
        book = str(r.get("book", "")).lower().strip()
        ch = int(r.get("chapter", 0))
        idx = int(r.get("verse_idx", 0))
        wav_path = f"cherokee_new_testament/split_audio/{book}_{ch:02d}_{idx:02d}.wav"
        emitted = (r.get("greedy_hypothesis") or r.get("emitted_text") or "").strip()
        raw_sentence = (r.get("reconciled_phonetics") or emitted).strip()
        cv_text = normalize_standard2_cv(raw_sentence)
        if not cv_text:
            continue

        row = {"path": wav_path, "cv_text": cv_text, "tone_tokens": ""}
        if book not in book_rows:
            book_rows[book] = []
        book_rows[book].append(row)

        splits = CHAPTER_SPLITS.get(book, {})
        valid_ch = set(splits.get("valid_chapters", []))
        test_ch = set(splits.get("test_chapters", []))

        if ch in valid_ch:
            valid_rows.append(row)
        elif ch in test_ch:
            test_rows.append(row)
        else:
            train_rows.append(row)

        if r.get("greedy_confidence") is not None:
            phrase_confs.append(float(r["greedy_confidence"]))
        word_scores = [
            w["confidence"]
            for w in r.get("words", [])
            if w.get("confidence") is not None
        ]
        if word_scores:
            worst_word_confs.append(min(word_scores))

    # Save per-book train_csvs
    for book, rows in book_rows.items():
        csv_path = Path(args.train_csvs_dir) / f"{book}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "cv_text", "tone_tokens"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {csv_path}: {len(rows)} verses")

    # Save training splits
    train_out = Path(args.output_dir) / "bible-wav2vec2-train.csv"
    valid_out = Path(args.output_dir) / "bible-wav2vec2-valid.csv"
    test_out = Path(args.output_dir) / "bible-wav2vec2-test.csv"

    for path, rows in [
        (train_out, train_rows),
        (valid_out, valid_rows),
        (test_out, test_rows),
    ]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "cv_text", "tone_tokens"])
            writer.writeheader()
            writer.writerows(rows)
        pct = len(rows) / len(accepted_records) * 100 if accepted_records else 0
        print(f"Saved {path}: {len(rows)} verses ({pct:.1f}%)")

    # Confidence summary
    if phrase_confs:
        print("\n--- Confidence Statistics (Accepted Phrases) ---")
        print(
            f"Phrase Mean Conf:  Min: {min(phrase_confs):.4f} | "
            f"Median: {statistics.median(phrase_confs):.4f} | "
            f"Mean: {statistics.mean(phrase_confs):.4f} | "
            f"Max: {max(phrase_confs):.4f}"
        )
    if worst_word_confs:
        print(
            f"Worst Word in Verse: Min: {min(worst_word_confs):.4f} | "
            f"Median: {statistics.median(worst_word_confs):.4f} | "
            f"Mean: {statistics.mean(worst_word_confs):.4f} | "
            f"Max: {max(worst_word_confs):.4f}"
        )


if __name__ == "__main__":
    main()
