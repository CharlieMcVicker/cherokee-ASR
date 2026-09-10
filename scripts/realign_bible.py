#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
realign_bible.py

Realigns New Testament books (Mark, Matthew) using:
1. Baseline pre-Bible ASR model (charliemcvicker/asr-cherokee:5464d15).
2. CachedASREmissionsExtractor to cache and reuse chapter emissions.
3. ConfusionMatrixCostMetric loaded from confusion_cost_matrix_prebible.json.
4. Phonological syllabary/ASR reconciliation.
5. Slices audio into 16kHz mono WAV files in cherokee_new_testament/split_audio/.
6. Exports full alignment records to cherokee_new_testament/alignments/.
7. Exports training CSVs to cherokee_new_testament/train_csvs/.
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydub import AudioSegment

from transcription.alignment.calibrated_distance_metrics import (
    PhonologicalConfusionCostMetric,
)
from transcription.alignment.distance_metrics import (
    ConfusionMatrixCostMetric,
    DistanceMetric,
)
from transcription.alignment.extractors import (
    ASREmissionsExtractor,
    CachedASREmissionsExtractor,
    CherokeeASRExtractor,
)
from transcription.alignment.reconciliation import reconcile_word_intervals
from transcription.models.asr_model import CherokeeASRModel
from transcription.new_testament.pipeline import align_chapter, load_chapter_transcript

BASE_DIR = Path(__file__).resolve().parent.parent
NT_DIR = BASE_DIR / "cherokee_new_testament"
AUDIO_SRC_DIR = NT_DIR / "audio_source"
TRANSCRIPTS_DIR = NT_DIR / "book_transcripts"
SPLIT_AUDIO_DIR = NT_DIR / "split_audio"
ALIGNMENTS_DIR = NT_DIR / "alignments"
TRAIN_CSVS_DIR = NT_DIR / "train_csvs"
PRAAT_OUT_DIR = BASE_DIR / "output_praat" / "new_testament"
DEFAULT_COST_MATRIX_PATH = (
    BASE_DIR / "runs" / "evaluation" / "confusion_cost_matrix_prebible.json"
)
DEFAULT_CACHE_DIR = BASE_DIR / "runs" / "cache" / "emissions"
DEFAULT_REVISION = "5464d15"

BOOK_CONFIGS = {
    "mark": {"chapters": 16, "name": "Mark"},
    "matthew": {"chapters": 28, "name": "Matthew"},
}


def get_default_extractor_and_metric(
    model_revision: str = DEFAULT_REVISION,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    cost_matrix_path: Path = DEFAULT_COST_MATRIX_PATH,
) -> Tuple[CachedASREmissionsExtractor, DistanceMetric]:
    token = os.environ.get("HF_TOKEN", None)
    asr_model = CherokeeASRModel.from_pretrained_or_best(
        path_or_repo="charliemcvicker/asr-cherokee",
        revision=model_revision,
        token=token,
    )
    base_extractor = CherokeeASRExtractor(model=asr_model, skip_vad=False)
    cached_extractor = CachedASREmissionsExtractor(
        extractor=base_extractor,
        cache_dir=cache_dir,
        cache_key_prefix=f"charliemcvicker_asr-cherokee_{model_revision}",
    )

    if cost_matrix_path.exists():
        base_metric = ConfusionMatrixCostMetric.from_json(cost_matrix_path)
        distance_metric = PhonologicalConfusionCostMetric(base_metric=base_metric)
    else:
        raise FileNotFoundError(
            f"Confusion cost matrix not found at: {cost_matrix_path}"
        )

    return cached_extractor, distance_metric


def realign_book(
    book: str,
    distance_metric: Optional[DistanceMetric] = None,
    emissions_extractor: Optional[ASREmissionsExtractor] = None,
    model_revision: str = DEFAULT_REVISION,
    export_praat: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    book_key = book.lower().strip()
    if book_key not in BOOK_CONFIGS:
        raise ValueError(
            f"Unknown book: '{book}'. Supported: {list(BOOK_CONFIGS.keys())}"
        )

    config = BOOK_CONFIGS[book_key]
    num_chapters = config["chapters"]
    book_display = config["name"]

    SPLIT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    ALIGNMENTS_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_CSVS_DIR.mkdir(parents=True, exist_ok=True)

    if emissions_extractor is None or distance_metric is None:
        def_extractor, def_metric = get_default_extractor_and_metric(
            model_revision=model_revision
        )
        emissions_extractor = emissions_extractor or def_extractor
        distance_metric = distance_metric or def_metric

    records: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, str]] = []
    durations_sec: List[float] = []

    print(f"\n==================================================")
    print(f"Starting Realignment of Book of {book_display} ({num_chapters} chapters)")
    print(f"==================================================")

    for ch in range(1, num_chapters + 1):
        ch_str = f"{ch:02d}"
        audio_path = AUDIO_SRC_DIR / f"{book_key}_{ch_str}.mp3"
        transcript_path = TRANSCRIPTS_DIR / f"{book_key}_{ch_str}.json"

        if not audio_path.exists():
            print(f"[Warning] Audio file not found: {audio_path}, skipping.")
            continue
        if not transcript_path.exists():
            print(f"[Warning] Transcript JSON not found: {transcript_path}, skipping.")
            continue

        print(f"\n--- Aligning {book_display} Chapter {ch_str}/{num_chapters:02d} ---")
        out_praat_ch = PRAAT_OUT_DIR / f"{book_key}_{ch_str}"

        alignment = align_chapter(
            audio_path=audio_path,
            transcript_path=transcript_path,
            output_dir=out_praat_ch,
            export_praat=export_praat,
            reconcile=True,
            distance_metric=distance_metric,
            emissions_extractor=emissions_extractor,
            model_revision=model_revision,
        )

        full_audio = AudioSegment.from_file(audio_path)
        transcript_data = load_chapter_transcript(transcript_path)

        for v_idx, v in enumerate(alignment.aligned_chunks, 1):
            if not v.words or v.end_sec <= v.start_sec:
                continue

            verse_num_str = f"{v_idx:02d}"
            wav_filename = f"{book_key}_{ch_str}_{verse_num_str}.wav"
            wav_path = SPLIT_AUDIO_DIR / wav_filename

            # Extract audio clip
            start_ms = int(v.start_sec * 1000)
            end_ms = int(v.end_sec * 1000)
            clip = full_audio[start_ms:end_ms]

            # Resample to 16kHz Mono 16-bit WAV
            clip = clip.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            clip.export(wav_path, format="wav")

            # Extract reconciled sentence string and word intervals
            syll_text = transcript_data.get(v.chunk_id, {}).get("cherokee", "")
            rec_word_intervals = reconcile_word_intervals(v.words, syll_text)

            rec_sentence = " ".join(
                w.word for w in rec_word_intervals if w.word
            ).strip()
            if not rec_sentence:
                rec_sentence = transcript_data.get(v.chunk_id, {}).get("phonetic", "")

            ref_sentence = (
                syll_text
                if syll_text
                else transcript_data.get(v.chunk_id, {}).get("phonetic", "")
            )

            words_data = [
                {
                    "word": w.word,
                    "start_sec": w.start_sec,
                    "end_sec": w.end_sec,
                    "confidence": w.confidence,
                    "flagged": w.flagged,
                    "emitted_word": w.emitted_word,
                }
                for w in rec_word_intervals
            ]

            dur_sec = round(v.end_sec - v.start_sec, 3)
            durations_sec.append(dur_sec)

            rel_audio_path = f"cherokee_new_testament/split_audio/{wav_filename}"

            record: Dict[str, Any] = {
                "verse_id": v.chunk_id,
                "book": book_key,
                "chapter": ch,
                "verse_idx": v_idx,
                "audio_path": rel_audio_path,
                "start_sec": round(v.start_sec, 3),
                "end_sec": round(v.end_sec, 3),
                "duration_sec": dur_sec,
                "reference_sentence": ref_sentence,
                "reconciled_phonetics": rec_sentence,
                "asr_hypothesis": v.emitted_text,
                "cost": float(v.distance_score),
                "words": words_data,
            }
            records.append(record)

            csv_rows.append({"path": rel_audio_path, "sentence": rec_sentence})

    # Save book-specific alignment records
    records_out_path = ALIGNMENTS_DIR / f"{book_key}_alignment_records.json"
    with open(records_out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(
        f"\n[Artifact] Saved {len(records)} alignment records to '{records_out_path}'"
    )

    # Save training CSV
    csv_out_path = TRAIN_CSVS_DIR / f"{book_key}.csv"
    with open(csv_out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "sentence"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[Artifact] Saved {len(csv_rows)} CSV rows to '{csv_out_path}'")

    if durations_sec:
        min_len = min(durations_sec)
        max_len = max(durations_sec)
        median_len = float(np.median(durations_sec))
        total_sec = sum(durations_sec)

        print(f"\n=== Audio Statistics for Book of {book_display} ===")
        print(f"Total verse segments : {len(durations_sec)}")
        print(f"Min verse length     : {min_len:.2f} seconds")
        print(f"Max verse length     : {max_len:.2f} seconds")
        print(f"Median verse length  : {median_len:.2f} seconds")
        print(
            f"Total audio duration : {total_sec:.2f} seconds ({total_sec/60:.2f} minutes / {total_sec/3600:.2f} hours)"
        )

    return records, csv_rows


def realign_all(
    model_revision: str = DEFAULT_REVISION,
    export_praat: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    cached_extractor, distance_metric = get_default_extractor_and_metric(
        model_revision=model_revision
    )

    all_records: List[Dict[str, Any]] = []
    book_results: Dict[str, List[Dict[str, Any]]] = {}

    for book in ["mark", "matthew"]:
        records, _ = realign_book(
            book=book,
            distance_metric=distance_metric,
            emissions_extractor=cached_extractor,
            model_revision=model_revision,
            export_praat=export_praat,
        )
        book_results[book] = records
        all_records.extend(records)

    # Save combined alignment records
    ALIGNMENTS_DIR.mkdir(parents=True, exist_ok=True)
    combined_out_path = ALIGNMENTS_DIR / "bible_alignment_records.json"
    with open(combined_out_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)
    print(
        f"\n[Artifact] Saved {len(all_records)} combined alignment records to '{combined_out_path}'"
    )

    return book_results


def main():
    parser = argparse.ArgumentParser(
        description="Realign Cherokee New Testament books with pre-Bible model and confusion cost metric."
    )
    parser.add_argument(
        "--book",
        choices=["mark", "matthew", "all"],
        default="all",
        help="Book to realign (default: all)",
    )
    parser.add_argument(
        "--model-revision",
        default=DEFAULT_REVISION,
        help=f"HF model revision (default: {DEFAULT_REVISION})",
    )
    parser.add_argument(
        "--no-praat",
        action="store_true",
        default=False,
        help="Skip Praat TextGrid export",
    )

    args = parser.parse_args()

    if args.book == "all":
        realign_all(
            model_revision=args.model_revision,
            export_praat=not args.no_praat,
        )
    else:
        records, _ = realign_book(
            book=args.book,
            model_revision=args.model_revision,
            export_praat=not args.no_praat,
        )
        # Also update combined if single book is run
        ALIGNMENTS_DIR.mkdir(parents=True, exist_ok=True)
        combined_path = ALIGNMENTS_DIR / "bible_alignment_records.json"
        existing_records = []
        if combined_path.exists():
            try:
                with open(combined_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    existing_records = [
                        r for r in existing if r.get("book") != args.book
                    ]
            except Exception:
                existing_records = []
        existing_records.extend(records)
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(existing_records, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
