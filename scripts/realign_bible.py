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
from transcription.models.asr_model import CherokeeASRModel
from transcription.new_testament.pipeline import (
    align_chapter,
    load_chapter_transcript,
    reconcile_syllabary_asr,
)

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
DEFAULT_MODEL_REPO = "charliemcvicker/length-only-20260704-155307-asr-cherokee-colon"
DEFAULT_REVISION = "76e62140955f4738abdab345ea34068b02d8d2a2"

BOOK_CONFIGS = {
    "mark": {"chapters": 16, "name": "Mark"},
    "matthew": {"chapters": 28, "name": "Matthew"},
}


def get_default_extractor_and_metric(
    model_repo: str = DEFAULT_MODEL_REPO,
    model_revision: str = DEFAULT_REVISION,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    cost_matrix_path: Path = DEFAULT_COST_MATRIX_PATH,
) -> Tuple[CachedASREmissionsExtractor, DistanceMetric]:
    token = os.environ.get("HF_TOKEN", None)
    asr_model = CherokeeASRModel.from_pretrained_or_best(
        path_or_repo=model_repo,
        revision=model_revision,
        token=token,
    )
    base_extractor = CherokeeASRExtractor(model=asr_model, skip_vad=False)
    prefix_slug = (
        f"{model_repo.replace('/', '_')}_{model_revision}"
        if model_revision
        else model_repo.replace("/", "_")
    )
    cached_extractor = CachedASREmissionsExtractor(
        extractor=base_extractor,
        cache_dir=cache_dir,
        cache_key_prefix=prefix_slug,
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
    model_repo: str = DEFAULT_MODEL_REPO,
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
            model_repo=model_repo,
            model_revision=model_revision,
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
            print(f"[Warning] Transcript not found: {transcript_path}, skipping.")
            continue

        print(f"\n>>> Realigning {book_display} Chapter {ch} ...")
        res = align_chapter(
            audio_path=audio_path,
            transcript_path=transcript_path,
            output_dir=PRAAT_OUT_DIR,
            export_praat=export_praat,
            distance_metric=distance_metric,
            emissions_extractor=emissions_extractor,
            model_revision=model_revision,
        )

        audio_seg = AudioSegment.from_file(str(audio_path))
        num_verses_aligned = len(res.aligned_chunks)
        print(f"    Aligned {num_verses_aligned} verses.")

        for chunk in res.aligned_chunks:
            verse_id = chunk.chunk_id
            start_sec = round(chunk.start_sec, 3)
            end_sec = round(chunk.end_sec, 3)
            dur = round(end_sec - start_sec, 3)
            durations_sec.append(dur)

            # Determine verse index & filename
            if verse_id.isdigit() and len(verse_id) == 6:
                v_num = int(verse_id[4:6])
                verse_idx = v_num
                split_filename = f"{book_key}_{ch:02d}_{v_num:02d}.wav"
            else:
                verse_idx = len(records) + 1
                split_filename = f"{book_key}_{ch_str}_{verse_id}.wav"

            split_out_path = SPLIT_AUDIO_DIR / split_filename

            # Slicing audio
            start_ms = int(start_sec * 1000)
            end_ms = int(end_sec * 1000)
            verse_audio = audio_seg[start_ms:end_ms]
            verse_audio = verse_audio.set_frame_rate(16000).set_channels(1)
            verse_audio.export(str(split_out_path), format="wav")

            # Load verse transcript text
            ch_data = load_chapter_transcript(transcript_path)
            verse_info = ch_data.get(verse_id, {})
            cherokee_text = (
                verse_info.get("cherokee")
                or verse_info.get("text")
                or verse_info.get("syllabary", "")
            )
            phonetic_text = verse_info.get("phonetic", "")
            english_text = verse_info.get("english", "")

            # Reconcile syllabary & ASR tokens
            asr_hyp = chunk.emitted_text or ""
            reconciled_syllabary, _ = reconcile_syllabary_asr(
                syllabary_text=cherokee_text,
                asr_hypothesis=asr_hyp,
            )

            words_list = [
                {
                    "word": w.word,
                    "start_sec": round(w.start_sec, 3),
                    "end_sec": round(w.end_sec, 3),
                    "confidence": w.confidence,
                    "flagged": w.flagged,
                    "emitted_word": w.emitted_word or "",
                }
                for w in chunk.words
            ]

            record = {
                "verse_id": verse_id,
                "book": book_key,
                "chapter": ch,
                "verse_idx": verse_idx,
                "audio_path": f"cherokee_new_testament/split_audio/{split_filename}",
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": dur,
                "reference_sentence": cherokee_text,
                "reconciled_phonetics": reconciled_syllabary,
                "asr_hypothesis": asr_hyp,
                "cost": round(chunk.distance_score, 4),
                "words": words_list,
                "cherokee_syllabary": cherokee_text,
                "phonetic": phonetic_text,
                "english": english_text,
            }
            records.append(record)

            csv_rows.append(
                {
                    "path": f"cherokee_new_testament/split_audio/{split_filename}",
                    "sentence": reconciled_syllabary or phonetic_text,
                }
            )

    # Export per-book alignment JSON
    book_alignments_path = ALIGNMENTS_DIR / f"{book_key}_alignment_records.json"
    with open(book_alignments_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(
        f"\n[Artifact] Saved {len(records)} alignment records to '{book_alignments_path}'"
    )

    # Export training CSV
    train_csv_path = TRAIN_CSVS_DIR / f"{book_key}.csv"
    with open(train_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "sentence"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(
        f"[Artifact] Saved training CSV with {len(csv_rows)} rows to '{train_csv_path}'"
    )

    train_csv_alt = TRAIN_CSVS_DIR / f"{book_key}_train.csv"
    with open(train_csv_alt, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "sentence"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(
        f"[Artifact] Saved training CSV with {len(csv_rows)} rows to '{train_csv_path}'"
    )

    if durations_sec:
        total_sec = sum(durations_sec)
        mean_len = float(np.mean(durations_sec))
        median_len = float(np.median(durations_sec))
        print(f"\n--- {book_display} Audio Slicing Summary ---")
        print(f"Total sliced segments: {len(durations_sec)}")
        print(f"Mean verse length    : {mean_len:.2f} seconds")
        print(f"Median verse length  : {median_len:.2f} seconds")
        print(
            f"Total audio duration : {total_sec:.2f} seconds ({total_sec/60:.2f} minutes / {total_sec/3600:.2f} hours)"
        )

    return records, csv_rows


def realign_all(
    model_repo: str = DEFAULT_MODEL_REPO,
    model_revision: str = DEFAULT_REVISION,
    export_praat: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    cached_extractor, distance_metric = get_default_extractor_and_metric(
        model_repo=model_repo,
        model_revision=model_revision,
    )

    all_records: List[Dict[str, Any]] = []
    book_results: Dict[str, List[Dict[str, Any]]] = {}

    for book in ["mark", "matthew"]:
        records, _ = realign_book(
            book=book,
            distance_metric=distance_metric,
            emissions_extractor=cached_extractor,
            model_repo=model_repo,
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
        "--model-repo",
        default=DEFAULT_MODEL_REPO,
        help=f"HF model repository (default: {DEFAULT_MODEL_REPO})",
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
            model_repo=args.model_repo,
            model_revision=args.model_revision,
            export_praat=not args.no_praat,
        )
    else:
        records, _ = realign_book(
            book=args.book,
            model_repo=args.model_repo,
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
