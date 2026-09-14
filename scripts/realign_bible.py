#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
realign_bible.py

Realigns New Testament books (Mark, Matthew) using:
1. CTCSegmentationAligner with syncope-aware DP trellis segmentation on continuous chapter audio.
2. Slices unclipped verse audio into 16kHz mono WAV files in cherokee_new_testament/split_audio/
   using natural inter-verse boundary partition points.
3. Performs phonological syllabary/ASR reconciliation.
4. Exports 4-tier Praat TextGrids to output_praat/new_testament/{book}_{ch}/.
5. Exports full alignment records to cherokee_new_testament/alignments/{book}_alignment_records.json and bible_alignment_records.json.
6. Exports training CSVs to cherokee_new_testament/train_csvs/.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydub import AudioSegment
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from transcription.alignment.ctc_aligner import CTCSegmentationAligner
from transcription.alignment.models import CTCAlignerConfig
from transcription.models.asr_model import CherokeeASRModel
from transcription.new_testament.pipeline import (
    align_chapter,
    load_chapter_transcript,
)

NT_DIR = BASE_DIR / "cherokee_new_testament"
AUDIO_SRC_DIR = NT_DIR / "audio_source"
TRANSCRIPTS_DIR = NT_DIR / "book_transcripts"
SPLIT_AUDIO_DIR = NT_DIR / "split_audio"
ALIGNMENTS_DIR = NT_DIR / "alignments"
TRAIN_CSVS_DIR = NT_DIR / "train_csvs"
PRAAT_OUT_DIR = BASE_DIR / "output_praat" / "new_testament"
DEFAULT_CACHE_DIR = BASE_DIR / "runs" / "cache" / "ctc_emissions"
DEFAULT_MODEL_REPO = "charliemcvicker/length-only-20260704-155307-asr-cherokee-colon"
DEFAULT_REVISION = "76e62140955f4738abdab345ea34068b02d8d2a2"
DEFAULT_SYNCOPE_PENALTY = 6.0
DEFAULT_INTRUSIVE_PENALTY = 0.1

BOOK_CONFIGS = {
    "mark": {"chapters": 16, "name": "Mark"},
    "matthew": {"chapters": 28, "name": "Matthew"},
}


def get_default_ctc_aligner(
    model_repo: str = DEFAULT_MODEL_REPO,
    model_revision: str = DEFAULT_REVISION,
    config: Optional[CTCAlignerConfig] = None,
) -> CTCSegmentationAligner:
    """Instantiates default CTCSegmentationAligner with cached emissions."""
    token = os.environ.get("HF_TOKEN", None)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    asr_model = CherokeeASRModel.from_pretrained_or_best(
        path_or_repo=model_repo,
        revision=model_revision,
        device=device,
        token=token,
    )
    aligner = CTCSegmentationAligner(
        model=asr_model,
        config=config or CTCAlignerConfig(cache_dir=DEFAULT_CACHE_DIR),
    )
    return aligner


def realign_book(
    book: str,
    chapter: Optional[int] = None,
    ctc_aligner: Optional[CTCSegmentationAligner] = None,
    model_repo: str = DEFAULT_MODEL_REPO,
    model_revision: str = DEFAULT_REVISION,
    export_praat: bool = True,
    aligner_config: Optional[CTCAlignerConfig] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    book_key = book.lower().strip()
    if book_key not in BOOK_CONFIGS:
        raise ValueError(
            f"Unknown book: '{book}'. Supported: {list(BOOK_CONFIGS.keys())}"
        )

    config = BOOK_CONFIGS[book_key]
    num_chapters = config["chapters"]
    book_display = config["name"]

    if chapter is not None:
        if chapter < 1 or chapter > num_chapters:
            raise ValueError(
                f"Invalid chapter {chapter} for {book_display}. Valid: 1 to {num_chapters}"
            )
        chapters_to_run = [chapter]
    else:
        chapters_to_run = list(range(1, num_chapters + 1))

    SPLIT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    ALIGNMENTS_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_CSVS_DIR.mkdir(parents=True, exist_ok=True)

    if ctc_aligner is None:
        ctc_aligner = get_default_ctc_aligner(
            model_repo=model_repo,
            model_revision=model_revision,
            config=aligner_config,
        )

    records: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, str]] = []
    durations_sec: List[float] = []
    total_flagged_words = 0

    scope_str = (
        f"Chapter {chapter}" if chapter is not None else f"All {num_chapters} chapters"
    )
    print(f"\n==================================================")
    print(f"Starting Continuous CTC Realignment of {book_display} ({scope_str})")
    print(f"==================================================")

    for ch in chapters_to_run:
        ch_str = f"{ch:02d}"
        audio_path = AUDIO_SRC_DIR / f"{book_key}_{ch_str}.mp3"
        transcript_path = TRANSCRIPTS_DIR / f"{book_key}_{ch_str}.json"

        if not audio_path.exists():
            print(f"[Warning] Audio file not found: {audio_path}, skipping.")
            continue
        if not transcript_path.exists():
            print(f"[Warning] Transcript not found: {transcript_path}, skipping.")
            continue

        ch_out_dir = PRAAT_OUT_DIR / f"{book_key}_{ch_str}"
        print(
            f"\n>>> Realigning {book_display} Chapter {ch} with CTCSegmentationAligner ..."
        )
        res = align_chapter(
            audio_path=audio_path,
            transcript_path=transcript_path,
            output_dir=ch_out_dir,
            export_praat=export_praat,
            export_manifest=True,
            engine="ctc",
            ctc_aligner=ctc_aligner,
            model_path=model_repo,
            model_revision=model_revision,
            aligner_config=aligner_config,
        )

        audio_seg = (
            AudioSegment.from_file(str(audio_path))
            .set_frame_rate(16000)
            .set_channels(1)
        )
        num_verses_aligned = len(res.aligned_chunks)
        print(f"    Aligned {num_verses_aligned} verses.")

        ch_data = load_chapter_transcript(transcript_path)

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

            # Slicing unclipped verse audio using natural inter-verse boundary partition points
            start_ms = int(start_sec * 1000)
            end_ms = int(end_sec * 1000)
            if end_ms <= start_ms:
                print(
                    f"    [Warning] Skipping zero or negative duration audio slice for {verse_id}: [{start_sec}s - {end_sec}s]"
                )
            else:
                verse_audio = audio_seg[start_ms:end_ms]
                verse_audio.export(str(split_out_path), format="wav")

            # Load verse transcript text
            verse_info = ch_data.get(verse_id, {})
            cherokee_text = (
                verse_info.get("cherokee")
                or verse_info.get("text")
                or verse_info.get("syllabary", "")
            )
            phonetic_text = verse_info.get("phonetic", "")
            english_text = verse_info.get("english", "")

            # Use direct CTC trellis emissions (syncope- and intrusion-aware)
            emitted_sentence = (chunk.emitted_text or "").strip()

            words_list = [
                {
                    "word": w.word,
                    "start_sec": round(w.start_sec, 3),
                    "end_sec": round(w.end_sec, 3),
                    "confidence": w.confidence,
                    "min_char_confidence": w.min_char_confidence,
                    "flagged": w.flagged,
                    "emitted_word": w.emitted_word or "",
                }
                for w in chunk.words
            ]

            for w in chunk.words:
                if w.flagged:
                    total_flagged_words += 1

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
                "reconciled_phonetics": emitted_sentence,
                "asr_hypothesis": emitted_sentence,
                "cost": round(chunk.distance_score, 4),
                "words": words_list,
                "cherokee_syllabary": cherokee_text,
                "phonetic": phonetic_text,
                "english": english_text,
                "has_anomalies": bool(chunk.has_anomalies),
            }
            records.append(record)

            # Only export non-anomalous verses with valid emitted text to the training dataset.
            # Strict quality control: no fallback to phonetic text. If emitted_text is missing or anomalous, do not export.
            if not chunk.has_anomalies and emitted_sentence:
                csv_rows.append(
                    {
                        "path": f"cherokee_new_testament/split_audio/{split_filename}",
                        "sentence": emitted_sentence,
                    }
                )
            elif not emitted_sentence:
                print(
                    f"    [Missing Emission Filtered] Excluded verse {verse_id} from training CSV due to missing emitted text."
                )
            else:
                print(
                    f"    [Anomaly Filtered] Excluded verse {verse_id} from training CSV due to flagged word(s)."
                )

    # Merge or save per-book alignment JSON
    book_alignments_path = ALIGNMENTS_DIR / f"{book_key}_alignment_records.json"
    if chapter is not None and book_alignments_path.exists():
        try:
            with open(book_alignments_path, "r", encoding="utf-8") as f:
                existing_book_records = json.load(f)
            other_ch_records = [
                r for r in existing_book_records if r.get("chapter") != chapter
            ]
            saved_records = sorted(
                other_ch_records + records,
                key=lambda x: (x.get("chapter", 0), x.get("verse_idx", 0)),
            )
        except Exception:
            saved_records = records
    else:
        saved_records = records

    with open(book_alignments_path, "w", encoding="utf-8") as f:
        json.dump(saved_records, f, indent=2, ensure_ascii=False)
    print(
        f"\n[Artifact] Saved {len(saved_records)} alignment records to '{book_alignments_path}'"
    )

    # Export training CSV
    train_csv_path = TRAIN_CSVS_DIR / f"{book_key}.csv"
    if chapter is not None and train_csv_path.exists():
        try:
            with open(train_csv_path, "r", encoding="utf-8") as f:
                existing_csv = list(csv.DictReader(f))
            # Remove rows matching current chapter prefix
            ch_prefix = f"cherokee_new_testament/split_audio/{book_key}_{chapter:02d}_"
            other_csv_rows = [
                r for r in existing_csv if not r.get("path", "").startswith(ch_prefix)
            ]
            saved_csv_rows = sorted(
                other_csv_rows + csv_rows, key=lambda x: x.get("path", "")
            )
        except Exception:
            saved_csv_rows = csv_rows
    else:
        saved_csv_rows = csv_rows

    with open(train_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "sentence"])
        writer.writeheader()
        writer.writerows(saved_csv_rows)
    print(
        f"[Artifact] Saved training CSV with {len(saved_csv_rows)} rows to '{train_csv_path}'"
    )

    if durations_sec:
        total_sec = sum(durations_sec)
        mean_len = float(np.mean(durations_sec))
        median_len = float(np.median(durations_sec))
        print(f"\n--- {book_display} ({scope_str}) Alignment Summary ---")
        print(f"Total sliced segments: {len(durations_sec)}")
        print(f"Flagged anomaly words: {total_flagged_words}")
        print(f"Mean verse length    : {mean_len:.2f} seconds")
        print(f"Median verse length  : {median_len:.2f} seconds")
        print(
            f"Total audio duration : {total_sec:.2f} seconds ({total_sec/60:.2f} minutes / {total_sec/3600:.2f} hours)"
        )

    return records, csv_rows


def realign_all(
    chapter: Optional[int] = None,
    model_repo: str = DEFAULT_MODEL_REPO,
    model_revision: str = DEFAULT_REVISION,
    export_praat: bool = True,
    aligner_config: Optional[CTCAlignerConfig] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    ctc_aligner = get_default_ctc_aligner(
        model_repo=model_repo,
        model_revision=model_revision,
        config=aligner_config,
    )

    all_records: List[Dict[str, Any]] = []
    book_results: Dict[str, List[Dict[str, Any]]] = {}

    for book in ["mark", "matthew"]:
        records, _ = realign_book(
            book=book,
            chapter=chapter,
            ctc_aligner=ctc_aligner,
            model_repo=model_repo,
            model_revision=model_revision,
            export_praat=export_praat,
            aligner_config=aligner_config,
        )
        book_results[book] = records
        all_records.extend(records)

    # Save combined alignment records
    ALIGNMENTS_DIR.mkdir(parents=True, exist_ok=True)
    combined_out_path = ALIGNMENTS_DIR / "bible_alignment_records.json"
    if chapter is not None and combined_out_path.exists():
        try:
            with open(combined_out_path, "r", encoding="utf-8") as f:
                existing_comb = json.load(f)
            other_comb = [r for r in existing_comb if r.get("chapter") != chapter]
            final_comb = sorted(
                other_comb + all_records,
                key=lambda x: (
                    x.get("book", ""),
                    x.get("chapter", 0),
                    x.get("verse_idx", 0),
                ),
            )
        except Exception:
            final_comb = all_records
    else:
        final_comb = all_records

    with open(combined_out_path, "w", encoding="utf-8") as f:
        json.dump(final_comb, f, indent=2, ensure_ascii=False)
    print(
        f"\n[Artifact] Saved {len(final_comb)} combined alignment records to '{combined_out_path}'"
    )

    return book_results


def main():
    parser = argparse.ArgumentParser(
        description="Realign Cherokee New Testament books with continuous CTC segmentation aligner."
    )
    parser.add_argument(
        "--book",
        choices=["mark", "matthew", "all"],
        default="all",
        help="Book to realign (default: all)",
    )
    parser.add_argument(
        "--chapter",
        "-c",
        type=int,
        default=None,
        help="Specific chapter number to realign (default: all chapters)",
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
        "--syncope-penalty",
        type=float,
        default=DEFAULT_SYNCOPE_PENALTY,
        help=f"CTC segmentation syncope penalty for vowel deletion (default: {DEFAULT_SYNCOPE_PENALTY})",
    )
    parser.add_argument(
        "--intrusive-penalty",
        type=float,
        default=DEFAULT_INTRUSIVE_PENALTY,
        help=f"CTC segmentation intrusive penalty for h/' insertion (default: {DEFAULT_INTRUSIVE_PENALTY})",
    )
    parser.add_argument(
        "--flag-min-confidence",
        type=float,
        default=0.01,
        help="Minimum word confidence threshold for anomaly flagging (default: 0.01)",
    )
    parser.add_argument(
        "--boundary-pad",
        type=float,
        default=0.1,
        help="Safety padding (in seconds) applied to verse boundaries (default: 0.1s)",
    )
    parser.add_argument(
        "--no-praat",
        action="store_true",
        default=False,
        help="Skip Praat TextGrid export",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Directory for caching CTC logits (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Disable disk caching for CTC logits",
    )

    args = parser.parse_args()

    aligner_config = CTCAlignerConfig(
        syncope_penalty=args.syncope_penalty,
        intrusive_penalty=args.intrusive_penalty,
        flag_min_confidence=args.flag_min_confidence,
        boundary_pad_sec=args.boundary_pad,
        cache=not args.no_cache,
        cache_dir=args.cache_dir,
    )

    if args.book == "all":
        realign_all(
            chapter=args.chapter,
            model_repo=args.model_repo,
            model_revision=args.model_revision,
            export_praat=not args.no_praat,
            aligner_config=aligner_config,
        )
    else:
        records, _ = realign_book(
            book=args.book,
            chapter=args.chapter,
            model_repo=args.model_repo,
            model_revision=args.model_revision,
            export_praat=not args.no_praat,
            aligner_config=aligner_config,
        )
        # Also update combined if single book is run
        ALIGNMENTS_DIR.mkdir(parents=True, exist_ok=True)
        combined_path = ALIGNMENTS_DIR / "bible_alignment_records.json"
        existing_records = []
        if combined_path.exists():
            try:
                with open(combined_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if args.chapter is not None:
                    existing_records = [
                        r
                        for r in existing
                        if not (
                            r.get("book") == args.book
                            and r.get("chapter") == args.chapter
                        )
                    ]
                else:
                    existing_records = [
                        r for r in existing if r.get("book") != args.book
                    ]
            except Exception:
                existing_records = []
        existing_records.extend(records)
        existing_records.sort(
            key=lambda x: (
                x.get("book", ""),
                x.get("chapter", 0),
                x.get("verse_idx", 0),
            )
        )
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(existing_records, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
