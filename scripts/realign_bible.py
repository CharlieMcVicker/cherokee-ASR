#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/realign_bible.py

Thin driver: realigns Cherokee New Testament books (Mark, Matthew) using
align_chapter() from ScripturePipeline with CTC segmentation and verse slicing.

The realign_book() function is retained for integration tests and programmatic use.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydub import AudioSegment

from digohwelisgi.pipelines.scripture import align_chapter as _align_chapter_pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
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

BOOK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "mark": {"chapters": 16},
    "matthew": {"chapters": 28},
}


def align_chapter(
    audio_path: Path, transcript_path: Path, output_dir: Path, **kwargs: Any
):
    """Module-level shim so tests can patch scripts.realign_bible.align_chapter."""
    return _align_chapter_pipeline(
        audio_path=audio_path,
        transcript_path=transcript_path,
        output_dir=output_dir,
        **kwargs,
    )


def _resolve_audio(audio_src_dir: Path, book: str, chapter: int) -> Optional[Path]:
    """Resolve chapter audio file, accepting any audio extension."""
    stem = f"{book}_{chapter:02d}"
    for ext in (".wav", ".mp3", ".flac", ".ogg"):
        p = audio_src_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def realign_book(
    book: str,
    chapter: Optional[int] = None,
    ctc_aligner: Optional[Any] = None,
    model_repo: str = DEFAULT_MODEL_REPO,
    model_revision: str = DEFAULT_REVISION,
    export_praat: bool = True,
    aligner_config: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Align one or all chapters of a NT book, writing records and training CSV rows."""
    from digohwelisgi.alignment.models import CTCAlignerConfig

    book_key = book.lower().strip()
    if book_key not in BOOK_CONFIGS:
        raise ValueError(
            f"Unknown book: '{book}'. Supported: {list(BOOK_CONFIGS.keys())}"
        )

    num_chapters = BOOK_CONFIGS[book_key]["chapters"]
    if aligner_config is None:
        aligner_config = CTCAlignerConfig(cache_dir=DEFAULT_CACHE_DIR)

    chapters = [chapter] if chapter is not None else range(1, num_chapters + 1)
    all_records: List[Dict[str, Any]] = []
    all_csv_rows: List[Dict[str, str]] = []

    for ch in chapters:
        audio_path = _resolve_audio(AUDIO_SRC_DIR, book_key, ch)
        transcript_path = TRANSCRIPTS_DIR / f"{book_key}_{ch:02d}.json"

        if audio_path is None or not transcript_path.exists():
            continue

        output_dir = PRAAT_OUT_DIR / f"{book_key}_{ch:02d}"
        output_dir.mkdir(parents=True, exist_ok=True)

        alignment = align_chapter(
            audio_path=audio_path,
            transcript_path=transcript_path,
            output_dir=output_dir,
            ctc_aligner=ctc_aligner,
            model_path=model_repo,
            model_revision=model_revision,
            export_praat=export_praat,
            aligner_config=aligner_config,
        )

        audio_seg = AudioSegment.from_file(str(audio_path))

        for idx, ac in enumerate(alignment.aligned_chunks):
            verse_id = ac.chunk_id
            emitted = (ac.emitted_text or "").strip()
            reconciled = getattr(ac, "reconciled_phonetics", "") or ""
            words = ac.words or []
            has_anomalies = (
                any(getattr(w, "flagged", False) for w in words) or not emitted
            )

            wav_filename = f"{book_key}_{ch:02d}_{idx + 1:02d}.wav"
            wav_path = SPLIT_AUDIO_DIR / wav_filename

            record: Dict[str, Any] = {
                "book": book_key,
                "chapter": ch,
                "verse_id": verse_id,
                "emitted_text": emitted,
                "reconciled_phonetics": reconciled,
                "has_anomalies": has_anomalies,
                "start_sec": ac.start_sec,
                "end_sec": ac.end_sec,
            }
            all_records.append(record)

            if not has_anomalies:
                start_ms = int((ac.start_sec or 0.0) * 1000)
                end_ms = int((ac.end_sec or 0.0) * 1000)
                SPLIT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
                audio_seg[start_ms:end_ms].export(str(wav_path), format="wav")
                all_csv_rows.append(
                    {
                        "path": str(wav_path),
                        "sentence": reconciled or emitted,
                        "verse_id": verse_id,
                    }
                )

    ALIGNMENTS_DIR.mkdir(parents=True, exist_ok=True)
    book_records_path = ALIGNMENTS_DIR / f"{book_key}_alignment_records.json"
    with open(book_records_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    TRAIN_CSVS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TRAIN_CSVS_DIR / f"{book_key}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "sentence", "verse_id"])
        writer.writeheader()
        writer.writerows(all_csv_rows)

    return all_records, all_csv_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Realign Cherokee NT books with ScripturePipeline."
    )
    parser.add_argument("--book", choices=["mark", "matthew", "all"], default="all")
    parser.add_argument("--chapter", "-c", type=int, default=None)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--model-revision", default=DEFAULT_REVISION)
    parser.add_argument("--no-praat", action="store_true", default=False)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--no-cache", action="store_true", default=False)
    args = parser.parse_args()

    from digohwelisgi.alignment.models import CTCAlignerConfig

    config = CTCAlignerConfig(cache=not args.no_cache, cache_dir=args.cache_dir)
    books = BOOK_CONFIGS if args.book == "all" else {args.book: BOOK_CONFIGS[args.book]}
    for book in books:
        realign_book(
            book=book,
            chapter=args.chapter,
            model_repo=args.model_repo,
            model_revision=args.model_revision,
            export_praat=not args.no_praat,
            aligner_config=config,
        )


if __name__ == "__main__":
    main()
