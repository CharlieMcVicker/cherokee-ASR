#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_ctc_segmentation_100_verses.py

Realigns 100 Bible verses using the new ctc-segmentation engine with syncope handling,
compares them against the existing baseline alignment system, and produces detailed
phonetic reconciliation diffs and summary metrics.
"""

import difflib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from pydub import AudioSegment
import torch

# Ensure repository root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from transcription.alignment.ctc_aligner import CTCSegmentationAligner
from transcription.alignment.normalizers import normalize_syllabary_for_alignment
from transcription.models.asr_model import CherokeeASRModel
from transcription.new_testament.pipeline import (
    load_chapter_transcript,
)

DEFAULT_MODEL_REPO = "charliemcvicker/length-only-20260704-155307-asr-cherokee-colon"
DEFAULT_REVISION = "76e62140955f4738abdab345ea34068b02d8d2a2"


def compute_char_diff(old_str: str, new_str: str) -> List[str]:
    """Generates human-readable diff descriptions between two phonetic strings."""
    diffs = []
    matcher = difflib.SequenceMatcher(None, old_str, new_str)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            diffs.append(f"'{old_str[i1:i2]}' -> '{new_str[j1:j2]}'")
        elif tag == "delete":
            diffs.append(f"dropped '{old_str[i1:i2]}'")
        elif tag == "insert":
            diffs.append(f"inserted '{new_str[j1:j2]}'")
    return diffs


def categorize_diff(old_rec: str, new_rec: str) -> List[str]:
    """Categorizes the nature of phonetic changes between old and new reconciliation."""
    categories = set()
    old_words = old_rec.split()
    new_words = new_rec.split()

    if len(old_words) != len(new_words):
        categories.add("word_boundary_shift")

    # Check for dropped vowels (syncope)
    vowels = set("aeiouv")
    for ow, nw in zip(old_words, new_words):
        if len(nw) < len(ow):
            dropped_chars = set(ow) - set(nw)
            if dropped_chars.intersection(vowels):
                categories.add("vowel_syncope_dropped")
        elif len(nw) > len(ow):
            inserted_chars = set(nw) - set(ow)
            if inserted_chars.intersection(vowels):
                categories.add("vowel_restored")

        if "h" in ow and "h" not in nw:
            categories.add("aspiration_removed")
        elif "h" not in ow and "h" in nw:
            categories.add("aspiration_added")

        if "'" in ow and "'" not in nw:
            categories.add("glottal_stop_removed")
        elif "'" not in ow and "'" in nw:
            categories.add("glottal_stop_added")

    if not categories and old_rec != new_rec:
        categories.add("orthographic_spelling_variation")

    return sorted(list(categories))


def run_benchmark(
    target_verse_count: int = 100,
    model_repo: str = DEFAULT_MODEL_REPO,
    model_revision: str = DEFAULT_REVISION,
    output_json_path: Optional[Path] = None,
    cache: bool = True,
    cache_dir: Optional[Path] = None,
    syncope_penalty: float = 6.0,
    intrusive_penalties: Optional[Any] = None,
    intrusive_min_logprobs: Optional[Any] = None,
    flag_min_char_confidence: float = 0.005,
    enforce_phonotactics: bool = True,
) -> Dict[str, Any]:
    print(f"Loading ASR Model: {model_repo} (rev: {model_revision})...")
    token = os.environ.get("HF_TOKEN", None)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    asr_model = CherokeeASRModel.from_pretrained(
        path_or_repo=model_repo,
        revision=model_revision,
        device=device,
        token=token,
    )

    aligner = CTCSegmentationAligner(
        model=asr_model,
        cache=cache,
        cache_dir=cache_dir,
        syncope_penalty=syncope_penalty,
        intrusive_penalties=intrusive_penalties,
        intrusive_min_logprobs=intrusive_min_logprobs,
        flag_min_char_confidence=flag_min_char_confidence,
        enforce_phonotactics=enforce_phonotactics,
    )

    # Load existing baseline alignment records
    baseline_records_path = (
        BASE_DIR
        / "cherokee_new_testament"
        / "alignments"
        / "mark_alignment_records.json"
    )
    if not baseline_records_path.exists():
        raise FileNotFoundError(
            f"Baseline alignment records not found: {baseline_records_path}"
        )

    with open(baseline_records_path, "r", encoding="utf-8") as f:
        all_baseline_records: List[Dict[str, Any]] = json.load(f)

    # Target 100 verses
    verses_to_align = all_baseline_records[:target_verse_count]
    print(f"Selected {len(verses_to_align)} verses for benchmark realignment.")

    results: List[Dict[str, Any]] = []
    identical_count = 0
    different_count = 0
    category_counts: Dict[str, int] = {}
    total_elapsed = 0.0

    print("\nStarting realignment & reconciliation with ctc-segmentation engine...")
    t_start_all = time.time()

    for idx, base_rec in enumerate(verses_to_align):
        verse_id = base_rec["verse_id"]
        rel_audio_path = base_rec["audio_path"]
        abs_audio_path = BASE_DIR / rel_audio_path

        if not abs_audio_path.exists():
            print(f"Warning: audio file {abs_audio_path} not found, skipping.")
            continue

        cherokee_text = base_rec["reference_sentence"]
        phonetic_text = base_rec["phonetic"]

        # Run intra-verse CTC segmentation
        t0 = time.time()
        aligned_chunk = aligner.align_verse_slice(
            audio_input=abs_audio_path,
            chunk_id=verse_id,
            phonetic_text=phonetic_text,
            syllabary_text=cherokee_text,
        )
        elapsed = time.time() - t0
        total_elapsed += elapsed

        # Use emitted hypothesis directly from backtracked CTC trellis path
        new_asr_hyp = (aligned_chunk.emitted_text or "").strip()
        new_reconciled_clean = new_asr_hyp

        old_reconciled = base_rec["reconciled_phonetics"].strip()

        is_different = old_reconciled != new_reconciled_clean
        if is_different:
            different_count += 1
        else:
            identical_count += 1

        diff_details = (
            compute_char_diff(old_reconciled, new_reconciled_clean)
            if is_different
            else []
        )
        diff_categories = (
            categorize_diff(old_reconciled, new_reconciled_clean)
            if is_different
            else []
        )

        for cat in diff_categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1

        res_item = {
            "verse_id": verse_id,
            "chapter": base_rec["chapter"],
            "verse_idx": base_rec["verse_idx"],
            "audio_path": rel_audio_path,
            "duration_sec": base_rec["duration_sec"],
            "reference_syllabary": cherokee_text,
            "phonetic_citation": phonetic_text,
            "old_asr_hypothesis": base_rec["asr_hypothesis"],
            "new_asr_hypothesis": new_asr_hyp,
            "old_reconciled_phonetics": old_reconciled,
            "new_reconciled_phonetics": new_reconciled_clean,
            "is_different": bool(is_different),
            "diff_details": diff_details,
            "diff_categories": diff_categories,
            "words_ctc_aligned": [
                {
                    "word": w.word,
                    "start_sec": float(w.start_sec),
                    "end_sec": float(w.end_sec),
                    "duration_sec": round(float(w.end_sec - w.start_sec), 3),
                    "emitted_word": w.emitted_word,
                    "confidence": float(w.confidence),
                    "flagged": bool(w.flagged),
                }
                for w in aligned_chunk.words
            ],
            "flagged_words": [
                {
                    "word": w.word,
                    "emitted_word": w.emitted_word,
                    "confidence": float(w.confidence),
                    "duration_sec": round(float(w.end_sec - w.start_sec), 3),
                }
                for w in aligned_chunk.words
                if w.flagged
            ],
            "has_anomalies": bool(aligned_chunk.has_anomalies),
        }
        results.append(res_item)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(verses_to_align):
            print(
                f"  [{idx + 1:3d}/{len(verses_to_align)}] Realigned... ({different_count} diffs so far)"
            )

    total_time = time.time() - t_start_all
    anomalous_verses = [
        {
            "verse_id": r["verse_id"],
            "reference_syllabary": r["reference_syllabary"],
            "flagged_words": r["flagged_words"],
        }
        for r in results
        if r["has_anomalies"]
    ]
    total_flagged_words = sum(len(r["flagged_words"]) for r in results)

    summary = {
        "total_verses": len(results),
        "identical_reconciliations": identical_count,
        "different_reconciliations": different_count,
        "diff_percentage": round((different_count / max(1, len(results))) * 100, 2),
        "total_flagged_words": total_flagged_words,
        "verses_with_anomalies": len(anomalous_verses),
        "anomalous_verses": anomalous_verses,
        "total_processing_time_sec": round(total_time, 2),
        "mean_verse_time_ms": round((total_elapsed / max(1, len(results))) * 1000, 2),
        "category_counts": category_counts,
        "model_repo": model_repo,
        "model_revision": model_revision,
        "results": results,
    }

    if output_json_path is None:
        out_dir = BASE_DIR / "runs" / "evaluation"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_json_path = out_dir / "ctc_segmentation_100_verses_comparison.json"

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[Artifact] Saved comparison report to: {output_json_path}")
    print("\n--- Summary ---")
    print(f"Total Verses Realigned : {summary['total_verses']}")
    print(
        f"Identical              : {summary['identical_reconciliations']} ({100 - summary['diff_percentage']}%)"
    )
    print(
        f"Different Reconciled   : {summary['different_reconciliations']} ({summary['diff_percentage']}%)"
    )
    print(
        f"Verses with Anomalies  : {summary['verses_with_anomalies']} (Total flagged words: {total_flagged_words})"
    )
    print(f"Mean Verse Alignment   : {summary['mean_verse_time_ms']:.1f} ms/verse")
    print("Difference Categories  :")
    for cat, count in category_counts.items():
        print(f"  - {cat:30s}: {count}")

    if anomalous_verses:
        print("\n--- Suspected Transcript Anomalies / Typos Detected ---")
        for av in anomalous_verses[:10]:
            print(f"  * Verse {av['verse_id']}:")
            for fw in av["flagged_words"]:
                print(
                    f"      - Transcript word: '{fw['word']}' -> Emitted: '{fw['emitted_word']}' (conf={fw['confidence']}, dur={fw['duration_sec']}s)"
                )

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark CTC segmentation on Bible verses with disk-cached emissions."
    )
    parser.add_argument(
        "--target-verse-count",
        "-n",
        type=int,
        default=100,
        help="Number of verses to benchmark (default: 100)",
    )
    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable acoustic emissions disk caching (default: True)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Custom directory for caching emissions",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to output comparison report JSON",
    )
    args = parser.parse_args()

    run_benchmark(
        target_verse_count=args.target_verse_count,
        cache=args.cache,
        cache_dir=args.cache_dir,
        output_json_path=args.output_json,
    )
