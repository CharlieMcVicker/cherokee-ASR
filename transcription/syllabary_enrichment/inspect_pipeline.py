# -*- coding: utf-8 -*-
"""
inspect_pipeline.py

Diagnostic CLI tool for step-by-step visualization of Cherokee Syllabary Phonetic Reconciliation.

Visual steps printed:
1. Syllabary Input & Base Transliteration
2. ASR Emitted Text
3. Character / Syllable Aligned Slices
4. Syllable Rule Action & Output
5. Target vs Reconciled String Diff & CER
"""

import os
import sys
import json
import argparse
import difflib
from typing import List, Dict, Any, Optional

from transcription.syllabary_enrichment.alignment_engine import (
    get_base_transliteration,
    align_character_syllable_detailed,
    CHEROKEE_SYLLABARY_MAP,
)
from transcription.syllabary_enrichment.enrich_syllabary import (
    reconcile_phonetics,
    _enrich_single_syllable,
    _get_base_syllable,
)
from transcription.syllabary_enrichment.evaluate_reconciliation import calculate_cer
from transcription.syllabary_enrichment.batch_inference_aligner import (
    process_batch_inference_and_alignment,
)


def format_diff(ref: str, hyp: str) -> str:
    """Generate human-readable character inline diff representation."""
    matcher = difflib.SequenceMatcher(None, ref, hyp)
    out = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            out.append(ref[i1:i2])
        elif tag == "replace":
            out.append(f"[-{ref[i1:i2]}-][+{hyp[j1:j2]}+]")
        elif tag == "delete":
            out.append(f"[-{ref[i1:i2]}-]")
        elif tag == "insert":
            out.append(f"[+{hyp[j1:j2]}+]")
    return "".join(out)


def inspect_record(rec: Dict[str, Any]) -> None:
    """Print detailed visual step-by-step pipeline transformations for a single record."""
    rec_id = (
        rec.get("id")
        or rec.get("audio_path")
        or rec.get("audio_filepath")
        or "Unknown Record"
    )
    syllabary_text = rec.get("syllabary_text") or rec.get("syllabary") or ""
    emitted_text = rec.get("emitted_text", "")
    target_text = rec.get("target") or rec.get("target_phonetics") or ""
    split = rec.get("split", "unknown")

    base_trans = get_base_transliteration(syllabary_text)

    # Perform detailed alignment dynamically or use cached detailed alignments if available
    detailed_alignments = align_character_syllable_detailed(
        syllabary_text, emitted_text
    )
    aligned_pairs = [(a.syllabary_char, a.emitted_text) for a in detailed_alignments]

    reconciled_text = reconcile_phonetics(
        syllabary_text=syllabary_text,
        base_transliteration=base_trans,
        emitted_text=emitted_text,
        aligned_pairs=aligned_pairs,
    )

    raw_cer = calculate_cer(target_text, emitted_text)
    rec_cer = calculate_cer(target_text, reconciled_text)

    print("=" * 80)
    print(f" PIPELINE INSPECTION: {rec_id} (Split: {split})")
    print("=" * 80)

    # Step 1: Syllabary Input & Base Transliteration
    print("\n--- STEP 1: Syllabary Input & Base Transliteration ---")
    print(f"  Syllabary Input:       {syllabary_text}")
    print(f"  Base Transliteration:  {base_trans}")

    # Step 2: ASR Emitted Text
    print("\n--- STEP 2: ASR Emitted Text ---")
    print(f"  ASR Emitted Text:      {emitted_text}")

    # Step 3: Character / Syllable Aligned Slices
    print("\n--- STEP 3: Character / Syllable Aligned Slices ---")
    slice_reprs = []
    for item in detailed_alignments:
        char = item.syllabary_char
        emitted_sub = item.emitted_text
        slice_reprs.append(f"('{char}'[{item.base_phonetic}] -> '{emitted_sub}')")
    print("  Aligned Slices: " + ", ".join(slice_reprs))

    # Step 4: Syllable Rule Action & Output
    print("\n--- STEP 4: Syllable Rule Action & Output ---")
    for item in detailed_alignments:
        char = item.syllabary_char
        emitted_sub = item.emitted_text
        base_phon = item.base_phonetic
        if char not in CHEROKEE_SYLLABARY_MAP:
            action = f"Pass-through non-syllabary symbol -> '{emitted_sub if emitted_sub != '' else char}'"
            out_phon = emitted_sub if emitted_sub != "" else char
        else:
            emitted_clean = emitted_sub.strip()
            if not emitted_clean:
                action = f"Vowel Syncopation / Drop check for base '{base_phon}' with empty ASR slice"
                out_phon = (
                    base_phon[:-1]
                    if len(base_phon) > 1 and base_phon[-1] in "aeiouvAEIOUV"
                    else base_phon
                )
            else:
                out_phon = _enrich_single_syllable(base_phon, emitted_clean)
                if out_phon == base_phon:
                    action = f"Base unchanged ({base_phon})"
                else:
                    action = (
                        f"Enriched ({base_phon} + ASR '{emitted_clean}' -> {out_phon})"
                    )
        print(
            f"  Char: {char:<3} | Base: {base_phon:<5} | ASR: {emitted_sub:<6} | Output: {out_phon:<6} | Action: {action}"
        )

    # Step 5: Target vs Reconciled String Diff & CER
    print("\n--- STEP 5: Target vs Reconciled String Diff & CER ---")
    print(f"  Target Phonetics:      {target_text}")
    print(f"  Reconciled Phonetics:  {reconciled_text}")
    print(f"  Inline Diff:           {format_diff(target_text, reconciled_text)}")
    print(f"  Raw ASR CER:           {raw_cer:.4f}")
    print(f"  Reconciled CER:        {rec_cer:.4f}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Cherokee Syllabary Enrichment Pipeline Inspector"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="training_data/processed/split_audio_syl_target.csv",
        help="Path to manifest CSV",
    )
    parser.add_argument(
        "--output-cache",
        type=str,
        default="training_data/processed/syllabary_enrichment_cache.json",
        help="Path to alignment cache JSON",
    )
    parser.add_argument(
        "--record-id",
        type=str,
        default=None,
        help="Inspect a specific record by ID or audio path filename substring",
    )
    parser.add_argument(
        "--top-errors",
        type=int,
        default=0,
        help="Inspect top N records with highest Reconciled CER",
    )
    args = parser.parse_args()

    records = process_batch_inference_and_alignment(
        manifest_path=args.manifest,
        output_cache_path=args.output_cache,
        force_recompute=False,
    )

    if args.record_id:
        matched = [
            r
            for r in records
            if args.record_id in r.get("audio_path", "")
            or args.record_id in r.get("audio_filepath", "")
            or args.record_id in r.get("id", "")
        ]
        if not matched:
            print(f"No record found matching '--record-id {args.record_id}'")
            sys.exit(1)
        for rec in matched:
            inspect_record(rec)
    elif args.top_errors > 0:
        # Calculate CER for all records and sort descending by Reconciled CER
        scored = []
        for rec in records:
            syllabary_text = rec.get("syllabary_text") or rec.get("syllabary") or ""
            emitted_text = rec.get("emitted_text", "")
            target_text = rec.get("target") or rec.get("target_phonetics") or ""
            base_trans = get_base_transliteration(syllabary_text)
            aligned_pairs = [
                (
                    (item[0], item[1])
                    if isinstance(item, (list, tuple))
                    else (item.syllabary_char, item.emitted_text)
                )
                for item in rec.get("aligned_pairs", [])
            ]
            rec_phon = reconcile_phonetics(
                syllabary_text, base_trans, emitted_text, aligned_pairs
            )
            cer_val = calculate_cer(target_text, rec_phon)
            scored.append((cer_val, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        print(f"Displaying top {args.top_errors} error records:")
        for cer_val, rec in scored[: args.top_errors]:
            inspect_record(rec)
    else:
        # Default: inspect first record
        if records:
            inspect_record(records[0])


if __name__ == "__main__":
    main()
