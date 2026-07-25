# -*- coding: utf-8 -*-
"""
evaluate_reconciliation.py

Orchestrates end-to-end evaluation of Cherokee Syllabary phonetic reconciliation:
1. Loads or computes batch alignment manifest using process_batch_inference_and_alignment capabilities / disk cache.
2. Runs reconcile_phonetics() on every record to derive reconciled phonetics.
3. Computes Raw CER (emitted_text vs target GT phonetics), Reconciled CER (reconciled_phonetics vs target GT phonetics),
   and relative improvement Delta CER.
4. Outputs formatted CLI summary metrics table broken down by split (train, validation, test, overall).
5. Saves eval_results.json artifact containing per-line predictions, target strings, and CER scores.
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional

from jiwer import cer as jiwer_cer

from transcription.syllabary_enrichment.batch_inference_aligner import (
    process_batch_inference_and_alignment,
)
from transcription.syllabary_enrichment.enrich_syllabary import reconcile_phonetics
from transcription.syllabary_enrichment.alignment_engine import get_base_transliteration


def safe_text(text: Optional[str]) -> str:
    """Ensure non-empty string for jiwer metrics calculation."""
    if text is None:
        return " "
    st = str(text).strip()
    return st if st else " "


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate using jiwer."""
    ref_safe = safe_text(reference)
    hyp_safe = safe_text(hypothesis)
    return float(jiwer_cer(ref_safe, hyp_safe))


def calculate_relative_improvement(raw_cer: float, reconciled_cer: float) -> float:
    """
    Calculate relative improvement delta CER percentage:
    ((raw_cer - reconciled_cer) / raw_cer) * 100% if raw_cer > 0 else 0.0.
    """
    if raw_cer <= 1e-9:
        return 0.0
    return float(((raw_cer - reconciled_cer) / raw_cer) * 100.0)


def run_evaluation(
    records: List[Dict[str, Any]], output_artifact_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates phonetic reconciliation over records and returns detailed metrics and per-line results.

    Each record should contain:
      - syllabary_text: Syllabary input string.
      - target_text / target_phonetics / target / ground_truth / base_transliteration: Target GT phonetics.
      - emitted_text: Raw ASR prediction string.
      - aligned_pairs: List of aligned pairs or SyllableAlignment objects.
      - split (optional): Dataset split ('train', 'validation', 'test', etc. - defaults to 'test' or 'unassigned').
    """
    eval_records = []
    split_records: Dict[str, List[Dict[str, Any]]] = {}

    for idx, rec in enumerate(records):
        syllabary_text = rec.get("syllabary_text") or rec.get("cherokee_text") or ""
        emitted_text = rec.get("emitted_text", "")
        aligned_pairs = rec.get("aligned_pairs", [])

        base_trans = (
            rec.get("base_transliteration")
            or rec.get("base_text")
            or get_base_transliteration(syllabary_text)
        )

        target_phonetics = (
            rec.get("target_phonetics")
            or rec.get("target_text")
            or rec.get("target")
            or rec.get("ground_truth")
            or base_trans
        )

        split = str(rec.get("split", "test")).strip().lower()
        if not split:
            split = "test"

        # Reconcile phonetics using phonetic rule merger engine
        reconciled_phonetics = reconcile_phonetics(
            syllabary_text=syllabary_text,
            base_transliteration=base_trans,
            emitted_text=emitted_text,
            aligned_pairs=aligned_pairs,
        )

        raw_cer = calculate_cer(target_phonetics, emitted_text)
        reconciled_cer = calculate_cer(target_phonetics, reconciled_phonetics)
        delta_cer = calculate_relative_improvement(raw_cer, reconciled_cer)

        eval_item = {
            "record_id": rec.get("id", rec.get("audio_filepath", f"record_{idx}")),
            "split": split,
            "syllabary_text": syllabary_text,
            "base_transliteration": base_trans,
            "emitted_text": emitted_text,
            "reconciled_phonetics": reconciled_phonetics,
            "target_phonetics": target_phonetics,
            "raw_cer": raw_cer,
            "reconciled_cer": reconciled_cer,
            "delta_cer": delta_cer,
        }

        eval_records.append(eval_item)
        if split not in split_records:
            split_records[split] = []
        split_records[split].append(eval_item)

    # Calculate metrics by split & overall
    metrics_by_split = {}

    # Process known splits plus any extra found in data
    all_splits = list(split_records.keys())
    for s in ["train", "validation", "test"]:
        if s not in all_splits:
            all_splits.append(s)

    for s in all_splits:
        items = split_records.get(s, [])
        if items:
            targets = [it["target_phonetics"] for it in items]
            emitteds = [it["emitted_text"] for it in items]
            reconcileds = [it["reconciled_phonetics"] for it in items]

            split_raw_cer = float(
                jiwer_cer(
                    [safe_text(t) for t in targets], [safe_text(e) for e in emitteds]
                )
            )
            split_rec_cer = float(
                jiwer_cer(
                    [safe_text(t) for t in targets], [safe_text(r) for r in reconcileds]
                )
            )
            split_delta = calculate_relative_improvement(split_raw_cer, split_rec_cer)

            metrics_by_split[s] = {
                "count": len(items),
                "raw_cer": split_raw_cer,
                "reconciled_cer": split_rec_cer,
                "delta_cer": split_delta,
            }
        else:
            metrics_by_split[s] = {
                "count": 0,
                "raw_cer": 0.0,
                "reconciled_cer": 0.0,
                "delta_cer": 0.0,
            }

    # Overall metrics across all records
    if eval_records:
        all_targets = [it["target_phonetics"] for it in eval_records]
        all_emitteds = [it["emitted_text"] for it in eval_records]
        all_reconcileds = [it["reconciled_phonetics"] for it in eval_records]

        overall_raw_cer = float(
            jiwer_cer(
                [safe_text(t) for t in all_targets],
                [safe_text(e) for e in all_emitteds],
            )
        )
        overall_rec_cer = float(
            jiwer_cer(
                [safe_text(t) for t in all_targets],
                [safe_text(r) for r in all_reconcileds],
            )
        )
        overall_delta = calculate_relative_improvement(overall_raw_cer, overall_rec_cer)

        metrics_by_split["overall"] = {
            "count": len(eval_records),
            "raw_cer": overall_raw_cer,
            "reconciled_cer": overall_rec_cer,
            "delta_cer": overall_delta,
        }
    else:
        metrics_by_split["overall"] = {
            "count": 0,
            "raw_cer": 0.0,
            "reconciled_cer": 0.0,
            "delta_cer": 0.0,
        }

    results = {
        "summary": metrics_by_split,
        "records": eval_records,
    }

    if output_artifact_path:
        os.makedirs(
            os.path.dirname(os.path.abspath(output_artifact_path)), exist_ok=True
        )
        with open(output_artifact_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def print_summary_table(summary: Dict[str, Dict[str, Any]]) -> None:
    """Print formatted CLI summary metrics table broken down by split."""
    header = f"{'Split':<15} | {'Count':<8} | {'Raw CER':<10} | {'Reconciled CER':<15} | {'Δ CER (%)':<10}"
    divider = "-" * len(header)

    print("\n" + "=" * len(header))
    print("PHONETIC RECONCILIATION EVALUATION SUMMARY")
    print("=" * len(header))
    print(header)
    print(divider)

    ordered_splits = ["train", "validation", "test"]
    for s in summary.keys():
        if s not in ordered_splits and s != "overall":
            ordered_splits.append(s)
    ordered_splits.append("overall")

    for s in ordered_splits:
        if s not in summary:
            continue
        data = summary[s]
        count = data["count"]
        raw_cer = f"{data['raw_cer']:.4f}"
        rec_cer = f"{data['reconciled_cer']:.4f}"
        delta = f"{data['delta_cer']:+.2f}%"
        print(f"{s:<15} | {count:<8} | {raw_cer:<10} | {rec_cer:<15} | {delta:<10}")

    print("=" * len(header) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Cherokee Syllabary Phonetic Reconciliation Evaluation Framework"
    )
    parser.add_argument(
        "manifest_path",
        type=str,
        help="Path to input dataset manifest JSON or JSONL file.",
    )
    parser.add_argument(
        "--output-cache",
        type=str,
        default="data/results/aligned_manifest_cache.json",
        help="Path to aligned manifest disk cache file (default: data/results/aligned_manifest_cache.json).",
    )
    parser.add_argument(
        "--output-artifact",
        type=str,
        default="data/results/eval_results.json",
        help="Path to save evaluation artifact JSON (default: data/results/eval_results.json).",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        default=False,
        help="Force recomputation of batch alignment manifest (default: False).",
    )

    args = parser.parse_args()

    # Load or compute batch alignment manifest
    records = process_batch_inference_and_alignment(
        manifest_path=args.manifest_path,
        output_cache_path=args.output_cache,
        force_recompute=args.force_recompute,
    )

    # Run evaluation
    results = run_evaluation(records, output_artifact_path=args.output_artifact)

    # Print CLI summary table
    print_summary_table(results["summary"])
    print(f"Evaluation results saved to {args.output_artifact}")


if __name__ == "__main__":
    main()
