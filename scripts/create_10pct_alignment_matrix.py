#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/create_10pct_alignment_matrix.py

Builds a dedicated, empirical confusion matrix and substitution cost JSON artifact
specifically calibrated to the ~10% operational Character Error Rate (CER) typical
of real-world Cherokee text-to-speech alignment.
"""

from __future__ import annotations

import os
import sys

# Auto-reexec with cherokee-asr conda environment if executed with an unconfigured python
try:
    import numpy as np
    import torch
except ImportError:
    for cand in [
        "/opt/homebrew/Caskroom/miniconda/base/envs/cherokee-asr/bin/python",
        os.path.expanduser("~/miniconda3/envs/cherokee-asr/bin/python"),
        os.path.expanduser("~/anaconda3/envs/cherokee-asr/bin/python"),
    ]:
        if os.path.exists(cand) and sys.executable != cand:
            os.execv(cand, [cand] + sys.argv)
    raise

import argparse
import json
import logging
from pathlib import Path

from digohwelisgi.alignment.distance_metrics import ConfusionMatrixCostMetric
from digohwelisgi.evaluation.confusion import (
    ConfusionAccumulator,
    character_levenshtein_align,
)
from digohwelisgi.evaluation.cost_engine import ConfusionCostEngine
from digohwelisgi.evaluation.evaluator import EvaluationRecord
from digohwelisgi.evaluation.manifold import PhoneticManifoldAnalyzer
from digohwelisgi.evaluation.visualizer import ManifoldVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("create_10pct_alignment_matrix")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dedicated ~10% operational alignment confusion and cost matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--records-jsonl",
        type=str,
        default="data/runs/evaluation/eval_records.jsonl",
        help="Path to cached eval_records.jsonl file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/runs/evaluation",
        help="Output directory for generated artifacts.",
    )
    parser.add_argument(
        "--min-cer",
        type=float,
        default=0.05,
        help="Lower CER bound for operational slice.",
    )
    parser.add_argument(
        "--max-cer",
        type=float,
        default=0.20,
        help="Upper CER bound for operational slice.",
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.1,
        help="Dirichlet smoothing prior alpha.",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=5,
        help="Minimum observation count for cost fallback.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Adjacency graph threshold for connected components.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records_path = Path(args.records_jsonl)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not records_path.exists():
        logger.error("Records file %s does not exist.", records_path)
        sys.exit(1)

    logger.info("Loading evaluation records from %s...", records_path)
    all_records: list[EvaluationRecord] = []
    with open(records_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                all_records.append(EvaluationRecord.model_validate_json(line_str))

    logger.info("Loaded %d total evaluation records.", len(all_records))

    # Derive standard active vocabulary
    dataset_vocab = sorted(list(set("".join(r.reference for r in all_records))))
    logger.info(
        "Active phonetic vocabulary (%d tokens): %s", len(dataset_vocab), dataset_vocab
    )

    # Filter for 10% CER operational distribution
    operational_records = [
        r
        for r in all_records
        if (args.min_cer <= r.cer <= args.max_cer)
        or (r.snr_tier == 15.0 and r.noise_type == "white")
    ]

    mean_cer = (
        float(np.mean([r.cer for r in operational_records]))
        if operational_records
        else 0.0
    )
    logger.info(
        "Filtered %d operational records matching ~10%% CER target (Mean CER: %.2f%%).",
        len(operational_records),
        mean_cer * 100.0,
    )

    # 1. Accumulate confusion counts
    logger.info("Accumulating character alignments and top-K posterior candidates...")
    accumulator = ConfusionAccumulator(vocab=dataset_vocab)
    for rec in operational_records:
        aligned = character_levenshtein_align(rec.reference, rec.hypothesis)
        accumulator.update_from_alignment(aligned, top_k_per_hyp_char=rec.top_k_tokens)

    cond_probs, labels = accumulator.get_conditional_probabilities(
        dirichlet_alpha=args.dirichlet_alpha
    )
    raw_counts, _ = accumulator.get_raw_counts(labels=labels)

    # 2. Compute cost matrix
    logger.info(
        "Computing normalized logarithmic distance costs via ConfusionCostEngine..."
    )
    cost_engine = ConfusionCostEngine()
    cost_dict = cost_engine.compute_costs(
        cond_probs=cond_probs,
        labels=labels,
        raw_counts=raw_counts,
        min_support=args.min_support,
    )

    cost_matrix = np.zeros_like(cond_probs)
    for i, ref in enumerate(labels):
        for j, hyp in enumerate(labels):
            cost_matrix[i, j] = cost_dict["unigram_costs"].get(ref, {}).get(hyp, 1.0)

    # 3. Block-diagonal reordering
    logger.info("Analyzing phonetic manifold and block-diagonal ordering...")
    analyzer = PhoneticManifoldAnalyzer(cond_probs=cond_probs, labels=labels)
    perm_indices, perm_labels = analyzer.get_block_diagonal_reordering(
        threshold=args.threshold
    )
    components = analyzer.get_connected_components(threshold=args.threshold)

    component_slices: list[tuple[int, int]] = []
    curr = 0
    for comp in components:
        size = len(comp)
        if size > 1:
            component_slices.append((curr, curr + size))
        curr += size

    reordered_probs = cond_probs[np.ix_(perm_indices, perm_indices)]
    reordered_costs = cost_matrix[np.ix_(perm_indices, perm_indices)]

    # 4. Save artifacts
    csv_path = out_dir / "confusion_matrix_10pct.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("," + ",".join(labels) + "\n")
        for i, ref in enumerate(labels):
            row_vals = [f"{cond_probs[i, j]:.6f}" for j in range(len(labels))]
            f.write(f"{ref}," + ",".join(row_vals) + "\n")
    logger.info("Saved 10%% operational confusion matrix CSV: %s", csv_path)

    json_path = out_dir / "confusion_cost_matrix_10pct.json"
    cost_engine.save_cost_artifact(cost_dict, json_path)
    logger.info("Saved 10%% operational cost matrix JSON: %s", json_path)

    heatmap_path = out_dir / "confusion_vs_cost_heatmap_10pct.png"
    ManifoldVisualizer.plot_side_by_side(
        confusion_matrix=reordered_probs,
        cost_matrix=reordered_costs,
        labels=perm_labels,
        save_path=heatmap_path,
        component_slices=component_slices,
        title=f"Cherokee 10% Operational Alignment Manifold (Empirical CER: {mean_cer * 100.0:.2f}%)",
    )
    logger.info("Saved 10%% operational heatmap plot: %s", heatmap_path)

    # 5. Verify ConfusionMatrixCostMetric loads and evaluates correctly
    logger.info("Verifying ConfusionMatrixCostMetric.from_json()...")
    metric = ConfusionMatrixCostMetric.from_json(json_path)
    sample_costs = [
        ("t", "k", "Plosive confusion"),
        ("m", "n", "Nasal confusion"),
        ("u", "i", "High vowel confusion"),
        ("s", "h", "Aspiration/Fricative"),
        ("t", "a", "Consonant -> Vowel"),
        ("k", "o", "Distant substitution"),
    ]

    print("\n" + "=" * 66)
    print("DEDICATED 10% OPERATIONAL ALIGNMENT MATRIX SUMMARY")
    print("=" * 66)
    print(f"Calibration Sample Count:    {len(operational_records):,}")
    print(f"Empirical Mean CER:          {mean_cer * 100.0:.2f}%")
    print(f"Active Phonetic Tokens:      {len(labels)}")
    print(
        f"Connected Components:        {len(components)} (threshold = {args.threshold})"
    )
    print("-" * 66)
    print(
        f"{'Reference':<10} {'Hypothesis':<10} {'Phonetic Relationship':<28} {'Cost':<8}"
    )
    print("-" * 66)
    for ref, hyp, desc in sample_costs:
        c = metric.get_sub_cost(ref, hyp)
        print(f"{ref:<10} {hyp:<10} {desc:<28} {c:<8.4f}")
    print("-" * 66)
    print("Generated Artifacts:")
    print(f"  • Cost Matrix JSON:   {json_path}")
    print(f"  • Confusion Matrix:   {csv_path}")
    print(f"  • Side-by-Side Plot:  {heatmap_path}")
    print("=" * 66 + "\n")


if __name__ == "__main__":
    main()
