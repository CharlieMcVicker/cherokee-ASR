#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/generate_regime_heatmaps.py

Processes cached evaluation records on disk (eval_records.jsonl), partitions
them into acoustic noise regimes (Low, Mid, High), accumulates confusion
and cost matrices for each regime independently, and generates individual
side-by-side heatmaps and a multi-panel comparative figure.
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
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from transcription.evaluation.confusion import (
    ConfusionAccumulator,
    character_levenshtein_align,
)
from transcription.evaluation.cost_engine import ConfusionCostEngine
from transcription.evaluation.evaluator import EvaluationRecord
from transcription.evaluation.manifold import PhoneticManifoldAnalyzer
from transcription.evaluation.visualizer import ManifoldVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_regime_heatmaps")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate regime-specific (low/mid/high) confusion & cost heatmaps from cached evaluation records.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--records-jsonl",
        type=str,
        default="runs/evaluation/eval_records.jsonl",
        help="Path to cached eval_records.jsonl file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/evaluation",
        help="Output directory for generated plots, CSVs, and cost JSON artifacts.",
    )
    parser.add_argument(
        "--low-snrs",
        nargs="+",
        type=float,
        default=[25.0, 15.0],
        help="SNR tiers (dB) assigned to the Low noise regime.",
    )
    parser.add_argument(
        "--mid-snrs",
        nargs="+",
        type=float,
        default=[5.0],
        help="SNR tiers (dB) assigned to the Mid noise regime.",
    )
    parser.add_argument(
        "--high-snrs",
        nargs="+",
        type=float,
        default=[0.0, -5.0],
        help="SNR tiers (dB) assigned to the High noise regime.",
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.1,
        help="Dirichlet smoothing prior alpha for conditional probability normalization.",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=5,
        help="Minimum observation count for empirical substitution cost estimation.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Adjacency graph threshold for connected component decomposition.",
    )
    return parser.parse_args()


def load_records_from_jsonl(path: Path) -> list[EvaluationRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation records file not found: {path}")

    logger.info("Loading cached evaluation records from %s...", path)
    records: list[EvaluationRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                rec = EvaluationRecord.model_validate_json(line_str)
                records.append(rec)
            except Exception as e:
                logger.warning("Line %d in %s failed to parse: %s", line_num, path, e)

    logger.info("Loaded %d evaluation records from disk cache.", len(records))
    return records


def process_regime(
    regime_name: str,
    records: list[EvaluationRecord],
    dataset_vocab: list[str],
    out_dir: Path,
    dirichlet_alpha: float = 0.1,
    min_support: int = 5,
    threshold: float = 0.05,
) -> dict[str, Any]:
    """Accumulate confusion, compute costs, reorder manifold, and render heatmaps for a regime."""
    mean_cer = float(np.mean([r.cer for r in records])) if records else 0.0
    logger.info(
        "Processing '%s' regime (%d records, Mean CER: %.2f%%)...",
        regime_name,
        len(records),
        mean_cer * 100.0,
    )

    # 1. Accumulate confusion counts
    accumulator = ConfusionAccumulator(vocab=dataset_vocab)
    for rec in records:
        aligned = character_levenshtein_align(rec.reference, rec.hypothesis)
        accumulator.update_from_alignment(aligned, top_k_per_hyp_char=rec.top_k_tokens)

    cond_probs, labels = accumulator.get_conditional_probabilities(
        dirichlet_alpha=dirichlet_alpha
    )
    raw_counts, _ = accumulator.get_raw_counts(labels=labels)

    # 2. Compute cost matrix
    cost_engine = ConfusionCostEngine()
    cost_dict = cost_engine.compute_costs(
        cond_probs=cond_probs,
        labels=labels,
        raw_counts=raw_counts,
        min_support=min_support,
    )

    cost_matrix = np.zeros_like(cond_probs)
    for i, ref in enumerate(labels):
        for j, hyp in enumerate(labels):
            cost_matrix[i, j] = cost_dict["unigram_costs"].get(ref, {}).get(hyp, 1.0)

    # 3. Block-diagonal reordering
    analyzer = PhoneticManifoldAnalyzer(cond_probs=cond_probs, labels=labels)
    perm_indices, perm_labels = analyzer.get_block_diagonal_reordering(
        threshold=threshold
    )
    components = analyzer.get_connected_components(threshold=threshold)

    # Build component slice boundaries for bounding box overlay
    component_slices: list[tuple[int, int]] = []
    curr = 0
    for comp in components:
        size = len(comp)
        if size > 1:
            component_slices.append((curr, curr + size))
        curr += size

    reordered_probs = cond_probs[np.ix_(perm_indices, perm_indices)]
    reordered_costs = cost_matrix[np.ix_(perm_indices, perm_indices)]

    # 4. Save artifacts for this regime
    clean_name = regime_name.lower().replace(" ", "_")
    csv_path = out_dir / f"confusion_matrix_{clean_name}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("," + ",".join(labels) + "\n")
        for i, ref in enumerate(labels):
            row_vals = [f"{cond_probs[i, j]:.6f}" for j in range(len(labels))]
            f.write(f"{ref}," + ",".join(row_vals) + "\n")

    json_path = out_dir / f"confusion_cost_matrix_{clean_name}.json"
    cost_engine.save_cost_artifact(cost_dict, json_path)

    heatmap_path = out_dir / f"confusion_vs_cost_heatmap_{clean_name}.png"
    ManifoldVisualizer.plot_side_by_side(
        confusion_matrix=reordered_probs,
        cost_matrix=reordered_costs,
        labels=perm_labels,
        save_path=heatmap_path,
        component_slices=component_slices,
        title=f"Phonetic Confusion vs. Substitution Cost: {regime_name} (Mean CER: {mean_cer * 100.0:.2f}%)",
    )

    return {
        "name": f"{regime_name} (CER: {mean_cer * 100.0:.1f}%)",
        "regime_key": clean_name,
        "sample_count": len(records),
        "mean_cer": mean_cer,
        "component_count": len(components),
        "confusion_matrix": reordered_probs,
        "cost_matrix": reordered_costs,
        "labels": perm_labels,
        "component_slices": component_slices,
        "heatmap_path": heatmap_path,
        "csv_path": csv_path,
        "json_path": json_path,
    }


def main() -> None:
    args = parse_args()
    records_path = Path(args.records_jsonl)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records_from_jsonl(records_path)
    if not records:
        logger.error("No evaluation records found in %s", records_path)
        sys.exit(1)

    # Derive standard active vocabulary from reference strings
    dataset_vocab = sorted(list(set("".join(r.reference for r in records))))
    logger.info(
        "Active phonetic vocabulary (%d tokens): %s", len(dataset_vocab), dataset_vocab
    )

    # Partition records by regime
    low_set = set(args.low_snrs)
    mid_set = set(args.mid_snrs)
    high_set = set(args.high_snrs)

    low_records = [r for r in records if r.snr_tier in low_set]
    mid_records = [r for r in records if r.snr_tier in mid_set]
    high_records = [r for r in records if r.snr_tier in high_set]

    logger.info(
        "Partitioned records: Low (SNRs %s) = %d, Mid (SNRs %s) = %d, High (SNRs %s) = %d",
        args.low_snrs,
        len(low_records),
        args.mid_snrs,
        len(mid_records),
        args.high_snrs,
        len(high_records),
    )

    regime_configs = [
        ("low", low_records, "Low Noise"),
        ("mid", mid_records, "Mid Noise"),
        ("high", high_records, "High Noise"),
    ]

    regime_results: list[dict[str, Any]] = []
    for short_name, regime_recs, title_name in regime_configs:
        if not regime_recs:
            logger.warning("No records found for regime %s, skipping.", short_name)
            continue
        res = process_regime(
            regime_name=short_name,
            records=regime_recs,
            dataset_vocab=dataset_vocab,
            out_dir=out_dir,
            dirichlet_alpha=args.dirichlet_alpha,
            min_support=args.min_support,
            threshold=args.threshold,
        )
        regime_results.append(res)

    # Render multi-panel comparative figure
    if regime_results:
        comparison_path = out_dir / "confusion_vs_cost_heatmaps_by_regime.png"
        logger.info(
            "Rendering comparative multi-regime figure to %s...", comparison_path
        )
        comp_regimes = []
        for r in regime_results:
            comp_regimes.append(
                {
                    "name": r["name"],
                    "confusion_matrix": r["confusion_matrix"],
                    "cost_matrix": r["cost_matrix"],
                    "component_slices": r["component_slices"],
                }
            )

        labels_to_use = regime_results[0]["labels"] if regime_results else dataset_vocab
        ManifoldVisualizer.plot_regime_comparison(
            regimes=comp_regimes,
            labels=labels_to_use,
            save_path=comparison_path,
            title="Cherokee Phonetic Confusion & Substitution Cost Manifolds Across Noise Regimes",
        )

    # Print summary report
    print("\n" + "=" * 64)
    print("NOISE REGIME EVALUATION & HEATMAP SUMMARY")
    print("=" * 64)
    print(f"{'Regime':<24} {'Samples':<10} {'Mean CER':<12} {'Components':<10}")
    print("-" * 64)
    for res in regime_results:
        print(
            f"{res['regime_key'].capitalize():<24} "
            f"{res['sample_count']:<10} "
            f"{res['mean_cer'] * 100.0:6.2f}%     "
            f"{res['component_count']:<10}"
        )
    print("\n--- Generated Regime Artifacts ---")
    for res in regime_results:
        print(f"  • {res['regime_key'].upper()}:")
        print(f"      Heatmap: {res['heatmap_path']}")
        print(f"      Matrix:  {res['csv_path']}")
        print(f"      Costs:   {res['json_path']}")
    if regime_results:
        print(f"  • MULTI-REGIME COMPARISON: {comparison_path}")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
