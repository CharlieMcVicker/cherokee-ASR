# -*- coding: utf-8 -*-
"""
generate_delta_cer_analysis.py

Generates histogram visualizations and a detailed breakdown of Delta CER for Cherokee syllabary enrichment
on training data. Identifies potential mislabeled training examples where syllabary enrichment or ASR reconciliation
differs significantly from ground-truth target text.
"""

import os
import json
import csv
from typing import List, Dict, Any

from transcription.syllabary_enrichment.alignment_engine import (
    get_base_transliteration,
    align_character_syllable,
)
from transcription.syllabary_enrichment.enrich_syllabary import reconcile_phonetics
from transcription.syllabary_enrichment.evaluate_reconciliation import calculate_cer


def draw_svg_histogram(
    values: List[float],
    title: str,
    xlabel: str,
    output_svg_path: str,
    color: str = "#1f77b4",
    num_bins: int = 25,
):
    """Generates a clean standalone SVG histogram without matplotlib."""
    if not values:
        return

    min_v, max_v = min(values), max(values)
    if min_v == max_v:
        min_v -= 0.1
        max_v += 0.1

    bin_width = (max_v - min_v) / num_bins
    bins = [min_v + i * bin_width for i in range(num_bins + 1)]
    counts = [0] * num_bins

    for v in values:
        idx = int((v - min_v) / bin_width)
        if idx >= num_bins:
            idx = num_bins - 1
        counts[idx] += 1

    max_count = max(counts) if counts else 1

    # SVG layout parameters
    width, height = 800, 500
    margin_left, margin_right = 70, 40
    margin_top, margin_bottom = 70, 60
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    bar_w = plot_width / num_bins

    svg_lines = []
    svg_lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="background-color: #ffffff; font-family: system-ui, -apple-system, sans-serif;">'
    )

    # Title
    svg_lines.append(
        f'<text x="{width/2}" y="35" text-anchor="middle" font-size="16" font-weight="bold" fill="#222222">{title}</text>'
    )

    # Axes
    x_axis_y = height - margin_bottom
    y_axis_x = margin_left
    svg_lines.append(
        f'<line x1="{y_axis_x}" y1="{x_axis_y}" x2="{width - margin_right}" y2="{x_axis_y}" stroke="#444444" stroke-width="2"/>'
    )
    svg_lines.append(
        f'<line x1="{y_axis_x}" y1="{margin_top}" x2="{y_axis_x}" y2="{x_axis_y}" stroke="#444444" stroke-width="2"/>'
    )

    # Bars
    for i in range(num_bins):
        c = counts[i]
        h = (c / max_count) * plot_height if max_count > 0 else 0
        x = y_axis_x + i * bar_w
        y = x_axis_y - h
        svg_lines.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 1:.1f}" height="{h:.1f}" fill="{color}" opacity="0.85" stroke="#111111" stroke-width="0.5"/>'
        )
        if c > 0 and num_bins <= 30:
            svg_lines.append(
                f'<text x="{x + bar_w/2:.1f}" y="{y - 4:.1f}" text-anchor="middle" font-size="10" fill="#333333">{c}</text>'
            )

    # Zero Line marker if zero in range
    if min_v <= 0.0 <= max_v:
        zero_x = y_axis_x + ((0.0 - min_v) / (max_v - min_v)) * plot_width
        svg_lines.append(
            f'<line x1="{zero_x:.1f}" y1="{margin_top}" x2="{zero_x:.1f}" y2="{x_axis_y}" stroke="#d62728" stroke-width="2" stroke-dasharray="4,4"/>'
        )
        svg_lines.append(
            f'<text x="{zero_x:.1f}" y="{margin_top - 10}" text-anchor="middle" font-size="11" fill="#d62728" font-weight="bold">Zero (0.0)</text>'
        )

    # X-axis Labels
    svg_lines.append(
        f'<text x="{y_axis_x}" y="{x_axis_y + 20}" text-anchor="middle" font-size="11" fill="#444444">{min_v:.3f}</text>'
    )
    svg_lines.append(
        f'<text x="{width - margin_right}" y="{x_axis_y + 20}" text-anchor="middle" font-size="11" fill="#444444">{max_v:.3f}</text>'
    )
    svg_lines.append(
        f'<text x="{width/2}" y="{height - 15}" text-anchor="middle" font-size="13" font-weight="500" fill="#333333">{xlabel}</text>'
    )

    # Y-axis Labels
    svg_lines.append(
        f'<text x="{y_axis_x - 10}" y="{x_axis_y}" text-anchor="end" font-size="11" fill="#444444">0</text>'
    )
    svg_lines.append(
        f'<text x="{y_axis_x - 10}" y="{margin_top + 10}" text-anchor="end" font-size="11" fill="#444444">{max_count}</text>'
    )
    svg_lines.append(
        f'<text x="20" y="{height/2}" text-anchor="middle" font-size="13" font-weight="500" fill="#333333" transform="rotate(-90 20 {height/2})">Frequency (Count)</text>'
    )

    svg_lines.append("</svg>")

    with open(output_svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_lines))
    print(f"Saved SVG histogram to {output_svg_path}")


def main():
    manifest_csv = "training_data/processed/split_audio_syl_target.csv"
    output_dir = "data/results"
    os.makedirs(output_dir, exist_ok=True)

    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = [r for r in reader if r.get("split") == "train"]

    print(f"Loaded {len(records)} training records from {manifest_csv}")

    analysis_data = []

    for idx, r in enumerate(records):
        audio_path = r.get("audio_path", f"train_{idx}")
        syllabary = r.get("syllabary", "").strip()
        gt_target = r.get("target", "").strip()

        base_trans = get_base_transliteration(syllabary)

        # 1. Base CER: CER of plain base transliteration vs ground truth target
        base_cer = calculate_cer(gt_target, base_trans)

        # 2. Reconciled/Enriched CER when using GT as emitted ASR (simulating ASR predictions)
        aligned_pairs = align_character_syllable(syllabary, gt_target)
        reconciled_phonetics = reconcile_phonetics(
            syllabary_text=syllabary,
            base_transliteration=base_trans,
            emitted_text=gt_target,
            aligned_pairs=aligned_pairs,
        )
        reconciled_cer = calculate_cer(gt_target, reconciled_phonetics)

        # Delta CER definitions:
        # A) Delta CER vs Base Transliteration (base_cer - reconciled_cer)
        #    Positive value = Improvement over standard base syllabary transliteration
        delta_cer_vs_base = base_cer - reconciled_cer

        # B) Delta CER vs ASR Emitted GT (reconciled_cer - 0.0)
        #    Positive value = Discrepancy/degradation introduced by syllabary alignment rules relative to GT
        delta_cer_vs_gt = reconciled_cer - 0.0

        item = {
            "record_id": audio_path,
            "syllabary": syllabary,
            "base_transliteration": base_trans,
            "gt_target": gt_target,
            "reconciled_phonetics": reconciled_phonetics,
            "base_cer": base_cer,
            "reconciled_cer": reconciled_cer,
            "delta_cer_vs_base": delta_cer_vs_base,
            "delta_cer_vs_gt": delta_cer_vs_gt,
        }
        analysis_data.append(item)

    # Save complete JSON analysis
    json_path = os.path.join(output_dir, "syllabary_enrichment_train_delta_cer.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON metrics to {json_path}")

    # Plot 1: SVG Histogram of Delta CER (Base CER - Enriched CER)
    deltas_vs_base = [d["delta_cer_vs_base"] for d in analysis_data]
    svg_path1 = os.path.join(output_dir, "delta_cer_base_vs_enriched_histogram.svg")
    draw_svg_histogram(
        deltas_vs_base,
        title="Distribution of Delta CER (Base Transliteration CER - Enriched CER) [N=1182]",
        xlabel="Delta CER (Positive = Syllabary Enrichment Improved Accuracy over Base)",
        output_svg_path=svg_path1,
        color="#1f77b4",
    )

    # Plot 2: SVG Histogram of Reconciled CER Discrepancy (Reconciled CER - GT CER)
    deltas_vs_gt = [d["delta_cer_vs_gt"] for d in analysis_data]
    svg_path2 = os.path.join(output_dir, "reconciliation_discrepancy_histogram.svg")
    draw_svg_histogram(
        deltas_vs_gt,
        title="Distribution of Reconciliation Discrepancy (Reconciled CER vs GT Target) [N=1182]",
        xlabel="CER Discrepancy (Positive = Syllabary Alignment Constraints Differ from GT Target)",
        output_svg_path=svg_path2,
        color="#ff7f0e",
    )

    # Identify potential mislabeled data
    mislabeled_candidates = sorted(
        analysis_data, key=lambda x: x["base_cer"], reverse=True
    )
    high_error_candidates = [c for c in mislabeled_candidates if c["base_cer"] > 0.05]

    print(
        f"\n--- POTENTIAL MISLABELED / HIGH DISCREPANCY EXAMPLES (Found {len(high_error_candidates)}) ---"
    )
    for c in high_error_candidates[:15]:
        print(f"ID: {c['record_id']}")
        print(f"  Syllabary  : {c['syllabary']}")
        print(f"  Base Trans : {c['base_transliteration']}")
        print(f"  Reconciled : {c['reconciled_phonetics']}")
        print(f"  GT Target  : {c['gt_target']}")
        print(
            f"  Base CER   : {c['base_cer']:.4f} | Reconciled CER: {c['reconciled_cer']:.4f}"
        )
        print()


if __name__ == "__main__":
    main()
