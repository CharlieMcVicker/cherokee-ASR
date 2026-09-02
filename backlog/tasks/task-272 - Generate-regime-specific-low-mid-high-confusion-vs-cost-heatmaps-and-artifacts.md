---
id: TASK-272
title: >-
  Generate regime-specific (low/mid/high) confusion vs cost heatmaps and
  artifacts
status: Done
assignee:
  - '@antigravity'
created_date: '2026-09-02 18:53'
updated_date: '2026-09-02 18:54'
labels:
  - evaluation
  - visualization
  - analysis
dependencies: []
priority: medium
type: feature
ordinal: 284000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Partition cached evaluation records on disk into low, mid, and high noise regimes, compute separate confusion matrices and substitution costs for each regime, and render separate 2D side-by-side heatmaps as well as a multi-regime comparison figure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Define low ([25, 15] dB), mid ([5] dB), and high ([0, -5] dB) noise regime partitions
- [x] #2 Accumulate confusion matrices and compute cost matrices for each noise regime independently from cached eval_records.jsonl
- [x] #3 Render individual side-by-side heatmaps for low, mid, and high noise regimes
- [x] #4 Render a combined multi-panel comparison figure across low/mid/high regimes
- [x] #5 Add CLI support to run_noisy_eval.py or a dedicated regime analysis script to generate these on demand
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Low, mid, high regime confusion matrices and cost matrices computed from disk cache
- [x] #2 Individual heatmap images generated for each regime
- [x] #3 Combined comparative multi-panel plot generated
- [x] #4 Tests added/updated and passing
<!-- DOD:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a regime analysis capability in visualizer.py or a dedicated module/script.
2. In scripts/generate_regime_heatmaps.py (and optionally flags in run_noisy_eval.py), load records from eval_records.jsonl.
3. Group records by regime: Low (SNR >= 15 dB), Mid (SNR == 5 dB), High (SNR <= 0 dB).
4. Compute ConfusionAccumulator, ConfusionCostEngine, and PhoneticManifoldAnalyzer for each regime.
5. Render individual side-by-side heatmaps: confusion_vs_cost_low.png, confusion_vs_cost_mid.png, confusion_vs_cost_high.png.
6. Render a combined comparison plot: confusion_vs_cost_heatmaps_by_regime.png.
7. Verify with unit test and generate final plots in runs/evaluation/.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created scripts/generate_regime_heatmaps.py and added ManifoldVisualizer.plot_regime_comparison(). Partitioned on-disk cached evaluation records into Low (25, 15 dB; 2.91% CER), Mid (5 dB; 29.41% CER), and High (0, -5 dB; 67.70% CER) regimes. Generated independent confusion matrices, substitution cost matrices, side-by-side heatmaps, and a 3-regime comparative figure.
<!-- SECTION:FINAL_SUMMARY:END -->
