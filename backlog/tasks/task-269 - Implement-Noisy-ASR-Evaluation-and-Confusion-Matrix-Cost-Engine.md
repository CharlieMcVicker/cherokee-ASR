---
id: TASK-269
title: Implement Noisy ASR Evaluation and Confusion Matrix Cost Engine
status: Done
assignee:
  - '@antigravity'
created_date: '2026-09-02 16:46'
updated_date: '2026-09-02 16:56'
labels:
  - evaluation
  - alignment
  - asr
dependencies: []
priority: high
type: feature
ordinal: 271000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement empirical perturbation-driven evaluation suite in transcription.evaluation to generate unigram confusion matrices and an asymmetric substitution cost metric for transcription.alignment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement AudioTransform base class and transforms (AdditiveNoise with white/pink/ambient, TimeWarp with vocoder pitch preservation, AcousticFilter Butterworth, ComposeTransforms) in transcription/evaluation/perturbations.py
- [x] #2 Implement NoisyEvaluator in transcription/evaluation/evaluator.py with multi-SNR sweep runner, CTC peak top-K extraction, and streaming JSONL sink
- [x] #3 Implement ConfusionAccumulator in transcription/evaluation/confusion.py with unigram soft-probability accumulation, Levenshtein alignment, <del>/<ins> handling, and Dirichlet smoothing
- [x] #4 Implement ConfusionCostEngine in transcription/evaluation/cost_engine.py with clamped normalized log cost and JSON serialization
- [x] #5 Implement PhoneticManifoldAnalyzer in transcription/evaluation/manifold.py with symmetric adjacency, thresholded connected components, and canonical block-diagonal reordering
- [x] #6 Implement visualizers in transcription/evaluation/visualizer.py including 2D block-diagonal heatmap, SNR drift curves, and 3D confusion mesh
- [x] #7 Implement ConfusionMatrixCostMetric in transcription/alignment/distance_metrics.py conforming to DistanceMetric protocol and loading cost JSON
- [x] #8 Implement CLI script scripts/run_noisy_eval.py for running sweeps and producing artifacts
- [x] #9 Add comprehensive unit tests in transcription/evaluation/tests/ and transcription/alignment/tests/
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 All unit tests pass in transcription/evaluation/tests and transcription/alignment/tests via pytest in cherokee-asr env
- [x] #2 Existing alignment tests pass with zero regressions
- [x] #3 Noisy evaluation CLI verified with dry-run or mock corpus
<!-- DOD:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create transcription/evaluation package directory structure and test scaffolding.
2. Implement AudioTransform, AdditiveNoise (RMS-calibrated white/pink/ambient), TimeWarp (phase vocoder pitch-preserving), AcousticFilter (Butterworth 300-3400Hz), and ComposeTransforms in perturbations.py.
3. Implement NoisyEvaluator with streaming JSONL sink, checkpoint-resume support, and CTC non-blank peak top-K extraction in evaluator.py.
4. Implement Wagner-Fischer character alignment and unigram ConfusionAccumulator with <del>/<ins> handling and Dirichlet smoothing in confusion.py.
5. Implement ConfusionCostEngine in cost_engine.py with numerically stabilized clamped normalized log-cost scaling and JSON export.
6. Implement PhoneticManifoldAnalyzer in manifold.py for symmetric adjacency, thresholded connected components, and canonical block-diagonal index reordering.
7. Implement visualizers in visualizer.py: 2D block-diagonal heatmap, SNR drift curves, and 3D surface plot.
8. Implement ConfusionMatrixCostMetric in transcription/alignment/distance_metrics.py satisfying DistanceMetric protocol.
9. Implement scripts/run_noisy_eval.py CLI driver.
10. Write comprehensive unit tests and verify full suite passes.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented empirical perturbation-driven evaluation suite in transcription.evaluation and integrated ConfusionMatrixCostMetric into transcription.alignment.distance_metrics. Full test suite passes with 182 tests across the codebase (zero regressions), and scripts/run_noisy_eval.py CLI verified with multi-SNR sweeps, JSONL streaming, and visual artifact generation.
<!-- SECTION:FINAL_SUMMARY:END -->
