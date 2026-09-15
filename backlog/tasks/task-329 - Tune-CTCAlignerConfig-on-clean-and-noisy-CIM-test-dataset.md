---
id: TASK-329
title: Tune CTCAlignerConfig on clean and noisy CIM test dataset
status: Done
assignee:
  - '@agent'
created_date: '2026-09-14 19:27'
updated_date: '2026-09-14 19:32'
labels: []
dependencies: []
ordinal: 345000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Perform systematic hyperparameter tuning of CTCAlignerConfig on the CIM test dataset (training_data/processed/cim-wav2vec2-test.csv) with CER/WER optimization against ground truth reference text. Evaluate both clean audio and pink-noise perturbed audio (15-20 dB SNR).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Build tuning evaluation runner for CTCAlignerConfig on clean and pink-noise perturbed CIM test dataset
- [x] #2 Execute parameter sweep on clean CIM test dataset and find optimal CER/WER configuration
- [x] #3 Execute parameter sweep on noisy CIM test dataset (additive pink noise ~18 dB SNR) and find optimal CER/WER configuration
- [x] #4 Generate structured evaluation report artifact and recommend/apply optimal CTCAlignerConfig settings
- [x] #5 Verify test suite passes with pytest
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create scripts/tune_ctc_aligner_config.py implementing pure domain data types and evaluation pipeline for CIM test dataset (training_data/processed/cim-wav2vec2-test.csv).
2. Pre-extract and cache ASR log-probabilities (lpz) for clean and pink-noise perturbed audio (18 dB SNR, fixed seed).
3. Execute hyperparameter grid search across syncope_penalty, per-token intrusive_penalties (/h/, /'/), intrusive_min_logprobs, enforce_phonotactics, and buffer lengths on both clean and noisy datasets.
4. Calculate CER, WER, and alignment metrics across all configurations and identify Pareto-optimal settings.
5. Save evaluation output JSON to runs/evaluation/ and generate structured report artifact.
6. Verify and update default CTCAlignerConfig if improvements are demonstrated.
7. Run full pytest suite to verify no regressions.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented scripts/tune_ctc_aligner_config.py to run systematic grid evaluation of CTCAlignerConfig across 88 candidate parameter configurations on both clean and pink-noise perturbed (18 dB SNR) CIM test dataset. Generated quantitative CER/WER leaderboard in runs/evaluation/ctc_aligner_config_tuning_results.json and authored detailed report artifact. Verified all 266 unit and integration tests pass with pytest.
<!-- SECTION:FINAL_SUMMARY:END -->
