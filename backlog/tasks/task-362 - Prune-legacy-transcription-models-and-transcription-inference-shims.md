---
id: TASK-362
title: Prune legacy transcription/models and transcription/inference shims
status: Done
assignee:
  - '@subagent'
created_date: '2026-09-23 15:55'
updated_date: '2026-09-23 16:12'
labels: []
dependencies: []
modified_files:
  - docs/alignment.md
  - scripts/batch_cache_emissions.py
  - scripts/benchmark_ctc_segmentation_100_verses.py
  - scripts/calibrate_intrusion_penalties.py
  - scripts/generate_confusion_matrix.py
  - scripts/realign_gs_mm.py
  - scripts/rescore_syllabary_dataset.py
  - scripts/run_noisy_eval.py
  - scripts/tune_ctc_aligner_config.py
  - syllabary_transcriber/app.py
  - transcription/alignment/ctc_aligner.py
  - transcription/alignment/tests/test_arpabet_inference.py
  - transcription/alignment/tests/test_cli.py
  - transcription/alignment/tests/test_ctc_aligner.py
  - transcription/apps/cli.py
  - transcription/cherokee/models/__init__.py
  - transcription/cherokee/models/loader.py
  - transcription/cherokee/orthography/__init__.py
  - transcription/core/audio/__init__.py
  - transcription/core/models/tests/test_models.py
  - transcription/evaluation/evaluator.py
  - transcription/inference/__init__.py
  - transcription/inference/infer.py
  - transcription/inference/labeler.py
  - transcription/models/__init__.py
  - transcription/models/asr_model.py
  - transcription/new_testament/tests/test_pipeline.py
  - transcription/training/evaluate_checkpoint.py
  - transcription/training/evaluate_local_checkpoints.py
  - transcription/training/evaluate_revisions.py
  - transcription/training/train.py
  - transcription/utils/evaluation.py
  - transcription/utils/test_asr_model.py
  - transcription/utils/test_evaluation.py
ordinal: 392300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
transcription/models/asr_model.py, WordConfidence, and transcription/inference/ shims were preserved for backward compatibility. Call sites across tests, training, and evaluation should import directly from transcription.cherokee.models.loader, transcription.core.models, and transcription.core.audio. Delete the legacy modules and update callers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 transcription/models/ directory and legacy shims deleted
- [x] #2 Call sites in training/ and evaluation/ updated to transcription.cherokee.models and transcription.core
- [x] #3 All tests pass and pyright reports 0 errors
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Replace all legacy imports across tests, scripts, evaluation, training, app, and docs from transcription.models and transcription.inference with canonical imports.\n2. Delete legacy shim directories: transcription/models/ and transcription/inference/.\n3. Verify test suite and typechecking (pytest and pyright transcription).\n4. Update backlog task modified files, check ACs, and summarize.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Pruned legacy transcription/models/ and transcription/inference/ directories and deleted all deprecated backward compatibility shims. Updated all call sites across tests, training, evaluation, scripts, and apps to import canonical classes (CherokeeASRModel, ASRResult, WordConfidence, ModelOutput, TARGET_SAMPLE_RATE, normalize_text) directly from transcription.cherokee.models, transcription.cherokee.orthography, transcription.core.models, and transcription.core.audio. Verified with pytest (442 passing tests) and pyright (0 errors).
<!-- SECTION:FINAL_SUMMARY:END -->
