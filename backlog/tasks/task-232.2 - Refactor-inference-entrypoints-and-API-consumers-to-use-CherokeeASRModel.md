---
id: TASK-232.2
title: Refactor inference entrypoints and API consumers to use CherokeeASRModel
status: Done
assignee:
  - '@myself'
created_date: '2026-08-30 19:18'
updated_date: '2026-08-30 19:23'
labels: []
dependencies: []
parent_task_id: TASK-232
ordinal: 225000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update all calling sites across the repository to use CherokeeASRModel instead of legacy raw get_model/infer_* helpers. Refactor single.py, batch.py, run.py, syllabary_transcriber/app.py, align_cli.py, aligner.py, batch_inference_aligner.py, evaluation.py, and scripts/generate_confusion_matrix.py.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Refactor single.py, batch.py, and run.py
- [x] #2 Refactor syllabary_transcriber/app.py to use CherokeeASRModel
- [x] #3 Refactor timestamping and syllabary enrichment modules (align_cli.py, aligner.py, batch_inference_aligner.py)
- [x] #4 Refactor evaluation tools (evaluation.py, generate_confusion_matrix.py)
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Refactored all inference entrypoints, CLI scripts, and API consumers across the codebase to utilize CherokeeASRModel directly instead of legacy raw get_model / infer helper routines. Updated single.py, batch.py, run.py, syllabary_transcriber/app.py, align_cli.py, aligner.py, batch_inference_aligner.py, evaluation.py, and scripts/generate_confusion_matrix.py while preserving all CLI arguments, flags, output formats, and behavioral compatibility. Verified all unit tests pass with 0 regressions.
<!-- SECTION:FINAL_SUMMARY:END -->
