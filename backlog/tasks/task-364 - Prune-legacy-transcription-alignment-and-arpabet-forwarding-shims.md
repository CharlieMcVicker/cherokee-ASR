---
id: TASK-364
title: Prune legacy transcription/alignment and arpabet forwarding shims
status: Done
assignee:
  - '@subagent'
created_date: '2026-09-23 15:55'
updated_date: '2026-09-23 16:25'
labels: []
dependencies: []
modified_files:
  - transcription/alignment/arpabet/codeswitched_preparer.py
  - transcription/alignment/arpabet/projector.py
  - transcription/alignment/calibrated_distance_metrics.py
  - transcription/alignment/normalizers.py
  - transcription/alignment/pipeline.py
  - transcription/alignment/tests/test_normalizers.py
  - transcription/alignment/tests/test_orthography.py
  - transcription/alignment/__init__.py
  - transcription/alignment/arpabet/__init__.py
  - transcription/alignment/ingestion.py
  - transcription/alignment/tests/test_aligner.py
  - transcription/alignment/tests/test_arpabet_projector.py
  - transcription/alignment/tests/test_calibrated_distance_metrics.py
  - transcription/alignment/tests/test_codeswitched_preparer.py
  - transcription/alignment/tests/test_ctc_aligner.py
  - transcription/alignment/tests/test_ingestion.py
  - transcription/alignment/tests/test_interview_realignment.py
  - transcription/alignment/tests/test_models_and_metrics.py
  - transcription/alignment/tests/test_syllabary_runners.py
  - transcription/pipelines/scripture/pipeline.py
ordinal: 394300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
transcription/alignment/arpabet/ (projector.py, codeswitched_preparer.py), transcription/alignment/calibrated_distance_metrics.py, normalizers.py, and pipeline.py remain as backward-compatibility forwarding wrappers. Clean break requires call sites to import directly from Tier 2 transcription.cherokee and Tier 3 transcription.pipelines, and removing the forwarding wrappers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 transcription/alignment/arpabet/ forwarding shims pruned
- [x] #2 transcription/alignment/ normalizers.py, calibrated_distance_metrics.py, and pipeline.py deleted
- [x] #3 All tests pass and pyright reports 0 errors
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Identify all imports and references across codebase to transcription.alignment.arpabet, transcription.alignment.calibrated_distance_metrics, transcription.alignment.normalizers, and transcription.alignment.pipeline.
2. Update callers across tests and source files to import directly from Tier 2 transcription.cherokee and Tier 3 transcription.pipelines (or transcription.cherokee.orthography, transcription.cherokee.codeswitching, transcription.cherokee.distance, transcription.cherokee.phonotactics, transcription.pipelines.dialogue, etc.).
3. Delete legacy forwarding shim files:
   - transcription/alignment/arpabet/ (projector.py, codeswitched_preparer.py, __init__.py)
   - transcription/alignment/normalizers.py
   - transcription/alignment/calibrated_distance_metrics.py
   - transcription/alignment/pipeline.py
4. Update transcription/alignment/__init__.py to remove exports from the pruned files, and relocate/update alignment tests.
5. Run pytest and pyright to ensure zero errors and no regressions.
6. Commit changes with TASK-364 prefix and update Backlog metadata.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Pruned legacy forwarding shims in transcription/alignment/: removed projector.py and codeswitched_preparer.py from arpabet/, and deleted normalizers.py, calibrated_distance_metrics.py, and pipeline.py. Updated all callers and unit tests across the codebase to import directly from Tier 2 transcription.cherokee and Tier 3 transcription.pipelines.dialogue. Verified with 425 passing pytest tests and 0 Pyright errors.
<!-- SECTION:FINAL_SUMMARY:END -->
