---
id: TASK-384
title: Migrate training module to digohwelisgi.cherokee.training
status: Done
assignee:
  - '@self'
created_date: '2026-11-03 19:45'
updated_date: '2026-11-03 19:48'
labels: []
dependencies: []
modified_files:
  - digohwelisgi/cherokee/training/__init__.py
  - digohwelisgi/cherokee/training/train.py
  - digohwelisgi/cherokee/training/prepare_csv.py
  - digohwelisgi/cherokee/training/prepare_conrad_csv.py
  - digohwelisgi/cherokee/training/evaluate_checkpoint.py
  - digohwelisgi/cherokee/training/evaluate_revisions.py
  - digohwelisgi/cherokee/training/evaluate_local_checkpoints.py
  - digohwelisgi/cherokee/training/datasets/__init__.py
  - digohwelisgi/cherokee/training/datasets/arpabet.py
  - digohwelisgi/cherokee/training/tests/test_train_pipeline.py
  - AGENTS.md
  - README.md
  - >-
    backlog/docs/architecture/audio/doc-13 -
    Audio-Segmentation-and-VAD-Masking.md
  - backlog/docs/architecture/models/doc-11 - Models-and-Inference-Subsystem.md
  - >-
    backlog/docs/architecture/training/doc-14 -
    Model-Training-and-Evaluation-Pipeline.md
ordinal: 417300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Relocate the Cherokee-specific training and checkpoint evaluation scripts under Tier 2 (digohwelisgi.cherokee.training) to restore architectural coherence and prepare for language-agnostic core training abstractions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 digohwelisgi.training is moved to digohwelisgi.cherokee.training without legacy compatibility shims
- [x] #2 All internal imports and references across tests, documentation, and tools are updated
- [x] #3 All existing training tests pass under pytest and pyright passes cleanly
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 pytest and pyright digohwelisgi pass with zero errors
<!-- DOD:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Move directory digohwelisgi/training to digohwelisgi/cherokee/training via git mv.
2. Update all imports from digohwelisgi.training to digohwelisgi.cherokee.training across codebase and tests.
3. Update AGENTS.md, README.md, and documentation references.
4. Run pytest and pyright digohwelisgi to verify zero regressions.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Relocated digohwelisgi.training to digohwelisgi.cherokee.training under Tier 2.
- Updated all internal package imports across tests, dataset loaders, and evaluation scripts.
- Synchronized AGENTS.md, README.md, and subsystem documentation.
- Verified with pyright digohwelisgi (0 errors, 0 warnings) and full pytest suite (369 passed).
<!-- SECTION:FINAL_SUMMARY:END -->
