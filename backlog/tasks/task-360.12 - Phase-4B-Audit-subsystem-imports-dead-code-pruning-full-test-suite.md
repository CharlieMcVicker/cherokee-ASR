---
id: TASK-360.12
title: 'Phase 4B: Audit subsystem imports, dead code pruning & full test suite'
status: To Do
assignee: []
created_date: '2026-09-21 20:33'
updated_date: '2026-09-21 20:41'
labels: []
dependencies: []
parent_task_id: TASK-360
ordinal: 390200
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Audit and update all import paths across transcription/evaluation/, transcription/training/, syllabary_transcriber/, prune dead legacy files (extractors.py, normalizers.py, threshold_finder.py, obsolete inference/*.py scripts), and verify zero pyright diagnostics and all unit/integration tests pass.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 transcription/evaluation/ and transcription/training/ updated to import from new modular paths
- [ ] #2 Obsolete files (extractors.py, normalizers.py, threshold_finder.py, inference/single.py, batch.py, run.py) deleted per Clean Break Protocol
- [ ] #3 pyright transcription reports 0 errors and 0 warnings
- [ ] #4 All unit and integration tests under pytest pass with zero failures
<!-- AC:END -->
