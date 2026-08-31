---
id: TASK-238.3
title: 'Migrate Callers, Eliminate Dead Code in timestamping, and Update Tests'
status: Done
assignee:
  - '@subagent'
created_date: '2026-08-31 20:38'
updated_date: '2026-08-31 20:43'
labels: []
dependencies: []
parent_task_id: TASK-238
ordinal: 240000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update pyproject.toml and transcription.new_testament.pipeline to use transcription.alignment directly. Delete obsolete duplicate timestamping files (aligner.py, exporter.py, align_cli.py, prepare_ground_truth.py, audio_segmenter.py) and update/consolidate all test suites into transcription.alignment.tests and verify 100% test pass rate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Update pyproject.toml align-cherokee script to point to transcription.alignment.cli:main
- [x] #2 Update transcription.new_testament.pipeline to consume transcription.alignment natively
- [x] #3 Remove dead code files under transcription/timestamping
- [x] #4 Create test_cli.py and test_pipeline.py under transcription/alignment/tests and update existing tests
- [x] #5 Ensure all unit tests in pytest transcription/ pass cleanly in cherokee-asr conda environment
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated align-cherokee script in pyproject.toml and new_testament pipeline to transcription.alignment.cli. Moved test_audio_segmenter to transcription.audio. Removed legacy to_verse_intervals adapter and deleted all dead code under transcription/timestamping. Configured pyproject.toml pytest configuration and verified 100% test pass rate across all 86 unit tests.
<!-- SECTION:FINAL_SUMMARY:END -->
