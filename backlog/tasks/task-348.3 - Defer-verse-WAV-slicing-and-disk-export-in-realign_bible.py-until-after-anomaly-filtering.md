---
id: TASK-348.3
title: >-
  Defer verse WAV slicing and disk export in realign_bible.py until after
  anomaly filtering
status: Done
assignee:
  - '@antigravity'
created_date: '2026-09-17 18:30'
updated_date: '2026-09-18 13:34'
labels:
  - alignment
  - performance
  - realign_bible
dependencies: []
parent_task_id: TASK-348
priority: medium
ordinal: 367000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In scripts/realign_bible.py, avoid executing pydub audio slicing and WAV export for verses that are discarded due to anomalies or missing emissions, and batch slice exports to minimize disk I/O overhead.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Audio slices are only exported to disk for valid non-anomalous verses that will be included in the training dataset and record manifests
- [x] #2 Skipped/filtered verses do not perform redundant disk I/O
- [x] #3 pytest and pyright pass with zero regressions
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. In realign_bible.py, defer audio slicing and WAV disk export until after checking chunk.has_anomalies and emitted_sentence\n2. Maintain relative audio_path shell string in alignment records manifest\n3. Update unit tests in test_pipeline.py to assert deferred slicing behavior\n4. Verify with pytest and pyright
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Deferred verse audio slicing and WAV disk export in realign_bible.py so that disk I/O is only executed for non-anomalous verses with valid emissions. Verified with test_pipeline.py unit tests, full test suite (277 tests), and pyright (0 errors).
<!-- SECTION:FINAL_SUMMARY:END -->
