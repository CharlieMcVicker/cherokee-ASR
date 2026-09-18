---
id: TASK-348.3
title: >-
  Defer verse WAV slicing and disk export in realign_bible.py until after
  anomaly filtering
status: To Do
assignee: []
created_date: '2026-09-17 18:30'
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
- [ ] #1 Audio slices are only exported to disk for valid non-anomalous verses that will be included in the training dataset and record manifests
- [ ] #2 Skipped/filtered verses do not perform redundant disk I/O
- [ ] #3 pytest and pyright pass with zero regressions
<!-- AC:END -->
