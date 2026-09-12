---
id: TASK-300.3
title: >-
  Refactor realign_bible.py and new_testament pipeline to use continuous chapter
  CTCSegmentationAligner
status: Done
assignee: []
created_date: '2026-09-12 19:26'
updated_date: '2026-09-12 19:34'
labels: []
dependencies: []
parent_task_id: TASK-300
ordinal: 315000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Refactor realign_bible.py and transcription/new_testament/pipeline.py to use CTCSegmentationAligner for continuous chapter alignment, slice verse audio without clipping, and export 4-tier Praat TextGrids, JSON manifests, and training CSVs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Update align_chapter in transcription/new_testament/pipeline.py to support CTCSegmentationAligner
- [x] #2 Refactor realign_bible.py to run full continuous chapter alignment with CTCSegmentationAligner
- [x] #3 Slice unclipped verse audio WAV files using natural inter-verse boundary partition points
- [x] #4 Export 4-tier Praat TextGrids (Chunks, Words, Padded Words, Reconciled/ASR), alignment records JSON, and train CSVs
<!-- AC:END -->
