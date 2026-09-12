---
id: TASK-300
title: >-
  Implement Full-Chapter Continuous CTC Segmentation Alignment Pipeline for
  Cherokee New Testament
status: To Do
assignee: []
created_date: '2026-09-12 19:22'
labels:
  - alignment
  - ctc-segmentation
  - new-testament
  - bible
dependencies: []
ordinal: 312000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Upgrade Cherokee Bible alignment pipeline from intra-verse benchmarking to full continuous chapter-level alignment using CTCSegmentationAligner, overlapping sliding-window inference, inter-verse silence partitioning, and 4-tier Praat/manifest export.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Implement overlapping sliding-window inference with margin trimming in CTCSegmentationAligner or CherokeeASRModel
- [ ] #2 Fix CTCSegmentationAligner.align() to use determine_utterance_segments for verse boundaries and harvest intra-verse word intervals from trellis
- [ ] #3 Refactor realign_bible.py to use CTCSegmentationAligner for Mark and Matthew continuous chapter alignment
- [ ] #4 Verify unclipped verse audio slicing and export 4-tier Praat TextGrids, JSON manifests, and training CSVs
- [ ] #5 Add comprehensive unit and integration tests for chapter alignment and regression check against 100-verse benchmark
<!-- AC:END -->
