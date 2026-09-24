---
id: TASK-300.2
title: >-
  Implement full-chapter verse segmentation with determine_utterance_segments
  and intra-verse word timing harvesting
status: Done
assignee: []
created_date: '2026-09-12 19:26'
updated_date: '2026-09-12 19:29'
labels: []
dependencies: []
parent_task_id: TASK-300
ordinal: 314000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Upgrade CTCSegmentationAligner.align() to use determine_utterance_segments for natural inter-verse silence partitioning and harvest intra-verse word intervals, character log-probability confidence scores, anomaly flags, and emitted hypotheses from the CTC trellis path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Use determine_utterance_segments to establish natural inter-verse silence partition boundaries
- [x] #2 Harvest word-level timings, state lists, character log-prob confidences, and anomaly flags for all words within each verse chunk
- [x] #3 Construct emitted text directly from trellis state paths and verify match with ground truth text
<!-- AC:END -->
