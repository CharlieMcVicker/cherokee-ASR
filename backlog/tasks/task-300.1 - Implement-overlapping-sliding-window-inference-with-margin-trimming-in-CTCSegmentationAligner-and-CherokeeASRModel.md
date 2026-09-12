---
id: TASK-300.1
title: >-
  Implement overlapping sliding-window inference with margin trimming in
  CTCSegmentationAligner and CherokeeASRModel
status: Done
assignee: []
created_date: '2026-09-12 19:26'
updated_date: '2026-09-12 19:29'
labels: []
dependencies: []
parent_task_id: TASK-300
ordinal: 313000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement overlapping sliding-window inference with context buffers and margin trimming to eliminate boundary distortion on long chapter audio.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add sliding-window inference with configurable chunk size, context overlap, and margin trimming in CherokeeASRModel / CTCSegmentationAligner
- [x] #2 Update get_logits_cached and extract_logits to use overlapping windowing on long inputs
- [x] #3 Ensure seamless logit concatenation across chunk boundaries
<!-- AC:END -->
