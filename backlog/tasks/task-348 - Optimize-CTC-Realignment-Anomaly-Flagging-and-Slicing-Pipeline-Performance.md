---
id: TASK-348
title: 'Optimize CTC Realignment, Anomaly Flagging, and Slicing Pipeline Performance'
status: To Do
assignee: []
created_date: '2026-09-17 18:30'
labels:
  - alignment
  - performance
  - optimization
dependencies: []
priority: high
ordinal: 364000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Optimize post-alignment processing and anomaly flagging in realign_bible.py and CTCSegmentationAligner when chapter acoustic log-probabilities (lpz) are cached, eliminating redundant neural forward passes, vectorizing confidence calculations, and deferring audio exports.
<!-- SECTION:DESCRIPTION:END -->
