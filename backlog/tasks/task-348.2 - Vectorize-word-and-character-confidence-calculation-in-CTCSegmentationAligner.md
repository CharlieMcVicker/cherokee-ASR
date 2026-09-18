---
id: TASK-348.2
title: Vectorize word and character confidence calculation in CTCSegmentationAligner
status: To Do
assignee: []
created_date: '2026-09-17 18:30'
labels:
  - alignment
  - performance
  - ctc_aligner
dependencies: []
parent_task_id: TASK-348
priority: medium
ordinal: 366000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In transcription/alignment/ctc_aligner.py, optimize CTCSegmentationAligner._extract_word_intervals() by vectorizing frame-by-frame Python loops over char_probs and state_list using NumPy operations or slices, speeding up character peak detection and anomaly confidence scoring.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 _extract_word_intervals replaces pure Python frame iteration with efficient vector/slice operations for character peak and logprob extraction
- [ ] #2 Calculated word_conf, min_char_prob, and flagged boolean outputs remain numerically consistent with existing test suites
- [ ] #3 pytest and pyright pass with zero regressions
<!-- AC:END -->
