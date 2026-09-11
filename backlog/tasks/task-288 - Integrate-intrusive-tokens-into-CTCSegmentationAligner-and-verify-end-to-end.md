---
id: TASK-288
title: Integrate intrusive tokens into CTCSegmentationAligner and verify end-to-end
status: Done
assignee:
  - '@agent'
created_date: '2026-09-11 17:46'
updated_date: '2026-09-11 17:46'
labels: []
dependencies: []
ordinal: 300000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update CTCSegmentationAligner to expose intrusive_tokens and intrusive_penalty parameters, pass them to CtcSegmentationParameters, and verify extraction of intrusive h and glottal stops directly in aligned word intervals.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add intrusive_tokens and intrusive_penalty parameters to CTCSegmentationAligner
- [x] #2 Verify intrusive h and glottal stop extraction on Cherokee Bible audio
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Integrated intrusive_tokens and intrusive_penalty into CTCSegmentationAligner, rebuilt ctc-segmentation Cython extension in-place, and verified end-to-end extraction of intrusive h and glottal stops directly on Cherokee audio.
<!-- SECTION:FINAL_SUMMARY:END -->
