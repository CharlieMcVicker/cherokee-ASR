---
id: TASK-307
title: >-
  Calibrate intrusive token penalty in CTCSegmentationAligner to suppress
  spurious intrusions
status: To Do
assignee: []
created_date: '2026-09-12 20:01'
updated_date: '2026-09-14 13:29'
labels: []
dependencies: []
ordinal: 323000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tune intrusive_penalty in CTCSegmentationAligner from 0.1 to 0.8-1.0 to prevent excessive insertion of spurious aspiration and glottal stop tokens in CTC trellis segmentation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Increase default intrusive_penalty from 0.1 to 0.8-1.0 in CTCSegmentationAligner
- [ ] #2 Rerun 100-verse benchmark to verify reduction of false-positive intrusive tokens
- [ ] #3 Verify no degradation in legitimate intrusive token alignments (e.g., hi'a)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Superseded by parent TASK-321 (TASK-318 and TASK-319 per-token intrusive calibration).
<!-- SECTION:NOTES:END -->
