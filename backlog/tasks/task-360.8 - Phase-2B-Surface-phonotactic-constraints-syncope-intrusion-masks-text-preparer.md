---
id: TASK-360.8
title: >-
  Phase 2B: Surface phonotactic constraints, syncope/intrusion masks & text
  preparer
status: To Do
assignee: []
created_date: '2026-09-21 20:32'
updated_date: '2026-09-21 20:41'
labels: []
dependencies: []
parent_task_id: TASK-360
ordinal: 388200
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Consolidate Cherokee surface constraints (*HH, *C'), vowel syncope masks, intrusion site masks, prepare_cherokee_text into transcription.cherokee.phonotactics, and relocate PhonologicalConfusionCostMetric / ConfusionMatrixCostMetric to transcription.cherokee.distance. Retire normalizers.py in favor of direct convert_orthography calls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Surface phonotactic rules and transition masks reside in transcription.cherokee.phonotactics.phonotactics
- [ ] #2 prepare_cherokee_text implements TextPreparerProtocol compatible with Tier 1 CTCSegmentationAligner
- [ ] #3 PhonologicalConfusionCostMetric and ConfusionMatrixCostMetric reside in transcription.cherokee.distance as pluggable DistanceMetrics for the DP aligner
- [ ] #4 Redundant normalizers.py retired in favor of direct convert_orthography calls
- [ ] #5 Unit tests pass with pyright transcription reporting 0 errors
<!-- AC:END -->
