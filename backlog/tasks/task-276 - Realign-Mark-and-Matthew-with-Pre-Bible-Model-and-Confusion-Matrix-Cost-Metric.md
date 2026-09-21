---
id: TASK-276
title: Realign Mark and Matthew with Pre-Bible Model and Confusion Matrix Cost Metric
status: Done
assignee:
  - '@myself'
created_date: '2026-09-10 17:50'
updated_date: '2026-09-14 13:26'
labels:
  - alignment
  - new-testament
  - dataset
dependencies: []
ordinal: 288000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute audio-transcript alignment across all chapters of Mark (16 chapters) and Matthew (28 chapters) using baseline model charliemcvicker/asr-cherokee:5464d15, CachedASREmissionsExtractor (from TASK-279), and ConfusionMatrixCostMetric configured with confusion_cost_matrix_prebible.json. Save full alignment records with timestamps, reference syllabary, ASR hypothesis, and per-verse alignment distance cost.\n\nDependencies: Requires TASK-275 (Confusion Matrix Cost Artifact) and TASK-279 (Disk-Cached ASR Emissions Extractor Layer).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Update chapter alignment pipeline to accept custom distance metric and model revision
- [ ] #2 Realign Mark (chapters 1-16) and record per-verse alignment scores and metadata
- [ ] #3 Realign Matthew (chapters 1-28) and record per-verse alignment scores and metadata
- [ ] #4 Export intermediate alignment records containing verse audio paths, start/end times, reference text, ASR hypothesis text, and cost score
- [ ] #5 Wrap emissions extraction in CachedASREmissionsExtractor (from TASK-279) to persist chapter ASR outputs to disk
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Superseded by continuous CTC segmentation alignment pipeline (TASK-285 through TASK-320). Full New Testament realignment is tracked and executed under TASK-320.
<!-- SECTION:FINAL_SUMMARY:END -->
