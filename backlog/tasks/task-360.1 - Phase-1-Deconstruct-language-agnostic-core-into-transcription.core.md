---
id: TASK-360.1
title: >-
  Phase 1A: Core alignment models, DP aligner with distance metrics & CTC
  trellis engine
status: To Do
assignee: []
created_date: '2026-09-21 20:24'
updated_date: '2026-09-21 20:41'
labels: []
dependencies: []
parent_task_id: TASK-360
ordinal: 387100
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extract pure domain models (TextChunk, TokenEmission, WordInterval, AlignedChunk, AlignmentOutput, CTCAlignerConfig), NeedlemanWunschWordAligner, SlidingWindowDTWAligner, and generic CTCSegmentationAligner into transcription.core.alignment. Both aligners consume ModelOutput (CTC consumes output.lpz, DP consumes output.decode_tokens()). DP aligner is cleanly configured by pluggable DistanceMetric strategy classes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Pure domain models (TextChunk, TokenEmission, WordInterval, AlignedChunk, AlignmentOutput, CTCAlignerConfig) reside in transcription.core.alignment.models
- [ ] #2 NeedlemanWunschWordAligner and SlidingWindowDTWAligner reside in transcription.core.alignment.dp, consuming ModelOutput.decode_tokens() and configured by DistanceMetric
- [ ] #3 DistanceMetric strategy interface and DefaultCERDistanceMetric reside in transcription.core.alignment.distance
- [ ] #4 CTCSegmentationAligner resides in transcription.core.alignment.ctc, directly consuming ModelOutput.lpz and injected TextPreparerProtocol
- [ ] #5 Alignment unit tests pass with pyright transcription reporting 0 errors
<!-- AC:END -->
