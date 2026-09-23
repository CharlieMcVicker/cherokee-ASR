---
id: TASK-360.1
title: >-
  Phase 1A: Core alignment models, DP aligner with distance metrics & CTC
  trellis engine
status: Done
assignee:
  - '@phase-1a-implementor'
created_date: '2026-09-21 20:24'
updated_date: '2026-09-22 15:11'
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
- [x] #1 Pure domain models (TextChunk, TokenEmission, WordInterval, AlignedChunk, AlignmentOutput, CTCAlignerConfig) reside in transcription.core.alignment.models
- [x] #2 NeedlemanWunschWordAligner and SlidingWindowDTWAligner reside in transcription.core.alignment.dp, consuming ModelOutput.decode_tokens() and configured by DistanceMetric
- [x] #3 DistanceMetric strategy interface and DefaultCERDistanceMetric reside in transcription.core.alignment.distance
- [x] #4 CTCSegmentationAligner resides in transcription.core.alignment.ctc, directly consuming ModelOutput.lpz and injected TextPreparerProtocol
- [x] #5 Alignment unit tests pass with pyright transcription reporting 0 errors
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create pure domain models in transcription/core/alignment/models.py
2. Define DistanceMetric protocol, DefaultCERDistanceMetric, and phonetic distance metrics in transcription/core/alignment/distance.py
3. Implement NeedlemanWunschWordAligner and SlidingWindowDTWAligner in transcription/core/alignment/dp.py consuming ModelOutput.decode_tokens() or TokenEmissions
4. Define TextPreparerProtocol and CTCSegmentationAligner in transcription/core/alignment/ctc.py consuming ModelOutput.lpz and injected text preparer
5. Wire up transcription/core/alignment/__init__.py and backward-compatibility re-exports in transcription/alignment/
6. Add comprehensive unit tests in transcription/core/alignment/tests/
7. Verify all tests pass and pyright transcription has 0 errors
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented pure language-agnostic core alignment package in transcription.core.alignment:
- models.py: Pure dataclasses TextChunk, TokenEmission, WordInterval, AlignedChunk, AlignmentMetrics, AlignmentOutput, and CTCAlignerConfig.
- distance.py: DistanceMetric protocol, DefaultCERDistanceMetric, PhonologicalDistanceMetric, LevenshteinDistanceMetric, CustomCallableDistanceMetric, and ConfusionMatrixCostMetric.
- dp.py: NeedlemanWunschWordAligner and SlidingWindowDTWAligner consuming ModelOutput (decode_tokens) or TokenEmission sequences, configurable via DistanceMetric.
- ctc.py: CTCSegmentationAligner consuming ModelOutput.lpz directly, with injected TextPreparerProtocol and vocabulary-filtered syncope/intrusive tokens.
- tests: Added test_core_alignment.py covering models, distance metrics, DP aligners, and CTC segmentation aligner.
- Verified with 412 passing pytest tests across the entire repository and 0 pyright errors.
<!-- SECTION:FINAL_SUMMARY:END -->
