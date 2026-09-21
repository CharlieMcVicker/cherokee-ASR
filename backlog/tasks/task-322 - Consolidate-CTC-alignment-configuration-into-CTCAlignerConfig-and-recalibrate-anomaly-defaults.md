---
id: TASK-322
title: >-
  Consolidate CTC alignment configuration into CTCAlignerConfig and recalibrate
  anomaly defaults
status: Done
assignee:
  - '@agent'
created_date: '2026-09-14 14:05'
updated_date: '2026-09-14 14:12'
labels: []
dependencies: []
ordinal: 338000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Consolidate all CTC alignment parameters (penalties, min logprobs, stride, anomaly thresholds, phonotactics enforcement, window/chunk parameters, caching) into a strongly-typed CTCAlignerConfig dataclass. Remove redundant CLI hyperparameters and exploded kwargs from realign_bible.py and pipeline.py. Recalibrate default anomaly detection threshold to 0.0 (and 1e-6 for strict char-level checks) to eliminate false positive anomaly flagging across clean verses, and regenerate benchmark evaluation artifacts and docs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CTCAlignerConfig dataclass is defined in transcription/alignment/models.py and adopted across CTCSegmentationAligner, pipeline.py, and realign_bible.py
- [x] #2 Redundant hyperparameter CLI flags are stripped from realign_bible.py and exploded kwargs removed from pipeline.py
- [x] #3 Default flag_min_char_confidence is set to 0.0 (and benchmark evaluates 0.0 with 12/100 anomalous verses vs previous 100/100)
- [x] #4 Benchmark comparison artifact runs/evaluation/ctc_segmentation_100_verses_comparison.json and docs/alignment.md are regenerated and accurate
- [x] #5 All unit and regression tests in pytest pass and pyright transcription reports 0 errors
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Define CTCAlignerConfig dataclass in transcription/alignment/models.py with clean defaults and types.
2. Refactor CTCSegmentationAligner to accept config: Optional[CTCAlignerConfig] = None (defaulting to CTCAlignerConfig()) and set default flag_min_char_confidence=0.0.
3. Update transcription/new_testament/pipeline.py align_chapter to accept aligner_config: Optional[CTCAlignerConfig] = None, removing exploded individual penalty/mask kwargs.
4. Clean up scripts/realign_bible.py by removing redundant hyperparameter CLI arguments.
5. Update benchmark script scripts/benchmark_ctc_segmentation_100_verses.py to use CTCAlignerConfig with default flag_min_char_confidence=0.0 and re-run benchmark to regenerate runs/evaluation/ctc_segmentation_100_verses_comparison.json and update docs/alignment.md.
6. Verify all 261 pytest tests and pyright transcription pass cleanly.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Consolidated all CTC alignment parameters into strongly-typed CTCAlignerConfig dataclass. Updated CTCSegmentationAligner, pipeline.py, realign_bible.py, calibrate_intrusion_penalties.py, and benchmark_ctc_segmentation_100_verses.py. Recalibrated default flag_min_char_confidence=0.0. Regenerated runs/evaluation/ctc_segmentation_100_verses_comparison.json showing 12/100 anomalous verses and 81 flagged words. Updated docs/alignment.md. Verified 261 pytest tests pass and pyright transcription reports 0 errors.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Consolidated CTC alignment configuration into frozen dataclass CTCAlignerConfig across CTCSegmentationAligner, align_chapter, realign_bible.py, and benchmark scripts. Recalibrated default flag_min_char_confidence to 0.0 (eliminating false positive anomalies while retaining 12/100 true anomalies and typo detection like Mark 1:1 yihstv). Regenerated benchmark artifacts in runs/evaluation/ and updated docs/alignment.md. Verified with 261 passing pytest tests and 0 pyright errors.
<!-- SECTION:FINAL_SUMMARY:END -->
