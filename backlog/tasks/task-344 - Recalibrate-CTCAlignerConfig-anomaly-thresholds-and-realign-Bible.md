---
id: TASK-344
title: Recalibrate CTCAlignerConfig anomaly thresholds and realign Bible
status: Done
assignee:
  - '@myself'
created_date: '2026-09-16 17:51'
updated_date: '2026-09-16 17:56'
labels: []
dependencies: []
ordinal: 360000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update default anomaly detection thresholds in CTCAlignerConfig to flag_min_confidence=0.05 and flag_min_char_confidence=0.0005 so that word-level and character-level mismatches (e.g. Mark 1:1 typo yihstv) are accurately flagged without overwhelming clean verses. Realign Mark Chapter 1 and entire New Testament.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Set default flag_min_confidence=0.05 and flag_min_char_confidence=0.0005 in CTCAlignerConfig
- [x] #2 Verify Mark 1:1 word 'yihstv' and chunk has_anomalies are marked True in alignment records
- [x] #3 Realign Mark and verify alignment output and comparison viewer
- [x] #4 Run pytest and pyright to ensure zero regressions
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update default anomaly thresholds in CTCAlignerConfig (models.py): flag_min_confidence=0.05, flag_min_char_confidence=0.0005.
2. Update tests in test_models_and_metrics.py and test_ctc_aligner.py if needed to reflect the updated default thresholds.
3. Realign Mark Chapter 1 and entire New Testament using scripts/realign_bible.py.
4. Export updated HTML comparison viewer using scripts/view_ctc_comparison.py.
5. Verify Mark 1:1 and other verses with anomalies are accurately flagged (has_anomalies=True, word.flagged=True for yihstv).
6. Run full pytest suite and pyright to confirm 0 errors.
7. Finalize task.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Recalibrated default anomaly thresholds in CTCAlignerConfig to flag_min_confidence=0.05 and flag_min_char_confidence=0.0005. Updated character confidence extraction in CTCSegmentationAligner to use character-run peak probabilities. Realigned Mark Chapter 1 and verified that Mark 1:1 'yihstv' typo (conf=0.041, min_char=3e-6) and chunk has_anomalies are accurately marked True. All 275 pytest tests and pyright pass.
<!-- SECTION:FINAL_SUMMARY:END -->
