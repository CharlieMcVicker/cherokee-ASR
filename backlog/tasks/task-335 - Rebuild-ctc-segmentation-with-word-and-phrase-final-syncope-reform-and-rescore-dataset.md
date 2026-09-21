---
id: TASK-335
title: >-
  Rebuild ctc-segmentation with word- and phrase-final syncope reform and
  rescore dataset
status: Done
assignee:
  - '@agent'
created_date: '2026-09-16 16:16'
updated_date: '2026-09-16 16:19'
labels: []
dependencies: []
ordinal: 351000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Recompile and reinstall the updated ctc-segmentation package containing the word- and phrase-final vowel syncope fixes. Run test suite and rescore against the 1,389-sample syllabary dataset across clean and noisy conditions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reinstall ctc-segmentation in editable mode and run ctc-segmentation test suite
- [x] #2 Run full workshop-transcription pytest suite to verify zero regressions
- [x] #3 Run scripts/rescore_syllabary_dataset.py and record before/after CER and WER metrics
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Recompile and reinstall ctc-segmentation via pip install -e /Users/julietmcvicker/code/ctc-segmentation.\n2. Run pytest in ctc-segmentation repo to ensure its tests pass.\n3. Run pytest in workshop-transcription repo.\n4. Run scripts/rescore_syllabary_dataset.py on clean and noisy conditions.\n5. Compare CER/WER with previous runs and analyze error breakdown.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Recompiled ctc-segmentation with word- and phrase-final vowel syncope fixes. Verified 50/50 tests passed in ctc-segmentation and 273/273 tests passed in workshop-transcription. Rescored all 1,389 samples: clean test CER dropped from 7.81% to 7.03% (all CER 7.16% -> 6.56%), noisy test CER dropped from 8.06% to 7.35% (all CER 7.54% -> 6.97%). Overall clean WER dropped by 4.18% (46.89% -> 42.71%).
<!-- SECTION:FINAL_SUMMARY:END -->
