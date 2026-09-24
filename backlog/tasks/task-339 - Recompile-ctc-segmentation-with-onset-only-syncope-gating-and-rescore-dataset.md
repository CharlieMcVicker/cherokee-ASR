---
id: TASK-339
title: Recompile ctc-segmentation with onset-only syncope gating and rescore dataset
status: Done
assignee:
  - '@agent'
created_date: '2026-09-16 16:44'
updated_date: '2026-09-16 16:45'
labels: []
dependencies: []
ordinal: 355000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Recompile and reinstall ctc-segmentation commit 80e7e8d0, verify unit tests in both repositories, and rescore the 1,389-sample syllabary dataset across clean and noisy conditions using cached LPZ.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reinstall ctc-segmentation in editable mode and run ctc-segmentation pytest suite
- [x] #2 Run full workshop-transcription pytest suite to verify zero regressions
- [x] #3 Run scripts/rescore_syllabary_dataset.py and record before/after CER/WER and 3-way divergence statistics
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Recompile and reinstall ctc-segmentation via pip install -e /Users/julietmcvicker/code/ctc-segmentation.\n2. Run pytest on ctc-segmentation test suite.\n3. Run pytest on workshop-transcription test suite.\n4. Run scripts/rescore_syllabary_dataset.py using cached LPZ matrices.\n5. Run 3-way comparative error breakdown (GT vs Greedy vs Guided) and report results.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Recompiled ctc-segmentation with commit 80e7e8d0 enforcing onset-only syncope gating. Verified 50/50 tests passed in ctc-segmentation and 274/274 tests passed in workshop-transcription. Rescored dataset: Clean test CER dropped to 4.11% (from 6.36%), clean all CER dropped to 3.69% (WER 23.52%). On valid noisy split, Guided (4.50% CER / 29.41% WER) beat Greedy (5.50% CER / 32.35% WER). Terminal vowel truncation was eliminated (610 errors -> 12).
<!-- SECTION:FINAL_SUMMARY:END -->
