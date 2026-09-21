---
id: TASK-338
title: >-
  Rebuild ctc-segmentation with departure-boundary contrastive gating and
  rescore dataset
status: Done
assignee:
  - '@agent'
created_date: '2026-09-16 16:41'
updated_date: '2026-09-16 16:42'
labels: []
dependencies: []
ordinal: 354000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Recompile and reinstall ctc-segmentation with departure-boundary contrastive gating for vowel syncope into blank. Run pytest and rescore all 1,389 dataset samples using cached LPZ to evaluate improvement in terminal vowel retention and CER/WER.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Recompile ctc-segmentation in editable mode and run ctc-segmentation pytest suite
- [x] #2 Run full workshop-transcription pytest suite to verify zero regressions
- [x] #3 Run scripts/rescore_syllabary_dataset.py and compare CER/WER and divergence statistics against greedy baseline
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Recompile and reinstall ctc-segmentation via pip install -e /Users/julietmcvicker/code/ctc-segmentation.\n2. Run pytest on ctc-segmentation test suite.\n3. Run pytest on workshop-transcription test suite.\n4. Run scripts/rescore_syllabary_dataset.py using cached LPZ matrices.\n5. Compute 3-way comparative analysis (GT vs Greedy vs Guided) and report CER/WER metrics.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Recompiled ctc-segmentation with commit 44822dcf. Verified all 52 tests in ctc-segmentation and 274 tests in workshop-transcription pass. Rescored dataset: Clean all CER 5.70% (test CER 6.36%), Noisy all CER 6.29% (test CER 6.70%). Analysis revealed t_departure algebraically reduced to t + offset_sum (arrival frame in silence), causing inter-word blank syncope to over-trigger.
<!-- SECTION:FINAL_SUMMARY:END -->
