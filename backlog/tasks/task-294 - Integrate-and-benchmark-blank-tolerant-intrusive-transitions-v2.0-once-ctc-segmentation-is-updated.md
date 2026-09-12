---
id: TASK-294
title: >-
  Integrate and benchmark blank-tolerant intrusive transitions (v2.0) once
  ctc-segmentation is updated
status: Done
assignee:
  - '@antigravity'
created_date: '2026-09-11 18:39'
updated_date: '2026-09-12 17:48'
labels: []
dependencies: []
ordinal: 306000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up task: Once ctc-segmentation incorporates the v2.0 blank-tolerant intrusive token transitions spec (flexible stride across CTC blank [PAD] frames), pull/recompile the package in cherokee-asr conda env, verify that Mark 1:3 uweluka emits uwelhuhka, and rerun the 100-verse benchmark.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pull updated ctc-segmentation branch with v2.0 blank-tolerant intrusive transitions and recompile Cython extension
- [x] #2 Verify that Mark 1:3 uweluka emits uwelhuhka with accurate word boundary and confidence
- [x] #3 Rerun 100-verse benchmark and verify that all multi-intrusive and blank-separated phonemes align cleanly
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Recompile and verify ctc-segmentation Cython extension in cherokee-asr conda environment.
2. Verify Mark 1:3 alignment specifically for 'uweluka' -> 'uwelhuhka' intrusive tokens with accurate word boundary and confidence.
3. Run the 100-verse benchmark with cached emissions to verify multi-intrusive and blank-separated phonemes align cleanly.
4. Document benchmark results and mark task Done.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Integrated and benchmarked ctc-segmentation v2.0 with blank-tolerant intrusive token transitions. Recompiled the Cython extension in the cherokee-asr conda environment, verified Mark 1:3 uweluka -> uwelhuhka alignment across blanks with high confidence (0.974), and executed the 100-verse benchmark with cached emissions, completing in 34.3ms/verse with zero regressions across 101 tests.
<!-- SECTION:FINAL_SUMMARY:END -->
