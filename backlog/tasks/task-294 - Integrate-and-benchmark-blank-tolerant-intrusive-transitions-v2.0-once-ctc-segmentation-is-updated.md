---
id: TASK-294
title: >-
  Integrate and benchmark blank-tolerant intrusive transitions (v2.0) once
  ctc-segmentation is updated
status: To Do
assignee: []
created_date: '2026-09-11 18:39'
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
- [ ] #1 Pull updated ctc-segmentation branch with v2.0 blank-tolerant intrusive transitions and recompile Cython extension
- [ ] #2 Verify that Mark 1:3 uweluka emits uwelhuhka with accurate word boundary and confidence
- [ ] #3 Rerun 100-verse benchmark and verify that all multi-intrusive and blank-separated phonemes align cleanly
<!-- AC:END -->
