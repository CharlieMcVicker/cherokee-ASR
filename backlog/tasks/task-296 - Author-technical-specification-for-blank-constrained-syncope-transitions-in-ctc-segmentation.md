---
id: TASK-296
title: >-
  Author technical specification for blank-constrained syncope transitions in
  ctc-segmentation
status: Done
assignee:
  - '@agent'
created_date: '2026-09-12 18:10'
updated_date: '2026-09-12 18:11'
labels: []
dependencies: []
ordinal: 308000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Author a comprehensive technical specification for ctc-segmentation engine to prevent accidental non-syncope consonant dropping during syncope transitions by enforcing that stride-2 jumps (c - 3 -> c) are only permitted when intermediate token c - 2 is a blank/PAD.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Document root cause of collateral consonant deletion in packed-character ground truth matrices
- [x] #2 Define updated syncope transition recurrence relation restricting c - 3 jumps to blank intermediate states
- [x] #3 Specify Cython (cython_fill_table) and Python backtracking implementation changes
- [x] #4 Provide verification test cases including Mark 1:1 yihstv typo detection
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Author comprehensive markdown specification docs/spec_blank_constrained_syncope_transitions.md detailing problem, math recurrence, Cython/Python code diffs, and verification criteria.\n2. Copy spec to ../ctc-segmentation/doc/spec_blank_constrained_syncope_transitions.md.\n3. Verify file contents and ensure all ACs are met.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Authored technical specification in docs/spec_blank_constrained_syncope_transitions.md and ../ctc-segmentation/doc/spec_blank_constrained_syncope_transitions.md detailing the root cause of collateral consonant omission during syncope jumps in packed-character ground truth matrices (such as Mark 1:1 yihstv), defining updated forward trellis recurrence and backtracking algorithms restricting 2-step syncope jumps to blank intermediate states, and providing verification criteria.
<!-- SECTION:FINAL_SUMMARY:END -->
