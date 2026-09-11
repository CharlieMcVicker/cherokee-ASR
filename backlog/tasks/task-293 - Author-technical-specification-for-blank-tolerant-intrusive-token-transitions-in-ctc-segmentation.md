---
id: TASK-293
title: >-
  Author technical specification for blank-tolerant intrusive token transitions
  in ctc-segmentation
status: Done
assignee: []
created_date: '2026-09-11 18:37'
updated_date: '2026-09-11 18:38'
labels: []
dependencies: []
ordinal: 305000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write a detailed technical specification in docs/ and ../ctc-segmentation/doc/ specifying blank-tolerant multi-frame window intrusive token transitions (handling [PAD] frames between previous token, intrusive token, and next token in Cython DP trellis).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Document acoustic gap problem where CTC blanks [PAD] surround intrusive peaks
- [x] #2 Define updated DP trellis recurrence relation allowing intervening blanks (t_prev to t_intrusive to t_next)
- [x] #3 Specify backtracking and state_list emission for blank-tolerant detours
- [x] #4 Commit spec to docs/spec_blank_tolerant_intrusive_transitions.md and ../ctc-segmentation/doc/
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Authored technical specification for blank-tolerant intrusive token transitions (v2.0) in docs/spec_blank_tolerant_intrusive_transitions.md and ../ctc-segmentation/doc/spec_blank_tolerant_intrusive_transitions.md, detailing the multi-frame window stride recurrence and backtracking to capture intrusive peaks separated by CTC blank frames (such as uweluka -> uwelhuhka).
<!-- SECTION:FINAL_SUMMARY:END -->
