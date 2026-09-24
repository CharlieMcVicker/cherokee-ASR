---
id: TASK-369.2
title: >-
  Implement Cherokee make_cherokee_projector factory and wire compound clitic
  segmentation in Tier 2
status: Done
assignee:
  - '@subagent'
created_date: '2026-09-24 14:52'
updated_date: '2026-09-24 15:20'
labels: []
dependencies: []
parent_task_id: TASK-369
ordinal: 401300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Parameterize Tier 1 SyntheticTargetProjector with Cherokee T/TH acoustic matrices and memoized dictionary using a clean factory function make_cherokee_projector(). Retain Cherokee compound clitic segmentation (JayᎢ -> Jay + Ꭲ) and Cherokee text preparation in transcription.cherokee.codeswitching. Relocate LibriSpeech dataset balancing to transcription.training.datasets.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement make_cherokee_projector() factory in transcription.cherokee.codeswitching returning generic SyntheticTargetProjector
- [x] #2 Maintain Cherokee compound clitic segmentation and script-level discrimination in transcription.cherokee.codeswitching
- [x] #3 Relocate LibriSpeech extraction / dataset balancing to transcription.training.datasets
- [x] #4 Add unit tests verifying Cherokee loanword projection (e.g. coffee -> khasi, JayᎢ -> tsei)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Implement make_cherokee_projector() in transcription/cherokee/codeswitching/
2. Wire transcription/cherokee/codeswitching/ to utilize Tier 1 core codeswitching engine
3. Relocate LibriSpeech dataset balancing / preparation to transcription/training/datasets/ or equivalent training module
4. Retain Cherokee compound clitic segmentation and script-level discrimination in transcription/cherokee/codeswitching/codeswitched_preparer.py
5. Add unit tests for Cherokee codeswitching and projector factory
6. Verify test suite and typechecking
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented make_cherokee_projector factory and SyntheticCherokeeTarget subclass in Tier 2 (transcription.cherokee.codeswitching), wired compound clitic segmentation, relocated LibriSpeech balancing to transcription.training.datasets.arpabet, and verified with 100% test pass (358 tests) and 0 pyright errors.
<!-- SECTION:FINAL_SUMMARY:END -->
