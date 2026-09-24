---
id: TASK-369.2
title: >-
  Implement Cherokee make_cherokee_projector factory and wire compound clitic
  segmentation in Tier 2
status: To Do
assignee: []
created_date: '2026-09-24 14:52'
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
- [ ] #1 Implement make_cherokee_projector() factory in transcription.cherokee.codeswitching returning generic SyntheticTargetProjector
- [ ] #2 Maintain Cherokee compound clitic segmentation and script-level discrimination in transcription.cherokee.codeswitching
- [ ] #3 Relocate LibriSpeech extraction / dataset balancing to transcription.training.datasets
- [ ] #4 Add unit tests verifying Cherokee loanword projection (e.g. coffee -> khasi, JayᎢ -> tsei)
<!-- AC:END -->
