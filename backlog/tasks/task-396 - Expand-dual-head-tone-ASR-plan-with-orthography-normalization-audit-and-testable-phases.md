---
id: TASK-396
title: >-
  Expand dual head tone ASR plan with orthography normalization audit and
  testable phases
status: Done
assignee:
  - '@self'
created_date: '2026-11-07 20:56'
updated_date: '2026-11-07 20:58'
labels: []
dependencies: []
modified_files:
  - dual_head_tone_asr_plan.md
ordinal: 429300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Audit all Cherokee spelling, phonotactic, and diacritic normalization steps needed for dual-head ASR (glottal unification, hiatus insertion, aspirated sonorant ordering, cluster normalization, long vowel vs mora representation, colon stripping, and non-Cherokee character cleaning). Structure implementation phases with distinct unit and integration testable milestones.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Identify and document all spelling/orthographic normalization requirements across the pipeline
- [x] #2 Restructure dual_head_tone_asr_plan.md phases into distinct unit-testable and integration-testable milestones
- [x] #3 Update dual_head_tone_asr_plan.md and verify all phase dependencies and criteria
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Audited all Cherokee spelling and phonotactic normalization requirements (Standard 2 contextual pre-aspiration, glottal stop unification, vowel hiatus resolution, consonant cascade ordering, mora factoring, and formatting/non-Cherokee character filtering). Restructured dual_head_tone_asr_plan.md into six distinct phases (Orthography, Moraic Generator, Dual Dataset, Model Architecture, Training Pipeline, Core Inference) each equipped with explicit Unit Tests and an automated Integration Gate.
<!-- SECTION:FINAL_SUMMARY:END -->
