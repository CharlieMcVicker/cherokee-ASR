---
id: TASK-388
title: >-
  Extract language-agnostic Wav2Vec2 trainer and dataset collators into
  digohwelisgi.core
status: To Do
assignee: []
created_date: '2026-11-03 19:45'
labels: []
dependencies: []
ordinal: 421300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Refactor the generic components of the Wav2Vec2 CTC trainer, PyTorch collators, and audio length grouping out of digohwelisgi.cherokee.training into Tier 1 (digohwelisgi.core.training) to provide a clean, language-agnostic foundational trainer.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Identify and isolate language-agnostic training components from Cherokee-specific trainers
- [ ] #2 Define generic Wav2Vec2 trainer and dataset protocols in Tier 1
- [ ] #3 Refactor Cherokee training scripts to consume the generic core trainer
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 All unit tests pass and training pipelines function without regressions
<!-- DOD:END -->
