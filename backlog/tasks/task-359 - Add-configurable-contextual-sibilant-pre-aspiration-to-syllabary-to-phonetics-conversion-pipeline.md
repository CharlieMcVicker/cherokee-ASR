---
id: TASK-359
title: >-
  Add configurable contextual sibilant pre-aspiration to syllabary-to-phonetics
  conversion pipeline
status: To Do
assignee: []
created_date: '2026-09-21 19:57'
labels: []
dependencies: []
ordinal: 385000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Certain Cherokee ASR models emit pre-aspiration (hs) postvocalically and preconsonantally but emit bare s word-initially and in affricates (ts/tsh). The syllabary-to-phonetics conversion pipeline should support modeling contextual pre-aspiration via a configurable flag (e.g. contextual_preaspiration) so that alignment targets match the specific acoustic model regime.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Implement contextual pre-aspiration rule suppressing h before s word-initially (^s) and in affricates (ts/tsh) during syllabary-to-TTH conversion
- [ ] #2 Add configurable flag to allow switching between contextual pre-aspiration and unconditional hs conversion
- [ ] #3 Add unit tests covering both contextual and unconditional conversion modes across word-initial, medial, and affricate positions
<!-- AC:END -->
