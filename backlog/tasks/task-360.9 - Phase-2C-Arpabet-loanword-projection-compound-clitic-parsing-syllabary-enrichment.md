---
id: TASK-360.9
title: >-
  Phase 2C: Arpabet loanword projection, compound clitic parsing & syllabary
  enrichment
status: To Do
assignee: []
created_date: '2026-09-21 20:32'
labels: []
dependencies: []
parent_task_id: TASK-360
ordinal: 388300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Consolidate SyntheticTargetProjector, G2P fallback, compound Latin stem + Syllabary clitic discrimination (JayᎢ -> Jay + Ꭲ), and syllable alignment reconciliation into transcription.cherokee.codeswitching and transcription.cherokee.enrichment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 SyntheticTargetProjector, CodeSwitchedPreparer, and CodeSwitchedToken reside in transcription.cherokee.codeswitching
- [ ] #2 Static dictionary paths resolved relative to data/arpabet_alignment/dictionaries/english_loanwords_tth.json
- [ ] #3 SyllableAlignmentEngine, reconcile_phonetics, and reconcile_alignment_words reside in transcription.cherokee.enrichment
- [ ] #4 All Arpabet and syllabary enrichment unit tests pass with 0 pyright errors
<!-- AC:END -->
