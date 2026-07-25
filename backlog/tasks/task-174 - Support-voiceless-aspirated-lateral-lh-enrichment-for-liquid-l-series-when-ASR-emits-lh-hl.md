---
id: TASK-174
title: >-
  Support voiceless/aspirated lateral lh enrichment for liquid l series when ASR
  emits lh/hl
status: Done
assignee:
  - '@agent'
created_date: '2026-07-25 20:45'
updated_date: '2026-07-25 20:46'
labels: []
dependencies: []
ordinal: 170000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When ASR emits 'lh' for an l-series syllabary character (la, le, li, lo, lu, lv), update enrich_syllabary.py so that the l-series base transitions to lh-series (e.g. li -> lhi, or li -> lh when syncopated).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Support l-series (la, le, li, lo, lu, lv) -> lh-series (lha, lhe, lhi, lho, lhu, lhv) toggle when ASR window contains 'lh'
- [x] #2 Ensure syncopation of li + ASR 'lh' produces 'lh' rather than 'l'
- [x] #3 Add unit test for l-series lateral aspiration and syncopated 'lh'
- [x] #4 All unit tests pass cleanly
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. In enrich_syllabary.py _enrich_single_syllable, add a lateral aspiration check for l-series base syllables (la, le, li, lo, lu, lv) when emitted contains 'lh'.\n2. Shift curr from l... to lh... before syncopation processing so that syncopation of lhi produces 'lh'.\n3. Add unit test in test_enrich_syllabary.py verifying 'li' + ASR 'lh' -> 'lh'.\n4. Run unit tests and verify.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated enrich_syllabary.py to handle lateral aspiration shift for liquid l-series (la, le, li, lo, lu, lv) -> lh-series when ASR emitted slice contains 'lh', preserving voiceless lateral aspiration upon vowel syncopation (e.g. li + ASR 'lh' -> 'lh'). Added unit test in test_enrich_syllabary.py and verified all 44 unit tests pass cleanly.
<!-- SECTION:FINAL_SUMMARY:END -->
