---
id: TASK-399
title: 'Phase 1: Orthography & Spelling Normalization Engine (Standard 2 TTH)'
status: Done
assignee:
  - '@self'
created_date: '2026-11-07 21:25'
updated_date: '2026-11-07 21:34'
labels: []
dependencies: []
modified_files:
  - digohwelisgi/cherokee/orthography/tones.py
  - digohwelisgi/cherokee/orthography/orthography.py
  - digohwelisgi/cherokee/orthography/__init__.py
  - digohwelisgi/cherokee/orthography/tests/test_tones_normalization.py
ordinal: 432300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enforce Standard 2 (Contextual Phonotactic TTH), glottal stop canonicalization, inter-vocalic hiatus resolution, and strict consonant cascade ordering across Cherokee orthography normalization.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Word-initial /s/ remains bare s- across all words while medial postvocalic /s/ becomes hs
- [x] #2 All glottal stop variants unify to ASCII apostrophe and illegal post-consonantal glottal stops (*C') are stripped
- [x] #3 Adjacent vowels across syllable boundaries receive epenthetic glottal stop insertion (V1V2 -> V1'V2)
- [x] #4 Consonant cascade rewrites execute in priority order without destructive double-conversions
- [x] #5 All existing and new orthography unit tests pass with 0 Pyright errors
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect existing orthography & tone functions and tests
2. Implement standardize_glottal_stops, resolve_vowel_hiatus, and updated respell_consonants in tones.py
3. Update convert_orthography in orthography.py and exports in __init__.py
4. Add unit tests in digohwelisgi/cherokee/orthography/tests/test_tones_normalization.py and pass Gate 1
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Phase 1 Orthography & Spelling Normalization Engine (Standard 2 TTH).
- Fixed contextual sibilant pre-aspiration regex in tones.py so word-initial /s/ stays bare s- while medial postvocalic /s/ becomes hs.
- Added standardize_glottal_stops, resolve_vowel_hiatus, and collapse_double_vowels.
- Updated convert_orthography and added test_tones_normalization.py.
- Verified 58/58 unit tests pass and 0 Pyright errors.
<!-- SECTION:FINAL_SUMMARY:END -->
