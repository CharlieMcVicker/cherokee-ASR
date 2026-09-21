---
id: TASK-356
title: >-
  Rename syllabary transliteration helper and enforce fail-fast token validation
  in pipeline
status: Done
assignee: []
created_date: '2026-09-21 17:04'
updated_date: '2026-09-21 17:10'
labels: []
dependencies: []
ordinal: 382000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Code review identified pejorative naming in syllabary_map.py and silent word repeating on token index overflow in pipeline.py.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Rename cherokee_to_bad_phonetics to syllabary_to_phonetics across syllabary_map.py and all call sites (orthography.py, tests)
- [x] #2 Raise explicit ValueError on token length mismatch / out-of-bounds in _build_syllabary_word_tier and _build_english_word_tier in pipeline.py
- [x] #3 Add unit tests verifying ValueError is raised on word count mismatches
- [x] #4 All existing tests pass with 0 errors
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Renamed cherokee_to_bad_phonetics to syllabary_to_phonetics across all modules and enforced fail-fast ValueError on word tier length mismatches with unit tests.
<!-- SECTION:FINAL_SUMMARY:END -->
