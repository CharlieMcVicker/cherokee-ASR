---
id: TASK-317
title: >-
  Integrate RAG-derived phonotactic rules into
  transcription/alignment/phonotactics.py
status: Done
assignee:
  - '@agent'
created_date: '2026-09-14 13:09'
updated_date: '2026-09-14 13:20'
labels: []
dependencies: []
modified_files:
  - transcription/alignment/phonotactics.py
  - transcription/alignment/tests/test_phonotactics.py
ordinal: 333000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Once the RAG phonotactics output is received, implement transcription/alignment/phonotactics.py with functions for generating phonotactic intrusion site masks and syncope masks in the /t/ /th/ and /k/ /kh/ /kw/ /khw/ consonant system. Add dedicated unit tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement phonotactics.py with Cherokee rule parser
- [x] #2 Generate accurate syncope_mask and intrusion_site_mask
- [x] #3 Handle Cherokee digraphs (kw, khw, th, kh, tsh, tlh, lh, nh, wh, yh)
- [x] #4 Add unit tests in test_phonotactics.py
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Define domain types and constants (PhonemeCategory, PhonotacticToken, PhonotacticAnalysis) in transcription/alignment/phonotactics.py.
2. Implement pure Cherokee phonetic tokenizer supporting all single vowels, consonants, and multi-char digraphs/trigraphs (kw, kwh, khw, th, kh, tsh, tlh, lh, nh, wh, yh, slh).
3. Implement get_syncope_mask(text: str) -> list[bool] for valid vowel deletion positions based on phonotactic rules.
4. Implement get_intrusion_site_mask(text: str) -> list[bool] identifying valid intrusion environments for /h/ and /'/ while enforcing *HH, *C', and *ChR constraints.
5. Implement analyze_phonotactics(text: str) -> PhonotacticAnalysis providing unified tokenization and phonotactic masks.
6. Create comprehensive test suite in transcription/alignment/tests/test_phonotactics.py testing all digraphs, edge cases, constraint violations, syncope, and intrusion masks.
7. Verify full test suite passes with pytest and passes pyright static typing.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented transcription/alignment/phonotactics.py with Cherokee phonetic tokenizer, syncope masking (get_syncope_mask), and intrusive laryngeal candidate masking (get_intrusion_site_mask) adhering to standardized Cherokee orthography (kwh, tlh, tsh, hs, lh, nh, wh, yh, '). Added comprehensive unit tests in test_phonotactics.py, passing 100% on pytest and pyright.
<!-- SECTION:FINAL_SUMMARY:END -->
