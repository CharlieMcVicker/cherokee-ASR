---
id: TASK-323
title: >-
  Derive alignment phonetics directly from Cherokee Syllabary using
  syllabary_map
status: Done
assignee:
  - '@agent'
created_date: '2026-09-14 14:23'
updated_date: '2026-09-14 14:26'
labels: []
dependencies: []
ordinal: 339000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace reliance on legacy/respelled phonetic transcript fields with direct transliteration from Cherokee Syllabary using transcription/utils/syllabary_map.py (CHEROKEE_SYLLABARY_MAP / cherokee_to_bad_phonetics). This ensures base phonetic forms (e.g. Ꮯ -> tli, Ꭶ -> ka, Ꭷ -> kha) without forced pre-/post-aspiration respellings, allowing the phonotactic intrusion engine to dynamically detect and penalize optional laryngeals accurately.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 normalize_syllabary_for_alignment translates Cherokee Syllabary characters to base phonetics via syllabary_map
- [x] #2 Pipeline ingestion and alignment scripts prioritize Cherokee Syllabary text over legacy phonetic fields
- [x] #3 All 261 pytest tests and pyright type checking pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update transcription/utils/syllabary_map.py cherokee_to_bad_phonetics to support both uppercase (U+13A0-U+13F5) and lowercase (U+AB70-U+ABBF) Cherokee Syllabary Unicode characters.
2. Update transcription/alignment/normalizers.py normalize_syllabary_for_alignment to detect and transliterate Cherokee Syllabary using cherokee_to_bad_phonetics into clean base phonetics.
3. Update load_bible_chunks, pipeline.py, and alignment scripts to prioritize Cherokee Syllabary text ('cherokee' / 'syllabary') for phonetic derivation.
4. Verify all pytest tests and pyright pass.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Directly mapped Cherokee Syllabary Unicode characters to base phonetics via syllabary_map (cherokee_to_bad_phonetics(text.upper())) in normalize_syllabary_for_alignment, prioritized Cherokee Syllabary in load_bible_chunks, pipeline.py, and CTCSegmentationAligner, regenerated 100-verse benchmark comparison, and verified 261 pytest tests and 0 pyright errors.
<!-- SECTION:FINAL_SUMMARY:END -->
