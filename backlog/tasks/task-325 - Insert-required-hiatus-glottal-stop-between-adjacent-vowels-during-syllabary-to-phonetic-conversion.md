---
id: TASK-325
title: >-
  Insert required hiatus glottal stop between adjacent vowels during
  syllabary-to-phonetic conversion
status: Done
assignee:
  - '@agent'
created_date: '2026-09-14 14:41'
updated_date: '2026-09-14 14:43'
labels: []
dependencies: []
ordinal: 341000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In Cherokee phonology, vowel hiatus does not exist. Whenever two vowels are adjacent across syllables without an onset consonant (e.g., ᎢᎾᎨᎢ -> inake'i, ᎯᎠ -> hi'a, ᎠᏍᎦᏅᏨᎢ -> askanvtsv'i), an intervening glottal stop /'/ is phonologically and acoustically required. Update syllabary-to-phonetic conversion in syllabary_map.py and normalize_syllabary_for_alignment to insert glottal stops between adjacent vowels as part of the canonical ground-truth path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 cherokee_to_bad_phonetics and normalize_syllabary_for_alignment insert glottal stop between adjacent vowels (e.g. inake'i, hi'a, askanvtsv'i)
- [x] #2 Single vowels and consonant-separated vowels are unaffected
- [x] #3 All pytest unit tests and pyright pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update cherokee_to_bad_phonetics in transcription/utils/syllabary_map.py to insert glottal stop /'/ between adjacent vowels ([aeiouvAEIOUV](?=[aeiouvAEIOUV])).
2. Ensure normalize_syllabary_for_alignment in transcription/alignment/normalizers.py preserves hiatus glottals.
3. Update unit tests in test_syllabary_map.py, test_normalizers.py, test_phonotactics.py, and test_ctc_aligner.py to cover hiatus glottal stops.
4. Run full test suite and pyright to verify zero regressions.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Inserted required hiatus glottal stop /'/ between adjacent vowels in cherokee_to_bad_phonetics and normalize_syllabary_for_alignment, preserving phonological glottals during punctuation normalization. Verified with 262 unit tests passing and benchmark evaluation showing 18 verses with anomalies.
<!-- SECTION:FINAL_SUMMARY:END -->
