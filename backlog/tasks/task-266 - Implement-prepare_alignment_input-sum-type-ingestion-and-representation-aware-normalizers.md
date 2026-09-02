---
id: TASK-266
title: >-
  Implement prepare_alignment_input sum-type ingestion and representation-aware
  normalizers
status: Done
assignee:
  - '@agent-normalizers'
created_date: '2026-09-02 15:33'
updated_date: '2026-09-02 15:39'
labels:
  - alignment
  - refactor
  - phonetics
dependencies: []
priority: high
ordinal: 268000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Refactor alignment ingestion to decide text normalizers based on input type (Bible Syllabary vs generic Phonetic):
1. In transcription/alignment/normalizers.py:
   - Implement normalize_syllabary_for_alignment(text: str) -> str (strips aspiration 'h', collapses whitespace, normalizes consonants).
   - Implement normalize_phonetics_for_alignment(text: str) -> str (preserves aspiration 'h', normalizes consonants).
2. In transcription/alignment/ingestion.py:
   - Implement prepare_alignment_input(...) consuming sum-type input arguments (bible_metadata vs chunk_list) and returning (chunks, source_lookup, chunk_normalizer, emissions_normalizer).
   - Syllabary input uses normalize_syllabary_for_alignment for both chunk and emission normalizers.
   - Generic phonetic input uses normalize_phonetics_for_alignment.
3. In transcription/alignment/cli.py:
   - Use prepare_alignment_input and initialize NeedlemanWunschWordAligner with the resolved normalizers.
4. Add unit tests covering normalizer strategies and prepare_alignment_input.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement normalize_syllabary_for_alignment and normalize_phonetics_for_alignment in normalizers.py
- [x] #2 Implement prepare_alignment_input in ingestion.py returning (chunks, source_lookup, chunk_normalizer, emissions_normalizer)
- [x] #3 Refactor cli.py and pipeline callers to use prepare_alignment_input
- [x] #4 Add unit tests in test_normalizers.py and test_ingestion.py
- [x] #5 Verify all unit tests pass and pyright reports 0 errors
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented representation-aware normalizers and prepare_alignment_input sum-type ingestion
<!-- SECTION:FINAL_SUMMARY:END -->
