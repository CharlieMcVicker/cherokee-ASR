---
id: TASK-327
title: Revert to phonetic citations with tl-to-dl mapping for alignment
status: Done
assignee:
  - '@antigravity'
created_date: '2026-09-14 14:56'
updated_date: '2026-09-14 14:59'
labels: []
dependencies: []
ordinal: 343000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Revert base alignment input from syllabary back to phonetic citations, mapping 'tl' -> 'dl' prior to consonant respelling so citation 'tl' defaults to unaspirated 'tl', preventing false syllabary digraph anomalies and enabling proper intrusion modeling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Update normalize_phonetics_for_alignment to map tl -> dl before consonant respelling and handle vowel hiatus
- [x] #2 Revert load_bible_chunks and pipeline to prioritize phonetic citation text over syllabary
- [x] #3 Set CTCSegmentationAligner default normalizer to normalize_phonetics_for_alignment and prioritize phonetic text in align_verse_slice
- [x] #4 Verify with unit/integration tests and run Mark Chapter 1 realignment to ensure enhha'i and spurious digraph anomalies are eliminated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update normalize_phonetics_for_alignment in transcription/alignment/normalizers.py to map 'tl' -> 'dl' prior to respell_consonants and insert hiatus glottal stops between adjacent vowels.
2. Update load_bible_chunks in transcription/alignment/ingestion.py to prioritize 'phonetic' / 'raw_phonetic' over 'cherokee' / 'syllabary'.
3. Update default normalizer in CTCSegmentationAligner and pipeline.py to normalize_phonetics_for_alignment.
4. Update align_verse_slice in CTCSegmentationAligner to prioritize phonetic_text.
5. Update unit tests in test_ingestion.py, test_normalizers.py, and test_ctc_aligner.py.
6. Run full test suite and test realigning Mark Chapter 1.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reverted base alignment text to phonetic citations, mapped 'tl' -> 'dl' before consonant respelling to preserve unaspirated defaults, and added hiatus glottal stop insertion. Verified with pytest (265 passed), pyright (0 errors), Mark Chapter 1 realignment, and 100-verse benchmark (anomalies reduced from 18 to 11, 'enhha'i' eliminated).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reverted base alignment text from syllabary back to phonetic citations. Updated normalize_phonetics_for_alignment to map 'tl' -> 'dl' prior to respell_consonants so lateral stops default to unaspirated 'tl' (allowing proper intrusion modeling) and insert hiatus glottal stops between adjacent vowels. Updated load_bible_chunks, pipeline, and CTCSegmentationAligner to prioritize phonetic citations. Verified end-to-end: 265 unit tests pass, Mark 1 realigned cleanly with valid phonotactic sequence 'enha'i', and 100-verse benchmark anomaly rate dropped from 18 to 11 verses.
<!-- SECTION:FINAL_SUMMARY:END -->
