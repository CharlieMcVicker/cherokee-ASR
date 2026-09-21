---
id: TASK-359
title: >-
  Add configurable contextual sibilant pre-aspiration to syllabary-to-phonetics
  conversion pipeline
status: Done
assignee:
  - '@agent'
created_date: '2026-09-21 19:57'
updated_date: '2026-09-21 20:06'
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
- [x] #1 Implement contextual pre-aspiration rule suppressing h before s word-initially (^s) and in affricates (ts/tsh) during syllabary-to-TTH conversion
- [x] #2 Add configurable flag to allow switching between contextual pre-aspiration and unconditional hs conversion
- [x] #3 Add unit tests covering both contextual and unconditional conversion modes across word-initial, medial, and affricate positions
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update syllabary_map.py to support contextual_preaspiration flag in syllabary_to_phonetics (defaulting to True) and provide both base transliteration mapping and contextual pre-aspiration regex.
2. Update convert_orthography in orthography.py and normalize_syllabary_for_alignment in normalizers.py to accept contextual_preaspiration flag.
3. Update codeswitched_preparer.py (prepare_code_switched_token, create_groundtruth_for_code_switched_syllabary) to propagate contextual_preaspiration flag.
4. Update ingestion.py (load_syllabary_transcript, load_interview_transcript) and pipeline.py (align_syllabary_ctc) / models.py (CTCAlignerConfig) to support contextual_preaspiration.
5. Add comprehensive unit tests in transcription/utils/tests/ and transcription/alignment/tests/ verifying both contextual and unconditional conversion modes across word-initial, medial, and affricate positions.
6. Verify entire test suite with pytest and run pyright transcription.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added CHEROKEE_SYLLABARY_BASE_MAP and updated syllabary_to_phonetics to support contextual_preaspiration (default True) alongside unconditional hs mode. Propagated contextual_preaspiration flag through convert_orthography, normalize_syllabary_for_alignment, normalize_phonetics_for_alignment, prepare_code_switched_token, create_groundtruth_for_code_switched_syllabary, load_syllabary_transcript, load_interview_transcript, CTCAlignerConfig, and align_syllabary_ctc. Verified with 390 passing unit tests and 0 Pyright errors.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented configurable contextual sibilant pre-aspiration across the entire Cherokee syllabary and code-switched alignment pipeline. Contextual mode suppresses leading 'h' before word-initial 's' (^s) and affricates (ts/tsh) while applying pre-aspiration postvocalically/medially to match ASR acoustic training regimes. Unconditional mode is retained via the configurable flag for older models. Validated via comprehensive unit tests in test_syllabary_map.py, test_orthography.py, and test_codeswitched_preparer.py with 100% test pass rate (390 passed) and 0 pyright errors.
<!-- SECTION:FINAL_SUMMARY:END -->
