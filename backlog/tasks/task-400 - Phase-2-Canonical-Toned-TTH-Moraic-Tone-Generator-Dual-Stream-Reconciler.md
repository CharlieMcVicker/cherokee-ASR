---
id: TASK-400
title: 'Phase 2: Canonical Toned TTH, Moraic Tone Generator & Dual-Stream Reconciler'
status: Done
assignee:
  - '@self'
created_date: '2026-11-07 21:25'
updated_date: '2026-11-07 21:42'
labels: []
dependencies:
  - TASK-399
modified_files:
  - digohwelisgi/cherokee/orthography/tones.py
  - digohwelisgi/cherokee/orthography/__init__.py
  - digohwelisgi/cherokee/orthography/tests/test_moraic_generator.py
ordinal: 433300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Convert raw Feeling phonetics to canonical numeric-tone TTH ground truth, factor into (cv_text, tone_tokens) moraic streams, and reconstruct canonical toned TTH via fast slot matching and Needleman-Wunsch DP alignment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 to_canonical_toned_tth converts raw Feeling phonetics into canonical numeric-tone TTH using syllabary word-boundary alignment for asterisk cleanup
- [x] #2 to_moraic_tone_tokens factors canonical toned TTH into clean single-vowel cv_text and moraic tone_tokens (C, 2_1, 3_1, 1_2..4_2, |)
- [x] #3 reconcile_dual_stream supports both fast and dp modes and achieves 100% exact round-trip reconstruction across all valid Feeling sentences in sentence_audio.csv
- [x] #4 moraic_tokens_to_pitch_contour maps moraic tone tokens to pitch contour coordinates for Praat/visual inspection
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Implement clean_asterisks_with_syllabary and to_canonical_toned_tth in tones.py
2. Implement to_moraic_tone_tokens(raw_or_canonical) returning (cv_text, tone_tokens)
3. Implement reconcile_dual_stream(cv_text, tone_tokens, mode="dp"|"fast") and moraic_tokens_to_pitch_contour
4. Add unit tests in test_moraic_generator.py and run offline round-trip verification across all valid sentences in sentence_audio.csv (Integration Gate 2)
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Phase 2 Canonical Toned TTH, Moraic Tone Generator & Dual-Stream Reconciler.
- Added clean_asterisks_with_syllabary and to_canonical_toned_tth.
- Implemented to_moraic_tone_tokens, reconcile_dual_stream (dp and fast modes), and moraic_tokens_to_pitch_contour.
- Verified 100% round-trip accuracy across all 1,860 valid Feeling sentences in sentence_audio.csv.
<!-- SECTION:FINAL_SUMMARY:END -->
