---
id: TASK-324
title: >-
  License vowel coda laryngeals and consonant post-aspiration in phonotactics
  intrusion mask
status: Done
assignee:
  - '@agent'
created_date: '2026-09-14 14:39'
updated_date: '2026-09-14 14:40'
labels: []
dependencies: []
ordinal: 340000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update get_intrusion_site_mask in transcription/alignment/phonotactics.py to license both vowel coda laryngeals (pre-consonantal / post-vocalic) and consonant post-aspiration into following vowels (e.g. l -> h -> i for ukvwalhi, k -> h -> V, t -> h -> V, tl -> h -> V). This ensures the CTC trellis graph can accurately align voiceless sonorants and aspirated stops without requiring pre-corrupted citation texts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 get_intrusion_site_mask licenses intrusion on vowels following plain stops and sonorants (licensing post-aspiration)
- [x] #2 get_intrusion_site_mask preserves Cherokee phonotactic constraints (*HH, *ChR)
- [x] #3 All unit tests in test_phonotactics.py and test_ctc_aligner.py pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update get_intrusion_site_mask in transcription/alignment/phonotactics.py:
   a. Coda laryngeals (pre-consonantal / post-vocalic): License intrusion before consonants and sibilants following vowels.
   b. Consonant post-aspiration: License intrusion on vowels following plain stops (k, t, tl, kw, etc.) and plain sonorants (l, n, w, y) allowing C -> h -> V transitions in the CTC trellis.
   c. Enforce constraints (*HH, *ChR).
2. Update transcription/alignment/tests/test_phonotactics.py with test cases covering post-aspiration (e.g. ukvwali -> ukvwalhi).
3. Re-run benchmark scripts and pytest to verify.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated get_intrusion_site_mask in transcription/alignment/phonotactics.py to license both pre-consonantal coda laryngeals (V -> [h|'] -> C) and post-consonantal aspiration on vowels (C -> [h] -> V), enabling voiceless sonorants like ukvwalhi and aspirated stops to align seamlessly from base syllabary. Added unit tests in test_phonotactics.py, regenerated benchmark comparison, and verified all 262 tests pass.
<!-- SECTION:FINAL_SUMMARY:END -->
