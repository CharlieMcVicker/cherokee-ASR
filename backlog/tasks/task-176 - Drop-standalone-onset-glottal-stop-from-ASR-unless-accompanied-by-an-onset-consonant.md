---
id: TASK-176
title: >-
  Drop standalone onset glottal stop from ASR unless accompanied by an onset
  consonant
status: Done
assignee:
  - '@agent'
created_date: '2026-07-25 20:50'
updated_date: '2026-07-25 20:50'
labels: []
dependencies: []
ordinal: 172000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update glottal stop handling in enrich_syllabary.py so that an onset glottal stop from ASR (e.g. ASR ''a' for base 'ya') is dropped rather than injected as ''ya' unless the emitted slice contains another onset consonant (e.g. ''ky', ''th', ''hy').
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Only preserve ASR onset glottal stop if it accompanies another onset consonant in the emitted slice
- [x] #2 Drop onset glottal stop when ASR emitted is vowel-initial or glottal-only (e.g. ya + ASR ''a' -> 'ya')
- [x] #3 Add unit test for onset glottal stop filtering
- [x] #4 All unit tests pass cleanly
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. In enrich_syllabary.py _enrich_single_syllable, update section C (glottal stop handling).\n2. If emitted.startswith('\''), check if emitted slice contains an onset consonant preceding any vowel (e.g. '\'ky', '\'th', '\'h').\n3. If emitted starts with '\'' without another onset consonant (e.g. '\'a', '\'o'), treat the onset glottal stop as a mistranscription of the base consonant (e.g. 'y') and do not prepand '\''.\n4. Preserve trailing glottal stops or onset glottal stops that accompany explicit onset consonants.\n5. Add unit test in test_enrich_syllabary.py and verify all tests pass.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated glottal stop handling in enrich_syllabary.py section C so that standalone ASR onset glottals (e.g. ASR ''a' for base 'ya') are treated as mistranscriptions of the onset consonant and dropped to output 'ya', while onset glottals accompanying explicit onset consonants (e.g. ''ky', ''th') are preserved. Added unit test in test_enrich_syllabary.py and verified all 46 tests pass.
<!-- SECTION:FINAL_SUMMARY:END -->
