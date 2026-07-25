---
id: TASK-168
title: Implement Phonetic Rule Merger Engine in enrich_syllabary.py
status: Done
assignee:
  - '@agent'
created_date: '2026-07-25 19:55'
updated_date: '2026-07-25 20:03'
labels: []
dependencies: []
ordinal: 164000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build core reconciliation engine that enriches immutable syllabary skeleton with ASR acoustic features. Focus strictly on 2 feature categories: (1) Vowel deletion/syncopation detection and (2) All aspiration and glottal activity (pre-aspiration h, glottal stops ', and laryngeal toggles such as t -> th, k -> kh).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement reconcile_phonetics(syllabary_text, base_transliteration, emitted_text, aligned_pairs) -> str
- [x] #2 Enforce Rule 1: Vowel deletion/syncopation based on aligned ASR window
- [x] #3 Enforce Rule 2: Intrusive glottal injection (pre-aspiration h and glottal stop ')
- [x] #4 Keep syllabary skeleton as immutable anchor
- [x] #5 Add unit tests verifying syncopation and glottal injection rules
- [x] #6 Implement reconcile_phonetics(syllabary_text, base_transliteration, emitted_text, aligned_pairs) -> str
- [x] #7 Enforce Rule 1: Vowel deletion/syncopation based on aligned ASR window
- [x] #8 Enforce Rule 2: All aspiration and glottal activity transfer (pre-aspiration h, glottal stop ', and t->th / k->kh laryngeal toggles)
- [x] #9 Keep syllabary skeleton as immutable structural anchor
- [x] #10 Add unit tests verifying syncopation, aspiration/laryngeal toggles, and glottal injection
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect transcription/syllabary_enrichment/ alignment & batch structures.\n2. Create enrich_syllabary.py in transcription/syllabary_enrichment.\n3. Implement reconcile_phonetics(syllabary_text, base_transliteration, emitted_text, aligned_pairs) -> str.\n4. Implement Rule 1: Vowel deletion/syncopation based on aligned ASR window.\n5. Implement Rule 2: All aspiration and glottal activity transfer (pre-aspiration h, glottal stop ', and t->th / k->kh laryngeal toggles).\n6. Keep syllabary skeleton as immutable anchor.\n7. Add comprehensive unit tests in test_enrich_syllabary.py.\n8. Mark ACs checked and set TASK-168 to Done.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented core Phonetic Rule Merger Engine in transcription/syllabary_enrichment/enrich_syllabary.py with reconcile_phonetics function. Enforces Rule 1 (vowel deletion/syncopation) and Rule 2 (pre-aspiration h, glottal stop ', and laryngeal toggles t->th, k->kh) while treating Cherokee Syllabary as the immutable structural anchor. Exported reconcile_phonetics in __init__.py and added comprehensive unit test suite in test_enrich_syllabary.py.
<!-- SECTION:FINAL_SUMMARY:END -->
