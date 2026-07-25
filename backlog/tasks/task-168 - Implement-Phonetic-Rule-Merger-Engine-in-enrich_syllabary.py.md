---
id: TASK-168
title: Implement Phonetic Rule Merger Engine in enrich_syllabary.py
status: To Do
assignee: []
created_date: '2026-07-25 19:55'
updated_date: '2026-07-25 19:57'
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
- [ ] #1 Implement reconcile_phonetics(syllabary_text, base_transliteration, emitted_text, aligned_pairs) -> str
- [ ] #2 Enforce Rule 1: Vowel deletion/syncopation based on aligned ASR window
- [ ] #3 Enforce Rule 2: Intrusive glottal injection (pre-aspiration h and glottal stop ')
- [ ] #4 Keep syllabary skeleton as immutable anchor
- [ ] #5 Add unit tests verifying syncopation and glottal injection rules
- [ ] #6 Implement reconcile_phonetics(syllabary_text, base_transliteration, emitted_text, aligned_pairs) -> str
- [ ] #7 Enforce Rule 1: Vowel deletion/syncopation based on aligned ASR window
- [ ] #8 Enforce Rule 2: All aspiration and glottal activity transfer (pre-aspiration h, glottal stop ', and t->th / k->kh laryngeal toggles)
- [ ] #9 Keep syllabary skeleton as immutable structural anchor
- [ ] #10 Add unit tests verifying syncopation, aspiration/laryngeal toggles, and glottal injection
<!-- AC:END -->
