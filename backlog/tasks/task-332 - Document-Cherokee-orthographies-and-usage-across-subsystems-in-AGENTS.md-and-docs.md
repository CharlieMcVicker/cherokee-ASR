---
id: TASK-332
title: >-
  Document Cherokee orthographies and usage across subsystems in AGENTS.md and
  docs
status: To Do
assignee: []
created_date: '2026-09-14 20:27'
labels: []
dependencies: []
ordinal: 348000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clearly document Cherokee orthographic representations (Unicode Syllabary, base DG Latin transliteration, and canonical TTH phonetic training/acoustic format) in AGENTS.md and docs/alignment.md. Clarify that the acoustic model output vocabulary uses TTH (contains tl, lh, th, kh, wh, yh, nh, but no dl or g), that tones/vowel length are orthogonal to consonant systems, and specify which modules/subsystems consume and produce each orthography.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 AGENTS.md updated with a dedicated Orthography & Phonetics section explaining SYLLABARY, DG, and TTH usage across pipelines and models
- [ ] #2 Explicitly document that Wav2Vec2/ASR acoustic emissions use TTH phonetics (tl, lh, th, kh, etc.) and contain no dl or g
- [ ] #3 docs/alignment.md updated to explain Orthography enum conversions and representation expectations
- [ ] #4 Documentation formatting and cross-references verified
<!-- AC:END -->
