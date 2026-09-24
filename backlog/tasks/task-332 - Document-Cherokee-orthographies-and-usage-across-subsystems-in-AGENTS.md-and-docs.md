---
id: TASK-332
title: >-
  Document Cherokee orthographies and usage across subsystems in AGENTS.md and
  docs
status: Done
assignee:
  - '@antigravity'
created_date: '2026-09-14 20:27'
updated_date: '2026-09-15 13:24'
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
- [x] #1 AGENTS.md updated with a dedicated Orthography & Phonetics section explaining SYLLABARY, DG, and TTH usage across pipelines and models
- [x] #2 Explicitly document that Wav2Vec2/ASR acoustic emissions use TTH phonetics (tl, lh, th, kh, etc.) and contain no dl or g
- [x] #3 docs/alignment.md updated to explain Orthography enum conversions and representation expectations
- [x] #4 Documentation formatting and cross-references verified
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update AGENTS.md with a comprehensive 'Cherokee Orthographies & Phonetics Reference' section detailing SYLLABARY, DG (Bible/dictionary), and canonical TTH (Acoustic/Training), emphasizing the strict absence of d, g, ch, j, q in TTH and the lateral mapping dl->tl, tl->tlh, hl->lh.
2. Update docs/alignment.md section on Text Normalization and Ground-Truth Ingestion to document the Orthography enum, conversion pipelines, and representation expectations.
3. Validate pytest test suite and pyright typechecks.
4. Verify cross-references and documentation integrity.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Documented the three primary Cherokee orthographies (SYLLABARY, DG, TTH), strict absence of d, g, ch, j, q in canonical TTH, lateral mapping dl->tl, tl->tlh, hl->lh, and Bible import ingestion with hyphens across AGENTS.md and docs/alignment.md. Validated with pytest (273 passing) and pyright (0 errors).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated AGENTS.md and docs/alignment.md with comprehensive Cherokee Orthographies and Phonetic Conventions documentation. Clarified that Wav2Vec2/CherokeeASRModel emissions strictly use canonical TTH (containing tl, tlh, lh, th, kh, wh, yh, nh, s, hs, ts, tsh, and no d, g, ch, j, q), documented that Bible imports on disk contain Syllabary and hyphenated DG transliteration converted via normalize_phonetics_for_alignment, and verified all 273 pytest tests and pyright static type checks.
<!-- SECTION:FINAL_SUMMARY:END -->
