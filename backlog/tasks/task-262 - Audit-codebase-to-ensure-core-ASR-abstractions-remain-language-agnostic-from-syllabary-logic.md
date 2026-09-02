---
id: TASK-262
title: >-
  Audit codebase to ensure core ASR abstractions remain language-agnostic from
  syllabary logic
status: To Do
assignee: []
created_date: '2026-09-02 15:01'
labels:
  - code-smell
  - architecture
  - models
dependencies: []
ordinal: 264000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Code smell / architectural concern identified in PR #3 review: ASR models output phonetic transcripts; converting phonetics to Cherokee syllabary is a downstream language-specific enrichment step. Audit the models and inference packages to ensure core ASR abstractions (ASRResult, model decoders, pipelines) remain language-agnostic, keeping syllabary conversion isolated in transcription/syllabary_enrichment or transcription/utils/syllabary_map.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Audit all core model and inference classes for implicit Cherokee-specific syllabary assumptions
- [ ] #2 Ensure syllabary conversion functions are cleanly exported and invoked only by downstream consumers who require Cherokee Syllabary
- [ ] #3 Verify all documentation accurately distinguishes acoustic phonetic ASR decoding from downstream syllabary transliteration/enrichment
- [ ] #4 Verify all tests pass and pyright typechecking succeeds
<!-- AC:END -->
