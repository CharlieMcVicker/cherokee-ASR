---
id: TASK-262
title: >-
  Audit codebase to ensure core ASR abstractions remain language-agnostic from
  syllabary logic
status: To Do
assignee: []
created_date: '2026-09-02 15:01'
updated_date: '2026-09-02 15:05'
labels:
  - code-smell
  - architecture
  - models
dependencies: []
priority: low
ordinal: 264000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Code smell / architectural concern identified in PR #3 review: ASR models output phonetic transcripts; converting phonetics to Cherokee syllabary is a downstream language-specific enrichment step.

Target areas:
1. `transcription/inference/infer.py`: Audit `greedy_inference` and other legacy helpers for embedded syllabary conversions. Decouple or clarify syllabary conversion boundaries.
2. `docs/models_and_inference.md` and `docs/alignment.md`: Update documentation to accurately describe `ASRResult` and the language-agnostic acoustic CTC inference pipeline versus downstream Cherokee Syllabary enrichment.
3. Verify that `transcription/syllabary_enrichment/` and `transcription/utils/syllabary_map.py` remain the dedicated modules for Cherokee-specific syllabary transformation.

Deliverables:
- Core inference abstractions remain language-agnostic (acoustic signal -> phonetic/character tokens).
- Syllabary conversion is only invoked explicitly by downstream callers.
- Documentation accurately reflects the architectural boundaries.
- All unit tests pass and pyright reports 0 errors.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Audit all core model and inference classes for implicit Cherokee-specific syllabary assumptions
- [ ] #2 Ensure syllabary conversion functions are cleanly exported and invoked only by downstream consumers who require Cherokee Syllabary
- [ ] #3 Verify all documentation accurately distinguishes acoustic phonetic ASR decoding from downstream syllabary transliteration/enrichment
- [ ] #4 Verify all tests pass and pyright typechecking succeeds
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Review `transcription/inference/infer.py` for any lingering implicit syllabary dependencies.
2. Update documentation in `docs/models_and_inference.md` and related guides.
3. Run `pytest` and `pyright transcription`.
4. Format and commit.
<!-- SECTION:PLAN:END -->
