---
id: TASK-360.2
title: 'Phase 2A: Cherokee orthography conversions, syllabary tables & tone normalizer'
status: To Do
assignee: []
created_date: '2026-09-21 20:24'
updated_date: '2026-09-21 20:32'
labels: []
dependencies: []
parent_task_id: TASK-360
ordinal: 388100
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Consolidate Orthography enum, canonical Unicode syllabary tables, DG/TTH converters, and tone/colon normalization into transcription.cherokee.orthography, and define Cherokee HuggingFace model configurations in transcription.cherokee.models.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Orthography enum, convert_orthography, and string cleaners reside in transcription.cherokee.orthography.orthography
- [ ] #2 Unicode syllabary lookup dictionaries reside in transcription.cherokee.orthography.syllabary_map
- [ ] #3 Tone stripping and colon normalization reside in transcription.cherokee.orthography.tones
- [ ] #4 CherokeeASRModel factory loading default Cherokee repositories resides in transcription.cherokee.models
- [ ] #5 Unit tests in test_orthography.py and test_syllabary_map.py pass with 0 pyright errors
<!-- AC:END -->
