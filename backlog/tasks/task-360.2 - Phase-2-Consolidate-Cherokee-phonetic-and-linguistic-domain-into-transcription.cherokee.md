---
id: TASK-360.2
title: 'Phase 2A: Cherokee orthography conversions, syllabary tables & tone normalizer'
status: Done
assignee:
  - '@phase-2a-implementor'
created_date: '2026-09-21 20:24'
updated_date: '2026-09-22 15:19'
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
- [x] #1 Orthography enum, convert_orthography, and string cleaners reside in transcription.cherokee.orthography.orthography
- [x] #2 Unicode syllabary lookup dictionaries reside in transcription.cherokee.orthography.syllabary_map
- [x] #3 Tone stripping and colon normalization reside in transcription.cherokee.orthography.tones
- [x] #4 CherokeeASRModel factory loading default Cherokee repositories resides in transcription.cherokee.models
- [x] #5 Unit tests in test_orthography.py and test_syllabary_map.py pass with 0 pyright errors
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create transcription/cherokee/orthography/syllabary_map.py with Cherokee Unicode syllabary lookup dictionaries.
2. Create transcription/cherokee/orthography/tones.py with tone stripping, colon normalization, and consonant respelling.
3. Create transcription/cherokee/orthography/orthography.py with Orthography enum, convert_orthography, and string cleaners.
4. Create transcription/cherokee/orthography/__init__.py exporting Orthography, convert_orthography, syllabary maps, and tone normalizers.
5. Create transcription/cherokee/models/loader.py and transcription/cherokee/models/__init__.py with CherokeeASRModel factory loading default Cherokee repositories.
6. Create transcription/cherokee/__init__.py.
7. Re-export in transcription/utils/orthography.py, transcription/utils/syllabary_map.py, and transcription/models/asr_model.py for backwards compatibility.
8. Add tests in transcription/cherokee/tests/.
9. Verify with pytest, pyright, and format with black.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Consolidated Cherokee orthography, Unicode syllabary tables, and tone/consonant normalizers into transcription.cherokee.orthography, and established CherokeeASRModel default loader factories in transcription.cherokee.models. Preserved full backwards compatibility with clean facades in transcription/utils/orthography.py, transcription/utils/syllabary_map.py, transcription/utils/tone_normalization.py, and transcription/models/asr_model.py. Verified with unit test suites in transcription/cherokee/tests/ (test_cherokee_orthography.py, test_cherokee_syllabary_map.py, test_cherokee_tones.py, test_cherokee_models.py) passing alongside all 433 tests in the full test suite with 0 pyright errors and black formatting verified.
<!-- SECTION:FINAL_SUMMARY:END -->
