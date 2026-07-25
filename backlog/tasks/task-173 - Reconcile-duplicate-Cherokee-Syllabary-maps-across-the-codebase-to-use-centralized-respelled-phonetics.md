---
id: TASK-173
title: >-
  Reconcile duplicate Cherokee Syllabary maps across the codebase to use
  centralized respelled phonetics
status: Done
assignee:
  - '@subagent'
created_date: '2026-07-25 20:42'
updated_date: '2026-07-25 20:44'
labels: []
dependencies: []
ordinal: 169000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
There are duplicate syllabary mappings across the codebase (e.g. alignment_engine.py CHEROKEE_SYLLABARY_MAP, prepare_validation_sheet.py cherokee_to_bad_phonetics) with inconsistencies such as Ꮏ mapping to hna instead of nha under respell_consonants. Consolidate into a single authoritative syllabary map module with respell_consonants applied.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Consolidate syllabary transliteration maps into a single centralized dictionary/module in transcription/utils/ or transcription/syllabary_enrichment/
- [x] #2 Ensure respell_consonants is applied consistently (including Ꮏ -> nha)
- [x] #3 Update alignment_engine, prepare_validation_sheet, and all callers to import the centralized syllabary map
- [x] #4 All unit tests pass cleanly
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Locate all syllabary-to-phonetic dictionaries in transcription/\n2. Centralize Cherokee syllabary mapping into a unified module (e.g. transcription/utils/syllabary.py or transcription/syllabary_enrichment/alignment_engine.py)\n3. Ensure respell_consonants is consistently applied to base transliteration (including Ꮏ -> nha)\n4. Refactor alignment_engine.py, prepare_validation_sheet.py, and callers to import the single authoritative map\n5. Run all unit tests with conda run -n cherokee-asr python -m unittest discover transcription/
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Centralized Cherokee Syllabary mapping into transcription/utils/syllabary_map.py, ensuring Ꮏ maps to 'nha' under respell_consonants rules. Refactored alignment_engine.py, prepare_validation_sheet.py, enrich_syllabary.py, inspect_pipeline.py, and make_json.py to import and use the centralized mapping. Added unit tests in test_syllabary_map.py and confirmed all 43 unit tests pass cleanly.
<!-- SECTION:FINAL_SUMMARY:END -->
