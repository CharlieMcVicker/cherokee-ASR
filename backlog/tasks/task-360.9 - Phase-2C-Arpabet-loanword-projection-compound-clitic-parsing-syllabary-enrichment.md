---
id: TASK-360.9
title: >-
  Phase 2C: Arpabet loanword projection, compound clitic parsing & syllabary
  enrichment
status: Done
assignee:
  - '@phase-2c-implementor'
created_date: '2026-09-21 20:32'
updated_date: '2026-09-22 15:49'
labels: []
dependencies: []
parent_task_id: TASK-360
ordinal: 388300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Consolidate SyntheticTargetProjector, G2P fallback, compound Latin stem + Syllabary clitic discrimination (JayᎢ -> Jay + Ꭲ), and syllable alignment reconciliation into transcription.cherokee.codeswitching and transcription.cherokee.enrichment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SyntheticTargetProjector, CodeSwitchedPreparer, and CodeSwitchedToken reside in transcription.cherokee.codeswitching
- [x] #2 Static dictionary paths resolved relative to data/arpabet_alignment/dictionaries/english_loanwords_tth.json
- [x] #3 SyllableAlignmentEngine, reconcile_phonetics, and reconcile_alignment_words reside in transcription.cherokee.enrichment
- [x] #4 All Arpabet and syllabary enrichment unit tests pass with 0 pyright errors
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create transcription/cherokee/codeswitching/ with projector.py, codeswitched_preparer.py, __init__.py (resolve static dict relative to repo root data/arpabet_alignment/dictionaries/english_loanwords_tth.json, provide get_english_loanwords_tth_dict).
2. Create transcription/cherokee/enrichment/ with syllable_alignment.py (containing SyllableAlignmentEngine, SyllableAlignment, align_character_syllable, align_character_syllable_detailed, reconcile_phonetics, reconcile_alignment_words, reconcile_word_intervals, reconcile_alignment_by_chunk) and __init__.py.
3. Update transcription.cherokee.__init__.py to re-export codeswitching and enrichment public symbols.
4. Maintain backwards compatibility shims in transcription/alignment/arpabet/ and transcription/syllabary_enrichment/ forwarding to new modules.
5. Create comprehensive unit tests in transcription/cherokee/tests/test_cherokee_codeswitching.py and test_cherokee_enrichment.py.
6. Verify existing and new test suites pass with pytest and pyright returns 0 errors.
7. Format with black and finalize Backlog task TASK-360.9.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
### Phase 2C Implementation Summary

#### 1. Code-Switching Architecture ()
- Migrated `SyntheticTargetProjector`, `CodeSwitchedPreparer`, `CodeSwitchedToken`, `TokenType`, `CodeSwitchedLineResult`, `split_compound_clitic`, and `create_groundtruth_for_code_switched_syllabary` into `transcription/cherokee/codeswitching/`.
- Resolved static dictionary and confusion matrix default paths relative to repository root: `data/arpabet_alignment/dictionaries/english_loanwords_tth.json` and `data/arpabet_alignment/matrices/acoustic_confusion_matrix_latest.json`. Added `get_english_loanwords_tth_dict` helper with cached loading.
- Created package root exports in `transcription/cherokee/codeswitching/__init__.py`.

#### 2. Enrichment Architecture (`transcription.cherokee.enrichment`)
- Migrated `SyllableAlignmentEngine`, `SyllableAlignment`, `align_character_syllable`, `align_character_syllable_detailed`, `reconcile_phonetics`, `reconcile_word_intervals`, `reconcile_alignment_words`, and `reconcile_alignment_by_chunk` into `transcription/cherokee/enrichment/`.
- Created package root exports in `transcription/cherokee/enrichment/__init__.py`.

#### 3. Backwards Compatibility Shims & Circular Import Elimination
- Created backwards compatibility forwarding shims in `transcription/alignment/arpabet/projector.py`, `transcription/alignment/arpabet/codeswitched_preparer.py`, `transcription/syllabary_enrichment/alignment_engine.py`, and `transcription/syllabary_enrichment/enrich_syllabary.py`.
- Eliminated circular imports between `transcription.alignment` and `transcription.cherokee` by using lazy `__getattr__` in `transcription.alignment.arpabet.__init__` (guarded for type checking with `TYPE_CHECKING`) and cleanly ordering imports in `transcription.cherokee.__init__`.

#### 4. Verification & Testing
- Added comprehensive unit tests in `transcription/cherokee/tests/test_cherokee_codeswitching.py` and `transcription/cherokee/tests/test_cherokee_enrichment.py`.
- All 463 tests pass across the entire codebase (`pytest`).
- 0 errors, 0 warnings across `pyright transcription`.
- Formatted modified code using `black`.
<!-- SECTION:FINAL_SUMMARY:END -->
