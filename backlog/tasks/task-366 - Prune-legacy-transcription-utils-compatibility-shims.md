---
id: TASK-366
title: Prune legacy transcription/utils compatibility shims
status: Done
assignee:
  - '@subagent'
created_date: '2026-09-23 15:55'
updated_date: '2026-09-23 16:33'
labels: []
dependencies: []
modified_files:
  - transcription/utils/orthography.py
  - transcription/utils/syllabary_map.py
  - transcription/utils/tone_normalization.py
  - transcription/utils/test_syllabary_map.py
  - transcription/cherokee/tests/test_cherokee_syllabary_map.py
  - transcription/training/prepare_conrad_csv.py
  - transcription/training/prepare_csv.py
  - transcription/alignment/tests/test_arpabet_types.py
  - transcription/alignment/tests/test_arpabet_inference.py
  - match_cnt.py
  - scripts/rescore_syllabary_dataset.py
  - scripts/tune_ctc_aligner_config.py
  - timestamping_test_data/make_json.py
  - docs/alignment.md
  - docs/syllabary_enrichment.md
  - docs/desktop_transcriber.md
ordinal: 396300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
transcription/utils/ (orthography.py, syllabary_map.py, tone_normalization.py) forward to transcription.cherokee.orthography. Call sites across training, evaluation, and tests should be updated to import from transcription.cherokee.orthography directly and the forwarding wrappers deleted per Clean Break Protocol.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 transcription/utils/ orthography and syllabary compatibility shims deleted
- [x] #2 Call sites in training/, evaluation/, and utils updated to transcription.cherokee.orthography
- [x] #3 All tests pass and pyright reports 0 errors
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Search for all references to transcription.utils across codebase.\n2. Update import call sites across training/, transcription/evaluation/, tests/, etc. to transcription.cherokee.orthography.\n3. Delete forwarding shims in transcription/utils/ (orthography.py, syllabary_map.py, tone_normalization.py) and clean up leftover files/tests if appropriate.\n4. Run pytest and pyright transcription to verify zero regressions.\n5. Commit changes with TASK-366 prefix, update task tracking, check ACs, and add final summary.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Pruned legacy compatibility shims from transcription/utils (orthography.py, syllabary_map.py, tone_normalization.py, test_syllabary_map.py). Consolidated test coverage into transcription.cherokee.tests.test_cherokee_syllabary_map, updated all callers and documentation across training/, alignment tests, and scripts to import directly from transcription.cherokee.orthography, and verified all 393 tests pass with 0 pyright errors.
<!-- SECTION:FINAL_SUMMARY:END -->
