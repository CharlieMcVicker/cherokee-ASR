---
id: TASK-331
title: Support lateral deaffrication (tl -> lh) in CTC trellis phonotactics
status: Done
assignee:
  - '@agent'
created_date: '2026-09-14 20:03'
updated_date: '2026-09-16 17:06'
labels: []
dependencies: []
ordinal: 347000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In Cherokee speech, written lateral affricate tl (e.g. from citation syllabary ᎣᏝ 'otla') frequently undergoes lateral deaffrication to lateral fricative/approximant lh ('olha'). To prevent the CTC segmentation trellis from penalizing spoken 'olha' or dropping alignment confidence, prepare_cherokee_text in transcription/alignment/phonotactics.py should license optional deletion/skip of the leading stop 't' in 'tl' clusters via the syncope/optional token mask.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 prepare_cherokee_text marks leading stop 't' in 'tl' clusters as eligible for optional deletion in syncope_mask
- [x] #2 Unit tests in test_phonotactics.py verify tl deaffrication masking behavior
- [x] #3 CTC aligner successfully aligns deaffricated speech (e.g. ᎣᏝ -> spoken olha) with high confidence
- [x] #4 All pytest tests and pyright static typing pass with zero regressions
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update get_syncope_mask in transcription/alignment/phonotactics.py to mark the initial 't' character in 'tl' and 'tlh' tokens as eligible for syncope.
2. Update CTCAlignerConfig.syncope_tokens in transcription/alignment/models.py to include 't'.
3. Add unit test in transcription/alignment/tests/test_phonotactics.py verifying tl deaffrication syncope masking.
4. Verify all unit tests pass with pytest and pyright.
5. Rescore the dataset with scripts/rescore_syllabary_dataset.py and verify performance.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented lateral deaffrication syncope masking in get_syncope_mask (marking leading 't' stop in 'tl' and 'tlh' clusters as eligible for deletion) and added 't' to CTCAlignerConfig.syncope_tokens default. Added unit test test_lateral_deaffrication_syncope_mask in test_phonotactics.py and verified all 275 pytest tests and pyright static type checks pass with zero errors.
<!-- SECTION:FINAL_SUMMARY:END -->
