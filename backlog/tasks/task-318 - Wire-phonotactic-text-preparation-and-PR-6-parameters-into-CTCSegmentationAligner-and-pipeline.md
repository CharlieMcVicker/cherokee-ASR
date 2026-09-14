---
id: TASK-318
title: >-
  Wire phonotactic text preparation and PR #6 parameters into
  CTCSegmentationAligner and pipeline
status: Done
assignee:
  - '@agent'
created_date: '2026-09-14 13:09'
updated_date: '2026-09-14 13:40'
labels: []
dependencies: []
ordinal: 334000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Upgrade CTCSegmentationAligner and pipeline.py to use prepare_cherokee_text with is_intrusive_site and is_syncope_token masks, and pass per-token intrusive_penalties and intrusive_min_logprobs to CtcSegmentationParameters. Update realign_bible.py CLI.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Integrate prepare_cherokee_text into CTCSegmentationAligner.align and align_verse_slice
- [x] #2 Forward intrusive_penalties, intrusive_min_logprobs, is_intrusive_site, and is_syncope_token to CtcSegmentationParameters
- [x] #3 Expose per-token parameters in realign_bible.py CLI
- [x] #4 Verify all unit tests in test_ctc_aligner.py and test_pipeline.py pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update CTCSegmentationAligner.__init__ to support intrusive_penalties (Dict[str, float] or float), intrusive_min_logprobs (Dict[str, float] or float), intrusive_max_stride (int), and enforce_phonotactics (bool).
2. Ensure prepare_cherokee_text is invoked in CTCSegmentationAligner.align and align_verse_slice with enforce_phonotactics and char_list, setting is_syncope_token and is_intrusive_site on CtcSegmentationParameters.
3. Pass intrusive_penalties, intrusive_min_logprobs, and intrusive_max_stride to CtcSegmentationParameters.
4. Expose intrusive_penalties, intrusive_min_logprobs, and phonotactics options in pipeline.py align_chapter and scripts/realign_bible.py CLI.
5. Add unit tests in test_ctc_aligner.py and test_pipeline.py to verify parameter forwarding, phonotactic masking, and per-token penalty behavior.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Wired phonotactic text preparation and PR #6 parameters (intrusive_penalties, intrusive_min_logprobs, intrusive_max_stride, enforce_phonotactics) into CTCSegmentationAligner (align and align_verse_slice), new_testament pipeline.py (align_chapter), and scripts/realign_bible.py. Verified with unit and pipeline tests in test_ctc_aligner.py and test_pipeline.py.
<!-- SECTION:FINAL_SUMMARY:END -->
