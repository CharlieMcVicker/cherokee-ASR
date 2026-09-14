---
id: TASK-318
title: >-
  Wire phonotactic text preparation and PR #6 parameters into
  CTCSegmentationAligner and pipeline
status: To Do
assignee: []
created_date: '2026-09-14 13:09'
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
- [ ] #1 Integrate prepare_cherokee_text into CTCSegmentationAligner.align and align_verse_slice
- [ ] #2 Forward intrusive_penalties, intrusive_min_logprobs, is_intrusive_site, and is_syncope_token to CtcSegmentationParameters
- [ ] #3 Expose per-token parameters in realign_bible.py CLI
- [ ] #4 Verify all unit tests in test_ctc_aligner.py and test_pipeline.py pass
<!-- AC:END -->
