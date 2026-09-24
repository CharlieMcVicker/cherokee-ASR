---
id: TASK-314
title: >-
  Tune CTC syncope and intrusive penalties to curb excessive syncope and /h/
  insertion
status: Done
assignee:
  - '@myself'
created_date: '2026-09-12 21:04'
updated_date: '2026-09-12 21:05'
labels: []
dependencies: []
ordinal: 330000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose syncope_penalty and intrusive_penalty (with increased defaults, e.g. syncope_penalty=6.0, intrusive_penalty=2.5) in CTCSegmentationAligner and realign_bible.py CLI, then rerun realignment on Mark Chapter 1 and evaluate the output.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add --syncope-penalty and --intrusive-penalty arguments to realign_bible.py
- [x] #2 Update default penalties or provide tuned parameters to reduce excessive vowel loss and spurious /h/
- [x] #3 Rerun Mark Chapter 1 realignment and verify cleaner emissions
- [x] #4 Run pytest and verify all tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Expose syncope_penalty and intrusive_penalty parameters in get_default_ctc_aligner, realign_book, realign_all, and CLI parser in realign_bible.py.
2. Set tuned defaults (syncope_penalty=6.0, intrusive_penalty=2.5) to prevent easy vowel dropping and breath /h/ insertion.
3. Rerun realignment for Mark Chapter 1 and evaluate the phonetic transcription output.
4. Run full pytest test suite to ensure no regressions.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Exposed syncope_penalty and intrusive_penalty in realign_bible.py with tuned defaults (syncope_penalty=6.0, intrusive_penalty=2.5). Reran realignment on Mark Chapter 1 and verified that excessive vowel dropping and spurious /h/ insertion were resolved, producing natural phonetic transcriptions (e.g. retaining vowels in tsinvhsithahsthi, tsiluhsilmi, akwhathihsthathihsthiyi). All 249 tests passing.
<!-- SECTION:FINAL_SUMMARY:END -->
