---
id: TASK-135
title: Create transcription spelling verification script
status: Done
assignee:
  - '@agent'
created_date: '2026-07-10 15:28'
updated_date: '2026-07-10 15:29'
labels: []
dependencies: []
ordinal: 131000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write a script to verify the correctness of human transcription in the target spelling system. Flag all characters that should be 'spelled away' (e.g. d/k/c) or anytime s is present without leading h. Test it on Cora Flute.TextGrid.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement script that reads TextGrid files and flags invalid characters (d, k, c) or 's' without leading 'h'
- [x] #2 Test the script on Cora Flute.TextGrid
- [x] #3 Provide reports of any transcription errors found in the file
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented transcription/utils/verify_spelling.py to parse TextGrid files and check the 'human' tier for forbidden characters (d, k, c) and 's' characters lacking a leading 'h'. Ran the verification on data/processed/praat/Cora Flute.TextGrid and found 93 errors across 30 intervals.
<!-- SECTION:FINAL_SUMMARY:END -->
