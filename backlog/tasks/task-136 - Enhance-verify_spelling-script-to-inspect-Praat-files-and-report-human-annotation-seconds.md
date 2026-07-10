---
id: TASK-136
title: >-
  Enhance verify_spelling script to inspect Praat files and report human
  annotation seconds
status: Done
assignee:
  - '@agent'
created_date: '2026-07-10 15:39'
updated_date: '2026-07-10 15:39'
labels: []
dependencies: []
ordinal: 132000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update transcription/utils/verify_spelling.py or create inspect_praat.py to also report the total duration of human annotations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement duration calculation for human annotations
- [x] #2 Output both spelling verification results and duration metrics
- [x] #3 Test and run on Cora Flute.TextGrid
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Upgraded verify_spelling.py to inspect_praat.py. The new script reports the total number of annotated seconds and detailed interval statistics (duration min/max/average) alongside spelling checks. Running the script on Cora Flute.TextGrid shows 226.79 seconds of human annotation time.
<!-- SECTION:FINAL_SUMMARY:END -->
