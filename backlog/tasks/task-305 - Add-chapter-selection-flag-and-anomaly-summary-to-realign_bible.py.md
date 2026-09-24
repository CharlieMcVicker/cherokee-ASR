---
id: TASK-305
title: Add chapter selection flag and anomaly summary to realign_bible.py
status: Done
assignee:
  - '@agent'
created_date: '2026-09-12 19:50'
updated_date: '2026-09-12 19:51'
labels: []
dependencies: []
ordinal: 321000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add --chapter CLI argument to scripts/realign_bible.py to enable processing and testing individual chapters, and print flagged anomaly summary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Support optional --chapter <int> argument in realign_bible.py and realign_book
- [x] #2 Display flagged words anomaly summary in CLI output
- [x] #3 Add test coverage for single chapter execution
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added --chapter/-c CLI argument to realign_bible.py, enabled single-chapter realignment with intelligent record merging, added anomaly word tracking to summary reporting, and added unit test coverage in test_pipeline.py.
<!-- SECTION:FINAL_SUMMARY:END -->
