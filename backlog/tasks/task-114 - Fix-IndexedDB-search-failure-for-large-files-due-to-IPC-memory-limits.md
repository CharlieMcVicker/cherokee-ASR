---
id: TASK-114
title: Fix IndexedDB search failure for large files due to IPC/memory limits
status: Done
assignee: []
created_date: '2026-07-09 20:35'
updated_date: '2026-07-09 20:35'
labels: []
dependencies: []
ordinal: 110000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Large files (e.g. 52,000 segments) cause IndexedDB search to crash because getAllFromIndex loads all words into memory at once, exceeding IPC size limit. Fix this by using a cursor instead of getAllFromIndex.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Modify searchAsync to use IndexedDB cursors to iterate over file_words instead of using getAllFromIndex
- [x] #2 Verify search functions correctly
- [x] #3 Mark task as Done and write final summary
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Refactored IndexedDB search logic in frontend/src/App.jsx to iterate over the 'by-file' index using a cursor (IDBKeyRange.only) instead of reading all word records for a file into memory at once with 'getAllFromIndex'. This prevents DOMException crash on large segment files (e.g. 52,000 segments) caused by exceeding Chrome/Firefox IPC size/memory boundaries.
<!-- SECTION:FINAL_SUMMARY:END -->
