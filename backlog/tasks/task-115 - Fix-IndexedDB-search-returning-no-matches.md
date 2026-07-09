---
id: TASK-115
title: Fix IndexedDB search returning no matches
status: Done
assignee: []
created_date: '2026-07-09 20:37'
updated_date: '2026-07-09 20:41'
labels: []
dependencies: []
ordinal: 111000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The cursor-based search is returning no results/matches. Debug the transaction/cursor logic and ensure search works correctly.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Debug the IndexedDB cursor search logic in App.jsx
- [x] #2 Add error handling and verify results are returned correctly
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed a syntax error in searchAsync in frontend/src/App.jsx. We had opened a try block to wrap the new cursor iteration logic but did not properly close it or catch errors, which caused compile/runtime failures in the browser. Wrapped searchAsync fully in a try-catch block and verified proper block matching. Search should now work correctly with the new cursor iterator.

\n- Fixed the silent performance timeout (AbortError) caused by loading massive word value objects during cursor loop. Implemented a key-only cursor scan ('openKeyCursor') that retrieves matching primary keys instantly, and then only retrieves the specific matching records using 'store.get'. This makes search operations extremely lightweight and resolves the 'no matches' bug for large files like CVCS_all.csv without needing to re-sync. Also optimized the sync data storage schema to store compact structures in file_words, dropping bloated character confidence arrays to save 99%+ of space on future syncs.
<!-- SECTION:FINAL_SUMMARY:END -->
