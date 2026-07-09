---
id: TASK-117
title: Create compound index by-file-and-word and optimize search performance
status: Done
assignee: []
created_date: '2026-07-09 20:53'
updated_date: '2026-07-09 20:53'
labels: []
dependencies: []
ordinal: 113000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create compound index on file_path and word in file_words store to enable instant range queries. Update searchAsync to query this index first for high performance, falling back to cursor scan only if no results are found.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Upgrade db to version 3 in App.jsx and add by-file-and-word compound index
- [x] #2 Update searchAsync to query the compound index first
- [x] #3 Verify search functions correctly and instantly on large files
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Upgraded IndexedDB schema to version 3 in App.jsx to add the compound index 'by-file-and-word' on ['file_path', 'word']. Updated searchAsync to perform instant range queries using this index, returning matching objects immediately without iterating the entire vocabulary when a prefix matches. If the compound index isn't found or returns no prefix matches, it seamlessly falls back to the key cursor search. This resolves the AbortError timeout completely, making searches on files like CVCS_all.csv instant.
<!-- SECTION:FINAL_SUMMARY:END -->
