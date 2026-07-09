---
id: TASK-116
title: Support phrase and substring search on greedy transcription column
status: Done
assignee: []
created_date: '2026-07-09 20:43'
updated_date: '2026-07-09 20:43'
labels: []
dependencies: []
ordinal: 112000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enable multi-word phrase search and literal substring search on the greedy_transcription column using an inverted index matching approach (split query, find matching segment intersections, and fetch/validate segments) to maintain high performance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Modify searchAsync in App.jsx to support multi-word queries
- [x] #2 Implement inverted index intersection and post-filtering for candidate segments
- [x] #3 Verify search returns correct results for phrase searches
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented phrase and substring search matching on greedy_transcription. Splitting multi-word search queries into multiple terms, matching all against key cursor entries in file_words, intersecting candidate segment indices where all terms exist, and then verifying/post-filtering with a literal string contains check on greedy_transcription. This allows users to perform fast multi-word phrase searches on the greedy transcription column without scanning all segments in the database directly.
<!-- SECTION:FINAL_SUMMARY:END -->
