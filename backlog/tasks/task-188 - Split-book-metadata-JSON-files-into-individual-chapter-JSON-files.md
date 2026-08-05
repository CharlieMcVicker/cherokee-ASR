---
id: TASK-188
title: Split book metadata JSON files into individual chapter JSON files
status: Done
assignee:
  - '@agent'
created_date: '2026-08-05 12:29'
updated_date: '2026-08-05 12:29'
labels: []
dependencies: []
ordinal: 184000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Split the book metadata JSON files in cherokee_new_testament/book_transcripts/ into chapter-level JSON files named like matthew_04.json, containining only verses from that specific chapter.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Metadata JSONs are split into chapter-level files named like <book_lowercase>_<chapter_2digits>.json
- [x] #2 Original combined book metadata files are either handled cleanly or chapter files are placed in cherokee_new_testament/book_transcripts/
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Split all combined book metadata files into individual per-chapter JSON files (260 chapter JSON files total across all 27 books) located at cherokee_new_testament/book_transcripts/<book_lowercase>_<chapter_2digits>.json (e.g. matthew_04.json).
<!-- SECTION:FINAL_SUMMARY:END -->
