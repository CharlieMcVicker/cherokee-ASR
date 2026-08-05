---
id: TASK-186
title: Copy New Testament book metadata files to book_transcripts folder
status: Done
assignee:
  - '@agent'
created_date: '2026-08-05 12:22'
updated_date: '2026-08-05 12:23'
labels: []
dependencies: []
ordinal: 182000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Copy all metadata.json files from ../phoenix/training_data/cnt/book_XX/ (where XX spans all New Testament books) into cherokee_new_testament/book_transcripts with descriptive file names indicating their source book.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Metadata files for all NT books are copied into cherokee_new_testament/book_transcripts
- [x] #2 Files are named based on their source book
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Copied all 27 New Testament metadata.json files from ../phoenix/training_data/cnt/book_01..27 to cherokee_new_testament/book_transcripts/ metadata_book_01.json .. metadata_book_27.json.
<!-- SECTION:FINAL_SUMMARY:END -->
