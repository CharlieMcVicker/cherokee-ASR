---
id: TASK-160
title: >-
  Move syllabary_word to alignment_manifest.json and remove Syllabary tier from
  Praat export
status: Done
assignee:
  - '@agent'
created_date: '2026-07-23 16:36'
updated_date: '2026-07-23 16:36'
labels: []
dependencies: []
ordinal: 156000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove Cherokee Syllabary tier from Praat TextGrid to prevent non-unicode display issues in Praat, and add syllabary_word to word intervals in alignment_manifest.json.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 alignment_manifest.json includes syllabary_word for each word interval when available
- [x] #2 Praat TextGrid remains clean 4-tier format
- [x] #3 Unit tests pass
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reverted Praat TextGrid to standard 4-tier format to avoid Praat non-unicode font rendering issues. Added 'syllabary_word' property to word objects in alignment_manifest.json when Cherokee Syllabary is present in source data.
<!-- SECTION:FINAL_SUMMARY:END -->
