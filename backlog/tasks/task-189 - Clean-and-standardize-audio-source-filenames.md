---
id: TASK-189
title: Clean and standardize audio source filenames
status: Done
assignee:
  - '@agent'
created_date: '2026-08-05 12:34'
updated_date: '2026-08-05 12:34'
labels: []
dependencies: []
ordinal: 185000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Inspect cherokee_new_testament/audio_source and standardize filenames so they clearly map to their respective New Testament books and chapters for easy matching with transcript JSONs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Audio source directory is inspected and files mapped
- [x] #2 Filenames are cleaned and standardized for easy matching
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Renamed all 52 audio files in cherokee_new_testament/audio_source/ to clean, standardized names matching <book_lowercase>_<chapter_2digits>.mp3 (e.g. matthew_04.mp3, mark_01.mp3, philemon_01.mp3), exactly matching the naming structure of the chapter JSON files.
<!-- SECTION:FINAL_SUMMARY:END -->
