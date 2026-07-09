---
id: TASK-124
title: Force full audio playback and remove segment selection from Listen tab
status: Done
assignee:
  - '@agent'
created_date: '2026-07-09 21:08'
updated_date: '2026-07-09 21:08'
labels: []
dependencies: []
ordinal: 120000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update View4 (Listen) component in frontend/src/App.jsx to always load the full source audio file instead of individual segment files. Remove the segment selection dropdown completely and process all segments for synced playback.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Modify View4 segment processing to always use all sorted segments
- [x] #2 Modify View4 WaveSurfer loading to always load the full source audio URL
- [x] #3 Remove the segment selector select element from View4 render method
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed segment selection logic from the Listen tab (View4 component in App.jsx), ensuring that selecting a source audio file always loads and plays the full original file while keeping the lyrics highlighting synced to all processed segments.
<!-- SECTION:FINAL_SUMMARY:END -->
