---
id: TASK-119
title: Implement speaker and segment dropdown pagination for Listen tab
status: Done
assignee: []
created_date: '2026-07-09 20:59'
updated_date: '2026-07-09 21:00'
labels: []
dependencies: []
ordinal: 115000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Group massive CSV files by speaker/session name to keep the options dropdown list size small. Provide a second dropdown to select individual segments for listening, preventing browser crashes from 50,000+ option elements.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Modify View4 grouping logic in App.jsx to group by speaker/session
- [x] #2 Add a second dropdown for Segment/File selection when multiple segments are present
- [x] #3 Update wavesurfer loading and lyricsData extraction to use the selected segment
- [x] #4 Verify the Listen tab functions correctly without freezing
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented speaker-level grouping and segment-level pagination dropdowns in the Listen tab (View4) of App.jsx. Segment WAV files in massive CSVs are grouped by speaker/session name (e.g. Elmer Panther, George Cochran). Selecting a speaker renders a secondary dropdown list to pick individual segments (e.g. Elmer Panther_segment_xxxx.wav) rather than rendering 52,000 options at once, preventing browser crashes and allowing the user to select and play any segment audio file.
<!-- SECTION:FINAL_SUMMARY:END -->
