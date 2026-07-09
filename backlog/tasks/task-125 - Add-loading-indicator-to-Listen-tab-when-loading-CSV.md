---
id: TASK-125
title: Add loading indicator to Listen tab when loading CSV
status: Done
assignee: []
created_date: '2026-07-09 21:12'
updated_date: '2026-07-09 21:12'
labels: []
dependencies: []
ordinal: 121000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Listen tab does not display a loading indicator or progress messages while fetching and parsing CSV data and segment manifests, which can take a long time. This task will add a visual loading state to keep the user informed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add an isLoading loading state to View4 in frontend/src/App.jsx
- [x] #2 Render a loading spinner or 'Loading...' message when data is being fetched/processed
- [x] #3 Clear previous audio files list and lyrics when a new CSV starts loading
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add state variables `loading` and `loadingStatus` to `View4` component in [App.jsx](file:///Users/charlesmcvicker/code/workshop-transcription/frontend/src/App.jsx).
2. Update the CSV loading `useEffect` hook in `View4` to:
   - Clear existing `audioFiles`, `lyricsData`, and `selectedAudio` when a new CSV is selected.
   - Set `loading` to true and set an appropriate status message (e.g. 'Loading transcription data...').
   - Set `loading` to false once fetching and parsing are done (both success and catch paths).
3. Update the render layout of `View4` to check if `loading` is true and show a premium loading spinner or loading message on screen.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added loading indicator and clear logic to Listen tab (View4) when loading a CSV
<!-- SECTION:FINAL_SUMMARY:END -->
