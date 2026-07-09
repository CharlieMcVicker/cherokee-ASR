---
id: TASK-128
title: Fix segment manifest matching in Listen tab lazy loader
status: Done
assignee: []
created_date: '2026-07-09 21:17'
updated_date: '2026-07-09 21:17'
labels: []
dependencies: []
ordinal: 124000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When segments are loaded lazily for the selected audio file in the Listen tab, all segments incorrectly share the manifest of the first segment (which contains duplicate start/end times). This breaks the highlighting logic during playback. This task will fix it by resolving the manifest individually for each segment row.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Update lazy loading useEffect in App.jsx to match manifest for each segment individually
- [x] #2 Verify highlighting behaves correctly during audio playback
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Modify the lazy loading `useEffect` hook in [App.jsx](file:///Users/charlesmcvicker/code/workshop-transcription/frontend/src/App.jsx) to resolve the manifest individually for each row inside the `.map` function.
2. Store the correct manifest on each segment so word starts/ends align correctly with the audio file timeline.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed lazy segment loading to resolve the manifest individually for each row inside the map function, restoring correct timing coordinates and playback highlighting.
<!-- SECTION:FINAL_SUMMARY:END -->
