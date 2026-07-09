---
id: TASK-126
title: Optimize CSV loading by allowing audio file selection first
status: Done
assignee: []
created_date: '2026-07-09 21:13'
updated_date: '2026-07-09 21:14'
labels: []
dependencies: []
ordinal: 122000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Loading a combined CSV file (like CVCS_all.csv containing 51 interviews and over 50,000 rows) loads too much data at once, leading to slow response times and browser memory issues. This task will optimize the process by allowing the user to select the source audio file/interview first, and then fetch only the segments for that specific file from the backend.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add /api/labeler/audio-files endpoint to server.py to get unique audio source/interview names from a CSV
- [x] #2 Update get_labeler_data (/api/labeler/data) in server.py to support an optional audio_file query parameter
- [x] #3 Update View4 in frontend/src/App.jsx to fetch the list of audio files first upon selecting a CSV
- [x] #4 Update View4 in frontend/src/App.jsx to fetch segments only for the selected audio file
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a new route `@app.get("/api/labeler/audio-files")` in [server.py](file:///Users/charlesmcvicker/code/workshop-transcription/server.py) to read the CSV file quickly and extract unique audio/speaker group names.
2. Modify `get_labeler_data` in [server.py](file:///Users/charlesmcvicker/code/workshop-transcription/server.py) to accept an optional `audio_file` query parameter. If provided, only parse and return segments matching that group name.
3. Update `View4` component in [App.jsx](file:///Users/charlesmcvicker/code/workshop-transcription/frontend/src/App.jsx):
   - Upon selecting a CSV, fetch the list of unique audio files from `/api/labeler/audio-files`.
   - Update the UI to render the Source Audio File dropdown immediately using this list.
   - When a source audio file is selected, fetch the detailed segment data for that specific file from `/api/labeler/data?file=...&audio_file=...`.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Optimized CSV loading by allowing the user to select the source audio file/interview first, and then only loading segments for the selected interview.
<!-- SECTION:FINAL_SUMMARY:END -->
