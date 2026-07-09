---
id: TASK-100
title: Support fallback paths for audio file retrieval
status: Done
assignee:
  - '@myself'
created_date: '2026-07-05 18:26'
updated_date: '2026-07-05 18:27'
labels: []
dependencies: []
ordinal: 96000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The frontend requests original audio files using the 'wav/' prefix. If the user places their original audio files in 'data/raw/', the server fails with a 404. We should add a fallback in the /api/audio endpoint to search other directories like 'data/raw/' if the file isn't found in the requested path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Modify get_audio endpoint in server.py to check 'data/raw/' when requested file in 'wav/' is missing
- [ ] #2 Ensure the endpoint works seamlessly for audio files located in 'data/raw/'
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Modify /api/audio endpoint in server.py to search 'data/raw' and other directories as fallback.\n2. Run syntax compile verification.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Aborted: The user requested to manually place the files instead of modifying server.py for audio file fallbacks.
<!-- SECTION:FINAL_SUMMARY:END -->
