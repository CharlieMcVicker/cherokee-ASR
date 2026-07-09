---
id: TASK-99
title: Exclude data/processed directory from file listing APIs
status: Done
assignee:
  - '@myself'
created_date: '2026-07-05 18:25'
updated_date: '2026-07-05 18:25'
labels: []
dependencies: []
ordinal: 95000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The frontend's listen tab lists all wav files recursively under the sandbox directory. Since data/processed contains a huge number of processed segments, it should be excluded to avoid cluttering the UI and loading too many audio files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Exclude the 'data/processed' directory in /api/files endpoint in server.py
- [x] #2 Exclude the 'data/processed' directory in /api/inference_files endpoint in server.py
- [x] #3 Ensure os.walk does not traverse into data/processed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Modify /api/files in server.py to prune 'processed' when walking the 'data' directory.\n2. Modify /api/inference_files in server.py to prune 'processed' when walking the 'data' directory.\n3. Run server check to ensure changes are syntax-valid.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Excluded the 'processed' subdirectory when walking the 'data' directory in both /api/files and /api/inference_files endpoints, preventing thousands of processed WAV segments from cluttering the frontend select options.
<!-- SECTION:FINAL_SUMMARY:END -->
