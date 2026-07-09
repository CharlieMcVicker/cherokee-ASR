---
id: TASK-101
title: Exclude all processed and frontend directories from file listings
status: Done
assignee:
  - '@myself'
created_date: '2026-07-05 18:29'
updated_date: '2026-07-05 18:29'
labels: []
dependencies: []
ordinal: 97000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The backend file listing APIs currently return audio files from training_data/processed, wav/data/processed, and frontend/public. We should prune all directories named 'processed' and the 'frontend' folder to keep the audio file lists clean.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Prune 'processed' and 'frontend' directories from dirs in /api/files endpoint in server.py
- [x] #2 Prune 'processed' and 'frontend' directories from dirs in /api/inference_files endpoint in server.py
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Modify /api/files in server.py to prune 'processed' and 'frontend' directories from the walk.\n2. Modify /api/inference_files in server.py to prune 'processed' and 'frontend' directories from the walk.\n3. Run syntax compilation check.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Pruned all directories named 'processed' and the 'frontend' directory from the os.walk inside both /api/files and /api/inference_files endpoints to ensure processed files and frontend assets are excluded.
<!-- SECTION:FINAL_SUMMARY:END -->
