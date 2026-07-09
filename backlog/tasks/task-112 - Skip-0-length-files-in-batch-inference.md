---
id: TASK-112
title: Skip 0-length files in batch inference
status: Done
assignee:
  - '@antigravity'
created_date: '2026-07-09 20:08'
updated_date: '2026-07-09 20:10'
labels: []
dependencies: []
ordinal: 108000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Avoid errors and skip 0-length audio files during batch inference
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Check file size and skip 0-length files in transcription/inference/batch.py
- [x] #2 Check file size and skip 0-length files in server.py batch inference endpoint
- [x] #3 Check and skip audio files that are too short (less than 400 frames/samples) to be processed by Wav2Vec2
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Modify transcription/inference/batch.py to filter out 0-length files.\n2. Modify server.py to skip copying 0-length files.\n3. Test the changes.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented checks to skip 0-length files, files that are too short (less than 400 frames/samples) to be processed by Wav2Vec2, and unreadable files in transcription/inference/batch.py and server.py.
<!-- SECTION:FINAL_SUMMARY:END -->
