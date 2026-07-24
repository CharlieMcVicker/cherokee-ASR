---
id: TASK-164
title: >-
  Move VAD audio segmentation functions from scripts/process_interviews.py to
  transcription.audio.segment
status: Done
assignee:
  - '@agent'
created_date: '2026-07-24 15:22'
updated_date: '2026-07-24 15:23'
labels: []
dependencies: []
ordinal: 160000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move get_best_parameters and split_long_segments_smart to transcription/audio/segment.py so timestamping module can import them without relying on scripts folder
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 get_best_parameters and split_long_segments_smart are in transcription.audio.segment
- [x] #2 audio_segmenter.py and process_interviews.py import from transcription.audio.segment
- [x] #3 align-cherokee CLI executes without missing scripts module error
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Moved VAD functions get_best_parameters and split_long_segments_smart into transcription.audio.segment and updated audio_segmenter.py and process_interviews.py to use relative/module imports.
<!-- SECTION:FINAL_SUMMARY:END -->
