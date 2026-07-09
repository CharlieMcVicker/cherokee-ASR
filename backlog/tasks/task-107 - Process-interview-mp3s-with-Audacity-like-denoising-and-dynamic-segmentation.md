---
id: TASK-107
title: Process interview mp3s with Audacity-like denoising and dynamic segmentation
status: Done
assignee:
  - '@agent'
created_date: '2026-07-09 19:13'
updated_date: '2026-07-09 19:16'
labels: []
dependencies: []
ordinal: 103000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Read interview MP3s, automatically find a silent section using pydub, perform noise reduction/denoising, segment the audio using dynamically tuned parameters (max 10s), and save segments as 16-bit 16kHz WAV files under a directory structured by the original MP3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Find silent profile via pydub silence detection
- [x] #2 Denoise audio using the noise profile
- [x] #3 Dynamically tune segmentation parameters to ensure no segment is > 10s
- [x] #4 Save segments as 16-bit 16kHz WAV in folders named after the source files
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated the segmentation process to avoid arbitrary splitting of long segments. Created a new smart splitter (split_long_segments_smart) that recursively targets segments > 10s by scanning for short/quiet silences (pauses) within active speech, or finding the absolute quietest 100ms window near the middle of the segment. This ensures we never cut audio in the middle of words, maintaining high ASR accuracy.
<!-- SECTION:FINAL_SUMMARY:END -->
