---
id: TASK-120
title: Add processed interview segments to segmentation manifest
status: Done
assignee:
  - '@agent'
created_date: '2026-07-09 21:03'
updated_date: '2026-07-09 21:05'
labels: []
dependencies: []
ordinal: 116000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extract metadata from segmented filenames in data/processed/denoised_segments/ and append them to audiofiles-to-transcribe/segmentation_manifest.csv with the correct relative paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Determine correct relative paths for segmented files in data/processed/denoised_segments/ from audiofiles-to-transcribe/segmentation_manifest.csv
- [x] #2 Write a script or run command to parse filenames and append new rows to segmentation_manifest.csv
- [x] #3 Verify correct pathing and format of appended rows
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and executed a Python script to scan all 52,644 wav files in data/processed/denoised_segments/, parse their start/end millisecond timestamps from filenames, map their speaker folder names to original MP3 filenames in cvcs-mp3s/, format their paths using '../data/processed/denoised_segments/' relative to 'audiofiles-to-transcribe/segmentation_manifest.csv', and append the sorted rows.
<!-- SECTION:FINAL_SUMMARY:END -->
