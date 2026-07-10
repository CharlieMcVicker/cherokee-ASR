---
id: TASK-130
title: Extract noisy segments from CVCS interviews
status: Done
assignee:
  - '@antigravity'
created_date: '2026-07-10 13:03'
updated_date: '2026-07-10 13:11'
labels: []
dependencies: []
ordinal: 126000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Use the segmentation manifest to locate all segments for CVCS interviews in cvcs-mp3s, extract them directly from the raw MP3 files without spectral noise subtraction, normalize their volume, and save them in data/processed/noisy_segments.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Parse segmentation_manifest.csv and filter for CVCS interview mp3 files
- [x] #2 Write a script to extract segments directly from the MP3 files without denoising
- [x] #3 Normalize the volume of each extracted segment
- [x] #4 Save segments in data/processed/noisy_segments/ retaining the speaker subfolder structure
- [x] #5 Verify and document the output segments count and size
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Parse segmentation_manifest.csv to group start and end times by MP3 filename.\n2. Write scripts/extract_noisy_segments.py to read each MP3 from cvcs-mp3s, convert/load as 16kHz mono, slice segments directly using manifest millisecond timestamps, normalize each using pydub, and export under data/processed/noisy_segments/ in corresponding speaker folders.\n3. Run the extraction script and monitor execution.\n4. Verify the output count and integrity of files.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Successfully extracted 52,644 raw (noisy) segments from the 51 interview MP3 files in cvcs-mp3s/ without spectral noise subtraction. Each segment was normalized in peak volume and exported as a 16kHz mono WAV file to data/processed/noisy_segments/<SpeakerName>/, fully matching the filenames and layout listed in segmentation_manifest.csv. Verified that 52,644 WAV files were generated, totalling 3.1 GB on disk.
<!-- SECTION:FINAL_SUMMARY:END -->
