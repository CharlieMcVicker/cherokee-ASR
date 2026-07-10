---
id: TASK-133
title: Create PRAAT TextGrid and Active Learning pipeline
status: Done
assignee:
  - '@agent'
created_date: '2026-07-10 14:36'
updated_date: '2026-07-10 14:39'
labels: []
dependencies: []
ordinal: 129000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build a pipeline to generate PRAAT TextGrid files with machine transcriptions, convert MP3 source files to 16kHz 16-bit WAVs, import human-approved/edited segments, and export active learning data.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Convert CSV source MP3 files to 16kHz 16-bit WAV files
- [x] #2 Generate TextGrid files with machine transcriptions mapped to a text tier
- [x] #3 Define workflow/conventions for human approval and editing in PRAAT (e.g., using specific tiers or labels)
- [x] #4 Parse updated TextGrids and export approved segments for active learning
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Locate all unique source files from 'data/results/cvcs_all_noisy.csv' (e.g. matching folder names like 'Rosanna Vann' to 'cvcs-mp3s/Rosanna Vann.mp3').
2. Convert source MP3 files to 16kHz, 16-bit, mono WAV files and save them to 'data/processed/praat/'.
3. Generate '.TextGrid' files in 'data/processed/praat/' with two tiers: 'machine' (pre-filled with greedy transcriptions from cvcs_all_noisy.csv) and 'human' (matching interval boundaries, empty text).
4. Fill all gaps between segments with empty intervals to satisfy Praat format requirements.
5. Create an export script that reads the edited '.TextGrid' files, parses the 'human' tier, extracts approved/rewritten segments from the 16kHz WAVs, saves them to 'data/processed/active_learning/', and writes an active learning manifest CSV.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented praat_prepare.py and praat_export.py scripts to automate the generation and exporting of PRAAT active learning data.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created a PRAAT TextGrid integration pipeline. The script 'transcription/utils/praat_prepare.py' converts MP3s to 16kHz mono WAVs and generates paired TextGrids with pre-aligned tiers ('machine' with ASR output, 'human' with blank intervals). The export script 'transcription/utils/praat_export.py' scans these TextGrids, extracts segments where text is present on the 'human' tier, saves them as WAV segments under 'data/processed/active_learning/', and populates 'active_learning_manifest.csv' for downstream active learning.
<!-- SECTION:FINAL_SUMMARY:END -->
