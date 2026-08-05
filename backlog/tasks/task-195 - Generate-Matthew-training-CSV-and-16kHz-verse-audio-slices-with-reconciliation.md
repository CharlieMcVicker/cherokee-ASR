---
id: TASK-195
title: Generate Matthew training CSV and 16kHz verse audio slices with reconciliation
status: Done
assignee:
  - '@agent-k'
created_date: '2026-08-05 13:25'
updated_date: '2026-08-05 13:36'
labels: []
dependencies: []
ordinal: 191000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Process all 28 chapters of Matthew using audio_source mp3s and book_transcripts JSON files. Slices audio into 16kHz WAV files named matthew_CH_VS.wav in cherokee_new_testament/split_audio, outputs CSV to cherokee_new_testament/train_csvs/matthew.csv with path and sentence columns, and computes min/max/median verse duration, total audio length, and stats for verses < 20s.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Process Matthew chapters 1-28 end-to-end using align_chapter with reconciliation
- [x] #2 Export 16kHz WAV clips named matthew_XX_YY.wav for each matched verse
- [x] #3 Output cherokee_new_testament/train_csvs/matthew.csv with path and sentence columns
- [x] #4 Compute and report min, max, median verse length, total audio duration, and < 20s stats
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create process_matthew_dataset.py script for all 28 chapters of Matthew\n2. Run alignment with reconciliation for matthew_01 to matthew_28\n3. Slice 16kHz WAV clips into cherokee_new_testament/split_audio/matthew_XX_YY.wav\n4. Write cherokee_new_testament/train_csvs/matthew.csv\n5. Calculate overall and < 20s audio duration statistics
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Processed all 28 chapters of the Book of Matthew. Generated 1,071 16kHz MONO WAV verse clips saved to cherokee_new_testament/split_audio/matthew_XX_YY.wav and created training CSV cherokee_new_testament/train_csvs/matthew.csv. Overall statistics: Min: 2.02s, Max: 45.98s, Median: 13.81s, Total audio: 15,908.25s (4.42 hours). Under 20s cutoff stats: 867 verses (81.0%) for 10,805.41s of audio (3.00 hours).
<!-- SECTION:FINAL_SUMMARY:END -->
