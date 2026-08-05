---
id: TASK-194
title: Generate Mark training CSV and 16kHz verse audio slices with reconciliation
status: Done
assignee:
  - '@agent-k'
created_date: '2026-08-05 13:14'
updated_date: '2026-08-05 13:21'
labels: []
dependencies: []
ordinal: 190000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Process all 16 chapters of Mark using audio_source mp3s and book_transcripts JSON files. Slices audio into 16kHz WAV files named mark_CH_VS.wav in cherokee_new_testament/split_audio, outputs CSV to cherokee_new_testament/train_csvs/mark.csv with path and sentence columns, and computes min/max/median verse duration and total audio length.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Create cherokee_new_testament/split_audio and cherokee_new_testament/train_csvs directories
- [x] #2 Process Mark chapters 1-16 end-to-end using align_chapter with reconciliation
- [x] #3 Export 16kHz WAV clips named mark_XX_YY.wav for each matched verse
- [x] #4 Output cherokee_new_testament/train_csvs/mark.csv with path and sentence columns
- [x] #5 Compute and report min, max, median verse length and total audio duration
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create script to iterate over mark_01 to mark_16 audio and transcript JSON files\n2. Run alignment with reconciliation for each chapter\n3. Slice each matched verse from full audio and export as 16kHz MONO WAV to cherokee_new_testament/split_audio/mark_XX_YY.wav\n4. Write CSV rows (path, sentence) to cherokee_new_testament/train_csvs/mark.csv\n5. Calculate audio statistics (min, max, median verse length, total seconds)
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Processed all 16 chapters of the Book of Mark. Generated 678 verse audio clips (16kHz MONO WAV) saved to cherokee_new_testament/split_audio/mark_XX_YY.wav. Created training CSV at cherokee_new_testament/train_csvs/mark.csv containing path and reconciled sentence columns. Computed verse audio statistics: Min: 2.90s, Max: 34.34s, Median: 12.99s, Total: 9,378.87 seconds (~2.6 hours).
<!-- SECTION:FINAL_SUMMARY:END -->
