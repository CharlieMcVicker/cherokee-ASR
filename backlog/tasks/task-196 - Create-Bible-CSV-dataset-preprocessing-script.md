---
id: TASK-196
title: Create Bible CSV dataset preprocessing script
status: Done
assignee:
  - '@agent'
created_date: '2026-08-05 13:45'
updated_date: '2026-08-05 13:45'
labels: []
dependencies: []
ordinal: 192000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write a python preprocessing script to split Mark and Matthew Bible CSVs by chapter boundaries into train, valid, and test sets.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Script splits mark.csv and matthew.csv by chapter boundaries without data leakage
- [x] #2 Generates bible-wav2vec2-train.csv, bible-wav2vec2-valid.csv, and bible-wav2vec2-test.csv in training_data/processed/
- [x] #3 Output CSVs match columns path and sentence
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create split script scripts/split_bible_dataset.py using chapter-level splits.
2. Execute script under conda env cherokee-asr.
3. Verify created CSV output files and row counts.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created scripts/split_bible_dataset.py which splits Mark and Matthew CSVs into 3 datasets based on chapter boundaries: bible-wav2vec2-train.csv (1,395 rows), bible-wav2vec2-valid.csv (182 rows), and bible-wav2vec2-test.csv (172 rows).
<!-- SECTION:FINAL_SUMMARY:END -->
