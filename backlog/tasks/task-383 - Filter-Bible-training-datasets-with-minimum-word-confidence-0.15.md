---
id: TASK-383
title: Filter Bible training datasets with minimum word confidence > 0.15
status: Done
assignee:
  - '@self'
created_date: '2026-11-03 19:18'
updated_date: '2026-11-03 19:20'
labels: []
dependencies: []
modified_files:
  - scripts/split_bible_dataset.py
  - data/projects/cherokee_new_testament/train_csvs/mark.csv
  - data/projects/cherokee_new_testament/train_csvs/matthew.csv
  - data/training/processed/bible-wav2vec2-train.csv
  - data/training/processed/bible-wav2vec2-valid.csv
  - data/training/processed/bible-wav2vec2-test.csv
ordinal: 416300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filter Bible training data on disk based on word-level alignment confidences (min word confidence > 0.15) to produce high-fidelity training splits without altering aligner configuration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Create downstream filtering procedure in scripts/split_bible_dataset.py to filter Bible verses with min word confidence > 0.15
- [x] #2 Update train_csvs/mark.csv (235 verses) and train_csvs/matthew.csv (372 verses) with the 0.15 confidence filter
- [x] #3 Regenerate bible-wav2vec2-train.csv, bible-wav2vec2-valid.csv, and bible-wav2vec2-test.csv splits
- [x] #4 Verify all unit tests pass with zero regressions
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update scripts/split_bible_dataset.py to support filtering by min_word_confidence directly from bible_alignment_records.json or train_csvs
2. Filter mark.csv and matthew.csv so every word has confidence > 0.15
3. Regenerate bible-wav2vec2-train.csv, bible-wav2vec2-valid.csv, and bible-wav2vec2-test.csv splits
4. Run pytest and pyright to ensure zero regressions
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR-Style Summary:
- Updated `scripts/split_bible_dataset.py` to implement a clean downstream filtering stage (`align -> confidences -> filtered list`) without modifying the aligner configuration or re-running alignment.
- Filtered verses where all word-level CTC segmentation confidences are > 0.15 directly from `bible_alignment_records.json`.
- Filtered `mark.csv` (235 verses) and `matthew.csv` (372 verses) for a total of 607 high-fidelity Bible verses.
- Regenerated training splits: `bible-wav2vec2-train.csv` (479 verses, 78.9%), `bible-wav2vec2-valid.csv` (71 verses, 11.7%), and `bible-wav2vec2-test.csv` (57 verses, 9.4%).
- Confirmed all 369 unit tests pass and pyright reports 0 errors.
<!-- SECTION:FINAL_SUMMARY:END -->
