---
id: TASK-377
title: >-
  Synchronize downstream Bible manifests and training datasets with realigned
  records
status: Done
assignee:
  - '@self'
created_date: '2026-11-02 21:07'
updated_date: '2026-11-02 21:08'
labels: []
dependencies: []
modified_files:
  - data/projects/cherokee_new_testament/alignments/bible_alignment_records.json
  - data/projects/cherokee_new_testament/alignments/mark_alignment_records.json
  - >-
    data/projects/cherokee_new_testament/alignments/matthew_alignment_records.json
  - data/projects/cherokee_new_testament/train_csvs/mark.csv
  - data/projects/cherokee_new_testament/train_csvs/matthew.csv
  - data/training/processed/bible-wav2vec2-test.csv
  - data/training/processed/bible-wav2vec2-train.csv
  - data/training/processed/bible-wav2vec2-valid.csv
  - uv.lock
ordinal: 410300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Synchronize per-book alignment records (mark_alignment_records.json, matthew_alignment_records.json), training CSVs (mark.csv, matthew.csv), and dataset splits (bible-wav2vec2-train/valid/test.csv) with the freshly realigned bible_alignment_records.json, clean up redundant backup file, and commit uv.lock and working tree.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Export mark_alignment_records.json and matthew_alignment_records.json from bible_alignment_records.json
- [x] #2 Regenerate train_csvs/mark.csv (426 verses) and train_csvs/matthew.csv (685 verses) excluding anomalous verses
- [x] #3 Run split_bible_dataset.py to update bible-wav2vec2-train, valid, and test CSV splits
- [x] #4 Remove redundant bible_alignment_records.backup.json
- [x] #5 Verify test suite passes and uv.lock is clean
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Slice mark_alignment_records.json and matthew_alignment_records.json from bible_alignment_records.json
2. Export mark.csv and matthew.csv filtering out verses with has_anomalies: True
3. Execute scripts/split_bible_dataset.py to update training splits in data/training/processed/
4. Remove bible_alignment_records.backup.json
5. Run pytest and uv lock --check to ensure tests pass and lockfile is verified
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Synchronized per-book alignment manifests (mark_alignment_records.json: 678 records, matthew_alignment_records.json: 1,071 records), training CSVs (mark.csv: 426 unflagged verses, matthew.csv: 685 unflagged verses), and train/validation/test splits (bible-wav2vec2-train.csv: 894 rows, bible-wav2vec2-valid.csv: 114 rows, bible-wav2vec2-test.csv: 103 rows) with the newly realigned bible_alignment_records.json. Removed redundant backup file and confirmed clean uv.lock and passing pytest test suite (358 passed).
<!-- SECTION:FINAL_SUMMARY:END -->
