---
id: TASK-394
title: Realign New Testament with checkpoint-8000 and recompute train data threshold
status: Done
assignee:
  - '@self'
created_date: '2026-11-04 17:04'
updated_date: '2026-11-04 17:55'
labels: []
dependencies: []
modified_files:
  - digohwelisgi/core/models/inference.py
  - scripts/realign_bible.py
  - scripts/split_bible_dataset.py
  - data/projects/cherokee_new_testament/alignments/bible_alignment_records.json
  - data/projects/cherokee_new_testament/train_csvs/mark.csv
  - data/projects/cherokee_new_testament/train_csvs/matthew.csv
  - data/projects/cherokee_new_testament/train_csvs/philemon.csv
  - data/projects/cherokee_new_testament/train_csvs/philippians.csv
  - data/projects/cherokee_new_testament/train_csvs/titus.csv
  - data/training/processed/bible-wav2vec2-train.csv
  - data/training/processed/bible-wav2vec2-valid.csv
  - data/training/processed/bible-wav2vec2-test.csv
ordinal: 427300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Realign all audio-backed Cherokee New Testament books (Mark, Matthew, Philemon, Philippians, Titus) using the newly promoted best model (checkpoint-8000). Re-slice audio, update Praat TextGrids and alignment manifests, analyze the new confidence distribution, and recompute the training data confidence filtering threshold.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Update scripts/realign_bible.py to resolve best_model.json dynamically and support all audio-backed books (Mark, Matthew, Philemon, Philippians, Titus)
- [x] #2 Execute realignment across all chapters with checkpoint-8000, generating sliced audio, TextGrids, and alignment manifests
- [x] #3 Analyze confidence score distributions and recompute the minimum word confidence training filtering threshold
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update scripts/realign_bible.py to use get_best_model_config() by default and add Philemon (1 ch), Philippians (4 ch), and Titus (3 ch) to BOOK_CONFIGS.
2. Verify ScripturePipeline and CTCSegmentationAligner support the updated model and check emissions caching directory.
3. Run realignment across Mark, Matthew, Philemon, Philippians, and Titus on MPS.
4. Analyze the output word-confidence distribution across all verse intervals, plot/summarize the distribution vs pre-Bible baseline, and determine the optimal filtering threshold.
5. Update train/valid/test split CSVs or manifests with the updated threshold and verify tests.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Realigned all 52 chapters of the Cherokee New Testament (Mark 1-16, Matthew 1-28, Philemon 1, Philippians 1-4, Titus 1-3) using checkpoint-8000 with hybrid MPS feature-extraction acceleration. Generated updated sliced verse audio, Praat TextGrids, alignment manifests, and book records. Recomputed training confidence distribution: verses with min word confidence > 0.15 increased from 607 to 1,144 (an 88.5% increase), with average verse confidence at 90.5%. Regenerated Bible training splits (939 train, 105 valid, 100 test).
<!-- SECTION:FINAL_SUMMARY:END -->
