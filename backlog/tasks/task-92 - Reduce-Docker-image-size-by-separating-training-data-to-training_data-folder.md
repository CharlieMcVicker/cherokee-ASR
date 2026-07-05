---
id: TASK-92
title: Reduce Docker image size by separating training data to training_data/ folder
status: Done
assignee:
  - '@myself'
created_date: '2026-07-03 16:13'
updated_date: '2026-07-03 16:15'
labels: []
dependencies: []
ordinal: 88000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move train/validation/test CSVs and the referenced audio files to a training_data/ folder, update paths in training scripts, and exclude training_data/ from the Docker image to reduce its size.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Create training_data/ directory structure and move cim-wav2vec2-train.csv, cim-wav2vec2-valid.csv, cim-wav2vec2-test.csv there
- [x] #2 Move referenced training/validation/test audio files (sentence_audio and conrad_wavs) to training_data/
- [x] #3 Update CSV file paths inside the CSVs to point to training_data/ instead of data/
- [x] #4 Update references to data paths in python files (e.g. train.py/transcription) to look for training data under training_data/
- [x] #5 Update Dockerfile to COPY training_data instead of data folder
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create training_data/ directory structure and move train/valid/test CSVs from data/processed to training_data/processed.\n2. Move sentence_audio/ and conrad_wavs/ from data/processed to training_data/processed/.\n3. Update CSV file paths inside the CSVs to point to training_data/ instead of data/.\n4. Update references to data paths in python files (e.g. train.py/transcription) to look for training data under training_data/.\n5. Update Dockerfile to copy training_data instead of data folder.\n6. Test and verify everything works.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Separated training data and inference data by relocating train/valid/test CSV splits and their referenced audio files from 'data/' to a new 'training_data/' folder. Updated paths inside the CSV splits and default argument values/configurations in python scripts (train.py, evaluate_checkpoint.py, evaluate_local_checkpoints.py, evaluate_revisions.py, prepare_conrad_csv.py, prepare_csv.py) to point to the new 'training_data/processed/' location. Updated Dockerfile to COPY 'training_data' instead of the full 'data' folder, ensuring training data is preserved in the training image without inflating it with unrelated data/raw or results files.
<!-- SECTION:FINAL_SUMMARY:END -->
