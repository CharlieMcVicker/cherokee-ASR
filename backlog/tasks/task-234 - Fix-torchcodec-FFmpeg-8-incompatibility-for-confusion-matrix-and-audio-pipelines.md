---
id: TASK-234
title: >-
  Fix torchcodec FFmpeg 8 incompatibility for confusion matrix and audio
  pipelines
status: Done
assignee:
  - '@antigravity'
created_date: '2026-08-30 22:09'
updated_date: '2026-08-30 22:24'
labels: []
dependencies: []
ordinal: 228000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
torchcodec fails to find FFmpeg 4-7 dynamic libraries (libavutil.59) when newer FFmpeg 8 is installed. Remove torchcodec dependency and use standard soundfile/torchaudio backend.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Remove torchcodec from pyproject.toml and conda environment
- [x] #2 Verify scripts/generate_confusion_matrix.py runs without torchcodec FFmpeg errors
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed fragile torchcodec dependency from pyproject.toml and conda environment. Refactored scripts/generate_confusion_matrix.py to use CherokeeASRModel directly with standard soundfile audio loading, allowing evaluation of all 4,120 samples cleanly without torchcodec/FFmpeg C-binding errors.
<!-- SECTION:FINAL_SUMMARY:END -->
