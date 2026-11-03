---
id: TASK-386
title: Package and publish Cherokee ASR training dataset to Hugging Face Hub
status: Done
assignee:
  - '@self'
created_date: '2026-11-03 19:45'
updated_date: '2026-11-03 20:03'
labels: []
dependencies:
  - TASK-385
modified_files:
  - digohwelisgi/cherokee/training/package_dataset.py
ordinal: 419300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the Cherokee-specific dataset packaging driver under Tier 2 that loads the 6 active Cherokee splits (chr and bible), packages the 2,442 referenced audio files (462MB), and publishes the dataset to charliemcvicker/cherokee-asr-data on Hugging Face Hub.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Define Cherokee dataset packaging specification targeting the 6 active CSV splits
- [x] #2 Resolve and verify the 2,442 referenced audio files across sentence_audio and split_audio
- [x] #3 Publish the dataset to charliemcvicker/cherokee-asr-data with documentation and manifest on Hugging Face Hub
- [x] #4 Verify that all 6 splits can be successfully parsed against staged audio
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Dataset is staged, validated, and accessible on Hugging Face
<!-- DOD:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create digohwelisgi/cherokee/training/package_dataset.py driver with Cherokee-specific path resolution for 6 splits.
2. Stage the 2,442 referenced audio files (462MB) and 6 split CSVs into a target staging directory.
3. Add README.md / dataset card for Hugging Face repo.
4. Execute upload to Hugging Face repo charliemcvicker/cherokee-asr-data.
5. Verify that all 6 splits can be successfully parsed against staged audio.
6. Run tests and typecheck.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Packaged and published the Cherokee ASR multi-domain dataset to private repository charliemcvicker/cherokee-asr-data on Hugging Face Hub.
- Configured Cherokee split specifications for the 6 active training splits.
- Extracted and staged the 2,442 referenced audio files (462.4 MB uncompressed, 384 MB tar.gz).
- Generated dataset card (README.md) and dataset_manifest.json.
- Uploaded single-stream tar.gz archive to bypass HF rate limits.
- Verified that 100% of referenced audio paths resolve cleanly against the dataset.
<!-- SECTION:FINAL_SUMMARY:END -->
