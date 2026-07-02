---
id: TASK-88
title: >-
  Fix resuming training from Hugging Face snapshot by locating checkpoint
  subdirectory
status: Done
assignee:
  - '@antigravity'
created_date: '2026-07-02 22:16'
updated_date: '2026-07-02 22:17'
labels: []
dependencies: []
ordinal: 84000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resuming training from a Hugging Face Hub snapshot fails with FileNotFoundError when trainer_state.json is not at the root but inside a checkpoint subdirectory, or when the repo doesn't contain the trainer state files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Find the checkpoint subdirectory (e.g. checkpoint-*) in the downloaded Hugging Face snapshot, and use it as resume_from_checkpoint
- [x] #2 If no checkpoint subdirectory exists but trainer_state.json is not found, print a helpful warning or fallback gracefully if possible
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved the FileNotFoundError when resuming training from Hugging Face Hub. Added logic in resolve_resume_checkpoint to scan the snapshot download directory for any 'checkpoint-*' subdirectories (matching the trainer state folder structure). If found, it uses the latest checkpoint directory. If no trainer state/checkpoint directory is found at all, it issues a warning and falls back to loading the model weights directly from the downloaded snapshot instead of starting completely from the base XLS-R checkpoint.
<!-- SECTION:FINAL_SUMMARY:END -->
