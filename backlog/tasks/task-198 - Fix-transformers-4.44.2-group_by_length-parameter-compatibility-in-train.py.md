---
id: TASK-198
title: Fix transformers 4.44.2 group_by_length parameter compatibility in train.py
status: Done
assignee:
  - '@agent'
created_date: '2026-08-05 15:00'
updated_date: '2026-08-05 15:00'
labels: []
dependencies: []
ordinal: 194000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update train.py to use group_by_length=True instead of train_sampling_strategy='group_by_length' for HuggingFace Transformers 4.44.2 compatibility.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 train.py uses group_by_length=True
- [x] #2 train.py initializes TrainingArguments without TypeError on transformers 4.44.2
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Downgraded transformers to 4.44.2 in local conda env and updated train.py TrainingArguments to use group_by_length=True.
<!-- SECTION:FINAL_SUMMARY:END -->
