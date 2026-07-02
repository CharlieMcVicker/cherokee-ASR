---
id: TASK-84
title: >-
  Integrate vowel-length masked evaluation at end of training and flag best
  models
status: Done
assignee:
  - '@myself'
created_date: '2026-07-02 21:05'
updated_date: '2026-07-02 21:05'
labels: []
dependencies: []
ordinal: 80000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update the post-training evaluation logic in train.py to compute both standard and vowel-length masked (strip_length) WER/CER metrics, and report/flag the best checkpoints for both configurations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Compute standard and vowel-length masked WER/CER for each checkpoint in train.py
- [x] #2 Rank and identify the best checkpoint for standard WER
- [x] #3 Rank and identify the best checkpoint for masked WER
- [x] #4 Log or print the best models clearly in the training output summary
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated train.py to import strip_length from the centralized inference helpers, calculate both unmasked and vowel-length masked WER/CER metrics at the end of training for all checkpoints, and explicitly report the best checkpoint for both standard and masked configurations.
<!-- SECTION:FINAL_SUMMARY:END -->
