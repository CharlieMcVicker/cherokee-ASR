---
id: TASK-98
title: Update strip_length masking logic for vowel-colon format and run evaluation
status: Done
assignee:
  - '@myself'
created_date: '2026-07-04 20:58'
updated_date: '2026-07-04 21:03'
labels: []
dependencies: []
ordinal: 94000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update the strip_length function in transcription/inference/infer.py to support the vowel-colon format (V:) for long vowels, then run evaluation on the best model from best_model.json.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Update strip_length in infer.py to collapse V: to V
- [x] #2 Run evaluate_checkpoint.py with the best model configuration
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated strip_length in infer.py to support vowel-colon format (V:) and modified evaluate_checkpoint.py to default to the best model from best_model.json. Re-ran evaluation, achieving a length-masked WER of 0.1863.
<!-- SECTION:FINAL_SUMMARY:END -->
