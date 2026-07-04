---
id: TASK-94
title: Update CSV preparation scripts for semicolon and colon formatting
status: Done
assignee:
  - '@agent'
created_date: '2026-07-04 15:21'
updated_date: '2026-07-04 15:21'
labels: []
dependencies: []
ordinal: 90000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update CSV prep scripts to handle semicolons and colons: drop wordfinal ';' and replace word medial ';' with ':', then remove all non-vowel-adjacent colons as a final step.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Modify prepare_csv.py and prepare_conrad_csv.py to drop word-final semicolons
- [x] #2 Modify scripts to replace word-medial semicolons with colons
- [x] #3 Modify scripts to remove all colons that are not placed by vowels
- [x] #4 Re-run the CSV preparation pipelines to verify output
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated both prepare_csv.py and prepare_conrad_csv.py to preprocess semicolons in raw transcription texts. Semicolons at the end of words (word-final) are stripped, and medial semicolons are replaced with colons (':'). Any colons that are not placed by vowels (i.e. those remaining after the tone normalizer doubles vowels like 'a:' -> 'aa') are stripped as a final step. Re-ran the preparation pipelines to successfully rebuild the training, validation, and test splits, and verified that both semicolons and colons are fully removed from the cleaned outputs.
<!-- SECTION:FINAL_SUMMARY:END -->
