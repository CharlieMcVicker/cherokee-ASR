---
id: TASK-95
title: Drop transcription rows containing f from CSV splits
status: Done
assignee:
  - '@agent'
created_date: '2026-07-04 15:23'
updated_date: '2026-07-04 15:24'
labels: []
dependencies: []
ordinal: 91000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update CSV prep scripts to drop any rows where the transcription contains the letter 'f'.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Modify prepare_csv.py to drop rows containing 'f'
- [x] #2 Modify prepare_conrad_csv.py to drop rows containing 'f'
- [x] #3 Re-run the preparation pipelines and verify that 'f' is no longer in the vocabulary/output characters
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Modified prepare_csv.py and prepare_conrad_csv.py to filter out and drop any rows containing the character 'f' in the final cleaned transcription text. Re-ran both scripts to rebuild the splits. Verified via Python that the rebuilt train, validation, and test CSV splits contain exactly zero occurrences of the letter 'f'.
<!-- SECTION:FINAL_SUMMARY:END -->
