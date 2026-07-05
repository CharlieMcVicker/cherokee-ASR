---
id: TASK-96
title: Drop transcription rows containing b from CSV splits
status: Done
assignee:
  - '@agent'
created_date: '2026-07-04 15:24'
updated_date: '2026-07-04 15:24'
labels: []
dependencies: []
ordinal: 92000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update CSV prep scripts to drop any rows where the transcription contains the letter 'b' in addition to 'f'.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Modify prepare_csv.py to drop rows containing 'b'
- [x] #2 Modify prepare_conrad_csv.py to drop rows containing 'b'
- [x] #3 Re-run the preparation pipelines and verify that 'b' is no longer in the vocabulary/output characters
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Modified prepare_csv.py and prepare_conrad_csv.py to filter out and drop any rows containing the character 'b' (in addition to 'f'). Re-ran both preparation pipelines and verified that 1 row containing 'b' was dropped from the conrad dataset. Verified via Python that the output CSV splits contain exactly zero occurrences of the characters 'f' and 'b'.
<!-- SECTION:FINAL_SUMMARY:END -->
