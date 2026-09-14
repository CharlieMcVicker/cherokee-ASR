---
id: TASK-311
title: >-
  Filter flagged anomaly verses from training data export and add test for Mark
  1:1 typo exclusion
status: Done
assignee:
  - '@myself'
created_date: '2026-09-12 20:45'
updated_date: '2026-09-12 20:46'
labels: []
dependencies: []
ordinal: 327000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filter out verses containing transcript typos or low-confidence alignment anomalies from generated training CSVs (train_csvs/mark.csv, etc.) and add an automated test verifying that Mark 1:1 is flagged and excluded from the training dataset in the full pipeline.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Exclude chunk.has_anomalies verses from training CSV rows in realign_bible.py
- [x] #2 Add test verifying Mark 1:1 is flagged for anomaly/typo and excluded from training CSV
- [x] #3 Ensure full test suite passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. In scripts/realign_bible.py, only add verses to csv_rows for training data export if not chunk.has_anomalies (excluding flagged verses).
2. Add a test in transcription/new_testament/tests/test_pipeline.py asserting that Mark 1:1 gets flagged as an anomaly and excluded from training CSV export in the pipeline.
3. Rerun realign_bible.py and verify Mark 1:1 is flagged in alignment records and absent from mark.csv.
4. Run full pytest suite to verify all tests pass.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Filtered anomalous verses (chunk.has_anomalies) from training CSV generation in realign_bible.py, and added unit tests in test_pipeline.py asserting that verses with transcript typos (such as Mark 1:1) are flagged with has_anomalies: True and excluded from the exported training CSVs.
<!-- SECTION:FINAL_SUMMARY:END -->
