---
id: TASK-320
title: >-
  Execute full New Testament continuous realignment and export production
  training datasets
status: To Do
assignee: []
created_date: '2026-09-14 13:09'
labels: []
dependencies: []
ordinal: 336000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run end-to-end continuous CTC segmentation realignment across all chapters of Mark and Matthew using the calibrated phonotactic masks and per-token penalties. Export production alignment records, high-quality training CSVs (with anomaly filtering), and 4-tier Praat TextGrids.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Run realign_bible.py across all chapters of Mark and Matthew
- [ ] #2 Verify anomaly filtering excludes all flagged verses from training CSVs
- [ ] #3 Verify Praat TextGrids and combined bible_alignment_records.json are exported
- [ ] #4 Generate dataset summary metrics (hours, mean verse duration, anomaly counts)
<!-- AC:END -->
