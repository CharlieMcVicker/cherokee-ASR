---
id: TASK-155
title: Add padded GT word boundary export tier to Praat TextGrid
status: Done
assignee:
  - '@antigravity'
created_date: '2026-07-23 14:51'
updated_date: '2026-07-23 14:51'
labels: []
dependencies: []
ordinal: 151000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add an extra export tier to praat export from gt timestamping module. Export word boundaries with 10ms added to pad the end of them (ie. word_end+=10msec) if words would overlap, fuse the groundtruth for them and emit one segment in the grid.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement padded GT word boundary tier logic with 10ms padding and overlap fusion
- [x] #2 Verify TextGrid output contains the new export tier and passes validation
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added Tier 3 'Padded Words' to Praat TextGrid export in transcription/timestamping/exporter.py with 10ms end padding and ground-truth word fusion logic when padded intervals overlap.
<!-- SECTION:FINAL_SUMMARY:END -->
