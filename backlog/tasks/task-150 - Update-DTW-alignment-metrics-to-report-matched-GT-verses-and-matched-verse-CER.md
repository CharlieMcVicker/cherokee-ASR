---
id: TASK-150
title: Update DTW alignment metrics to report matched GT verses and matched-verse CER
status: Done
assignee:
  - '@agent'
created_date: '2026-07-23 14:33'
updated_date: '2026-07-23 14:34'
labels: []
dependencies: []
ordinal: 146000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update compute_alignment_metrics to report matched vs total GT verses and compute CER exclusively within matched verse intervals.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Matched verse ratio calculated against total GT verses in provided metadata
- [x] #2 CER calculated exclusively across matched verses
- [x] #3 Tests pass for updated alignment metrics
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated compute_alignment_metrics to calculate CER exclusively across matched verses and combine matched GT verse count / total GT verses ratio.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated DTW alignment metrics in transcription.timestamping. Alignment metrics now report 'Matched GT Verses: matched_verses / total_verses (ratio%)' and calculate CER exclusively across matched verse regions rather than unaligned entire ground-truth book metadata. Verified all 9 unit tests pass.
<!-- SECTION:FINAL_SUMMARY:END -->
