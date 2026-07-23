---
id: TASK-149
title: Fix Praat TextGrid interval gaps and invalid bounds
status: Done
assignee:
  - '@agent'
created_date: '2026-07-23 14:29'
updated_date: '2026-07-23 14:29'
labels: []
dependencies: []
ordinal: 145000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix Praat TextGrid export to ensure intervals are fully contiguous without gaps/overlaps and xmin/xmax match total tier duration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Praat TextGrid exports load error-free in Praat
- [x] #2 Tests pass for TextGrid export validation
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added _build_contiguous_intervals in exporter.py to guarantee strict Praat TextGrid specifications (continuous interval ranges with gap-filling empty text intervals from 0.0 to total_end).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed Praat TextGrid export error 'Wrong xmin and xmax / TextGrid not finished'. Updated export_praat_textgrid in transcription.timestamping.exporter to build fully contiguous IntervalTiers by auto-inserting unannotated empty intervals for any gaps between start/end boundaries and tier start (0.0) / end times. Verified all 9 unit tests pass.
<!-- SECTION:FINAL_SUMMARY:END -->
