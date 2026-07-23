---
id: TASK-147
title: Fix overlapping word timestamps in DTW character alignment exporter
status: Done
assignee:
  - '@agent'
created_date: '2026-07-23 14:26'
updated_date: '2026-07-23 14:26'
labels: []
dependencies: []
ordinal: 143000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix word interval timestamp boundaries so words within a verse do not overlap when exported to Praat TextGrids or alignment manifests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Word intervals strictly sequential without overlapping timestamps
- [x] #2 Tests pass for timestamping alignment and exporter
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Refactored _align_words_char_range to clamp and partition word interval timestamps strictly sequentially (w_end <= next_w_start), eliminating interval overlap in exported Praat TextGrids.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed word timestamp overlap bug in transcription.timestamping.aligner. Ensured generated word intervals are strictly sequential (start_sec <= end_sec <= next_start_sec), preventing overlapping intervals in Praat TextGrid and JSON exports. Verified all 8 unit tests pass.
<!-- SECTION:FINAL_SUMMARY:END -->
