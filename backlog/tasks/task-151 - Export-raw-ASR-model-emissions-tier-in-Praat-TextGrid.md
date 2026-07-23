---
id: TASK-151
title: Export raw ASR model emissions tier in Praat TextGrid
status: Done
assignee:
  - '@agent'
created_date: '2026-07-23 14:35'
updated_date: '2026-07-23 14:36'
labels: []
dependencies: []
ordinal: 147000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add Tier 3 (Raw ASR Emissions) to Praat TextGrid exporter containing raw ASR token word boundaries and transcription.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Praat TextGrid exports include Tier 3 Raw ASR Emissions
- [x] #2 Tests pass for 3-tier Praat TextGrid export
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added raw_tokens field to AlignmentResult and added Tier 3 (Raw ASR Emissions) to Praat TextGrid exporter in exporter.py.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated Praat TextGrid exporter in transcription.timestamping to include Tier 3 (Raw ASR Emissions). The TextGrid file now outputs 3 tiers: Tier 1 (Verses), Tier 2 (Words - Aligned ground truth terms), and Tier 3 (Raw ASR Emissions - original word boundaries and transcriptions emitted by the model). Verified all 9 unit tests pass.
<!-- SECTION:FINAL_SUMMARY:END -->
