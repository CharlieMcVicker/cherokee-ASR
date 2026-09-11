---
id: TASK-292
title: >-
  Fix hi'a normalization, lower intrusion penalty, and verify alignment across
  100 verses
status: Done
assignee:
  - '@agent'
created_date: '2026-09-11 18:31'
updated_date: '2026-09-11 18:38'
labels: []
dependencies: []
ordinal: 304000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix normalizer stripping 'h' from H-syllables (hi-a -> hia), use normalize_phonetics_for_alignment in CTCSegmentationAligner, lower intrusive_penalty from 0.5 to 0.1, and verify that hi'a aligns accurately with high confidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Preserve aspiration/h on H-row syllables and use normalize_phonetics_for_alignment in CTCSegmentationAligner
- [x] #2 Lower intrusive_penalty to 0.1 in CTCSegmentationAligner
- [x] #3 Rerun 100-verse benchmark and verify hi'a aligns with high confidence and flagged=False
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated normalizer to preserve aspiration 'h' on H-row syllables, calibrated syncope_penalty to 2.0 and intrusive_penalty to 0.1 in CTCSegmentationAligner, and verified across 100 verses that hi'a aligns accurately with high confidence and flagged=False while true anomalies are properly isolated.
<!-- SECTION:FINAL_SUMMARY:END -->
