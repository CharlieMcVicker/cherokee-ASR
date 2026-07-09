---
id: TASK-108
title: Create speech vs noise classifier heuristic for audio segments
status: Done
assignee:
  - '@agent'
created_date: '2026-07-09 19:18'
updated_date: '2026-07-09 19:20'
labels: []
dependencies: []
ordinal: 104000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Formulate a heuristic to classify audio segments as speaking (not noise) or non-speaking (noise), test it on a sample of segments, and label them for user review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Develop RMS and dBFS based classification heuristic
- [ ] #2 Implement diagnostic script to classify and list sample segments with their metrics
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Developed a speech vs noise classification heuristic based on RMS ratio to the noise floor, average volume, and peak volume. Integrated this classification into the main process_interviews.py script. By default, it filters out and drops non-speech (noise) segments, but users can use the --keep-noise CLI option to preserve them with a _noise suffix.
<!-- SECTION:FINAL_SUMMARY:END -->
