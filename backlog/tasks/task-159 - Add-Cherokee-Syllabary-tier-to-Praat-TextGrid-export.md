---
id: TASK-159
title: Add Cherokee Syllabary tier to Praat TextGrid export
status: Done
assignee:
  - '@agent'
created_date: '2026-07-23 16:27'
updated_date: '2026-07-23 16:27'
labels: []
dependencies: []
ordinal: 155000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add an optional or automatic Cherokee Syllabary IntervalTier to Praat TextGrid export when syllabary text is present in the ground truth data, using padded word interval boundaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Syllabary tier exported in Praat TextGrid when cherokee_syllabary is present
- [x] #2 Syllabary tier matches padded word boundaries
- [x] #3 Unit tests cover Syllabary tier export
- [x] #4 All timestamping tests pass
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added optional Cherokee Syllabary IntervalTier to Praat TextGrid exporter in exporter.py. Maps Cherokee Syllabary text onto padded word boundaries when cherokee_syllabary is present in source data. Added unit tests in test_exporter.py.
<!-- SECTION:FINAL_SUMMARY:END -->
