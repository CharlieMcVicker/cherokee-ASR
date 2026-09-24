---
id: TASK-301
title: >-
  Fix boundary timing condition and unaligned word emission guarding in
  CTCSegmentationAligner
status: Done
assignee: []
created_date: '2026-09-12 19:36'
updated_date: '2026-09-12 19:40'
labels: []
dependencies: []
ordinal: 317000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix t >= 0.0 condition in CTCSegmentationAligner to preserve 0.0s start boundary tokens and guard unaligned words from erroneously slicing frame 0 characters and confidences.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Change timing filter from t > 0.0 to t >= 0.0 in align_verse_slice and align
- [x] #2 Guard unaligned words (empty w_timings) to emit empty string and 0.0 confidence without reading frame 0
- [x] #3 Adjust min_window_size and max_window_size parameter passing in align
<!-- AC:END -->
