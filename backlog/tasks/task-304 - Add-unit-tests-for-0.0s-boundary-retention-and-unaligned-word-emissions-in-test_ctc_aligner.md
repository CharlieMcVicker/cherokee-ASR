---
id: TASK-304
title: >-
  Add unit tests for 0.0s boundary retention and unaligned word emissions in
  test_ctc_aligner
status: Done
assignee: []
created_date: '2026-09-12 19:36'
updated_date: '2026-09-12 19:40'
labels: []
dependencies: []
ordinal: 320000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add targeted unit tests in test_ctc_aligner.py verifying 0.0s start timestamp retention, unaligned word empty emissions, and window size propagation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add test asserting 0.0s start timestamp is preserved
- [x] #2 Add test asserting unaligned word (empty timings) returns empty emitted string and 0.0 confidence without reading frame 0
- [x] #3 Verify all pytest tests pass
<!-- AC:END -->
