---
id: TASK-127
title: Flow short audio segments inline in Listen tab lyrics view
status: Done
assignee: []
created_date: '2026-07-09 21:16'
updated_date: '2026-07-09 21:17'
labels: []
dependencies: []
ordinal: 123000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Currently, each segment in the Listen tab is rendered as a block-level div. Since segments can be very short (often containing only one or two words, resulting in hundreds of segments per file), this causes the transcript/lyrics to display as a long, sparse vertical list. This task will update the layout to flow the segments inline, rendering them as a continuous paragraph of text.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Change the segment wrapper in View4 from a block-level div to an inline-block/span or style it to display inline
- [x] #2 Ensure correct spacing between words across segment boundaries
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Change the segment container wrapper in the lyrics view of `View4` from a block-level `div` to an inline `span`.
2. Append a trailing space `{" "}` after each segment span so that words across adjacent segments flow together horizontally as a continuous paragraph with correct spacing.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Changed the segment wrapper from a block-level div to an inline span, with trailing spaces, allowing short/noisy segments to flow horizontally like a natural transcript paragraph.
<!-- SECTION:FINAL_SUMMARY:END -->
