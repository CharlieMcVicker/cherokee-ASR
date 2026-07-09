---
id: TASK-129
title: Revert lyrics view to render segments on separate lines
status: Done
assignee: []
created_date: '2026-07-09 21:34'
updated_date: '2026-07-09 21:34'
labels: []
dependencies: []
ordinal: 125000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The user prefers to keep each segment on its own line rather than flowing them inline. This task will restore the block-level wrapper for segments while keeping the correct segment timing alignment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Change the segment wrapper back to a block-level div in App.jsx
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reverted the lyrics layout so that each segment is rendered in its own separate block/line (div) as requested, while retaining the correct timeline-aligned highlighting.
<!-- SECTION:FINAL_SUMMARY:END -->
