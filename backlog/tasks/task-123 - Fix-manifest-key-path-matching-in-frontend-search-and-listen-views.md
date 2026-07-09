---
id: TASK-123
title: Fix manifest key path matching in frontend search and listen views
status: Done
assignee:
  - '@agent'
created_date: '2026-07-09 21:07'
updated_date: '2026-07-09 21:08'
labels: []
dependencies: []
ordinal: 119000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the strict path matching logic where row.file_path and mainSeg.file_path are compared with manifest keys. The presence of leading '../' in newly added segments causes endsWith comparisons to fail. Normalize paths by stripping leading dot/dots and checking bidirectional endsWith.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Update View3 manifest matching logic in frontend/src/App.jsx
- [x] #2 Update View4 manifest matching logic in frontend/src/App.jsx
- [x] #3 Verify dropdown and segment selection work for multi-source files in the Listen tab
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated View3 and View4 manifest key matching loops in frontend/src/App.jsx to strip leading dot/dots and perform a robust bidirectional endsWith match. This fixes the issue where CSV data file_paths could not match manifest keys due to leading '../' folder prefixes.
<!-- SECTION:FINAL_SUMMARY:END -->
