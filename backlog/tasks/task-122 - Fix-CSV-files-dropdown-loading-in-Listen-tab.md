---
id: TASK-122
title: Fix CSV files dropdown loading in Listen tab
status: Done
assignee:
  - '@agent'
created_date: '2026-07-09 21:06'
updated_date: '2026-07-09 21:07'
labels: []
dependencies: []
ordinal: 118000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fetch CSV files from /api/files endpoint in the View4 (Listen) component to populate the Select Results CSV dropdown, which is currently empty.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add useEffect to View4 in frontend/src/App.jsx to fetch CSV files and setCsvFiles
- [x] #2 Verify dropdown lists the files correctly
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a useEffect hook to View4 component in frontend/src/App.jsx to fetch CSV files from /api/files, populating the 'Select Results CSV' dropdown so that transcription CSV options are shown.
<!-- SECTION:FINAL_SUMMARY:END -->
