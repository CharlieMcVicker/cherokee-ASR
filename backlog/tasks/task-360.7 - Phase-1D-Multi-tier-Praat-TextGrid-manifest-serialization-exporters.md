---
id: TASK-360.7
title: 'Phase 1D: Multi-tier Praat TextGrid & manifest serialization exporters'
status: To Do
assignee: []
created_date: '2026-09-21 20:32'
labels: []
dependencies: []
parent_task_id: TASK-360
ordinal: 387400
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extract language-agnostic multi-tier Praat TextGrid builders and structured JSON alignment manifest exporters into transcription.core.exporters.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 IntervalTier, TextGridBuilder, and export_textgrid reside in transcription.core.exporters.textgrid
- [ ] #2 export_manifest and export_debug_json reside in transcription.core.exporters.manifest
- [ ] #3 TextGrid and manifest serialization unit tests pass with formatting and boundary parity
<!-- AC:END -->
