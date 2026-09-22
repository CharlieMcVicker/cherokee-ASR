---
id: TASK-360.4
title: 'Phase 4A: Streamline driver scripts to thin pipeline wrappers'
status: To Do
assignee: []
created_date: '2026-09-21 20:25'
updated_date: '2026-09-21 20:33'
labels: []
dependencies: []
parent_task_id: TASK-360
ordinal: 390100
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Refactor scripts/realign_bible.py and scripts/realign_gs_mm_ctc.py into thin (<30 lines) declarative runners delegating directly to ScripturePipeline and DialogueAlignmentPipeline.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 scripts/realign_bible.py delegates all chapter orchestration and slicing to ScripturePipeline in <30 lines
- [ ] #2 scripts/realign_gs_mm_ctc.py delegates dialogue alignment to DialogueAlignmentPipeline in <30 lines
- [ ] #3 End-to-end execution of scripts/realign_gs_mm_ctc.py produces matching TextGrid and manifest outputs
<!-- AC:END -->
