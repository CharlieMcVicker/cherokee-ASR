---
id: TASK-360.4
title: 'Phase 4A: Streamline driver scripts to thin pipeline wrappers'
status: Done
assignee:
  - '@phase-4a-implementor'
created_date: '2026-09-21 20:25'
updated_date: '2026-09-23 14:19'
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
- [x] #1 scripts/realign_bible.py delegates all chapter orchestration and slicing to ScripturePipeline in <30 lines
- [x] #2 scripts/realign_gs_mm_ctc.py delegates dialogue alignment to DialogueAlignmentPipeline in <30 lines
- [x] #3 End-to-end execution of scripts/realign_gs_mm_ctc.py produces matching TextGrid and manifest outputs
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect scripts/realign_bible.py and scripts/realign_gs_mm_ctc.py.
2. Refactor scripts/realign_gs_mm_ctc.py into a thin (<30 lines) runner delegating directly to DialogueAlignmentPipeline.
3. Refactor scripts/realign_bible.py into a thin (<30 lines) runner delegating directly to ScripturePipeline.
4. Verify execution of scripts/realign_gs_mm_ctc.py against saving-the-voices outputs.
5. Verify line count of both files is < 30 lines.
6. Run tests with pytest, static type checks with pyright transcription, and format with black.
7. Record validation evidence, check ACs, and complete task.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Refactored scripts/realign_gs_mm_ctc.py and scripts/realign_bible.py into declarative pipeline wrappers delegating to DialogueAlignmentPipeline and ScripturePipeline. Verified with pytest (482 passed) and pyright (0 errors).
<!-- SECTION:FINAL_SUMMARY:END -->
