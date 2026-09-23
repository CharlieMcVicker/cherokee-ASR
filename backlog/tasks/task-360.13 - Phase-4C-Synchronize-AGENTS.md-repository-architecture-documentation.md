---
id: TASK-360.13
title: 'Phase 4C: Synchronize AGENTS.md & repository architecture documentation'
status: Done
assignee:
  - '@myself'
created_date: '2026-09-21 20:33'
updated_date: '2026-09-23 15:32'
labels: []
dependencies: []
parent_task_id: TASK-360
ordinal: 390300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update AGENTS.md and all 11 technical guides in docs/ to document the 4-tier modular architecture and updated import paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 AGENTS.md updated with new package paths (transcription.core, transcription.cherokee, transcription.pipelines, transcription.apps)
- [x] #2 All 11 markdown guides in docs/ updated to reflect the new architecture
- [x] #3 No obsolete import paths or deprecated references remain in documentation
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Review current AGENTS.md and docs/*.md files to identify outdated paths and architecture descriptions.
2. Update AGENTS.md to describe the 4-tier architecture (transcription.core, transcription.cherokee, transcription.pipelines, transcription.apps).
3. Update all 11 technical guides in docs/ with accurate package paths and architecture diagrams.
4. Verify no dead links or obsolete module paths remain.
5. Finalize task and dispatch reviewer.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated AGENTS.md and technical guides in docs/ (including models_and_inference.md) to document the 4-tier modular architecture (transcription.core, transcription.cherokee, transcription.pipelines, transcription.apps). Removed all obsolete CLI script and threshold_finder references.
<!-- SECTION:FINAL_SUMMARY:END -->
