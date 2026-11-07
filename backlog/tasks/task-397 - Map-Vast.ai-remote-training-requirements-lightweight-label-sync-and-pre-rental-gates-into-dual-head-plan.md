---
id: TASK-397
title: >-
  Map Vast.ai remote training requirements, lightweight label sync, and
  pre-rental gates into dual-head plan
status: Done
assignee:
  - '@self'
created_date: '2026-11-07 21:00'
updated_date: '2026-11-07 21:01'
labels: []
dependencies: []
modified_files:
  - dual_head_tone_asr_plan.md
ordinal: 430300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Incorporate Vast.ai guide (doc-2) requirements into dual_head_tone_asr_plan.md, formulate lightweight label-only synchronization bypassing audio re-uploading, and define the complete dependency path and pre-rental verification checklist.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Map lightweight label-only synchronization strategy skipping audio re-packaging
- [x] #2 Document all blockers and prerequisites standing in the way before GPU rental
- [x] #3 Update dual_head_tone_asr_plan.md with Vast.ai operational deployment phase and pre-rental checklist
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Integrated Vast.ai training workflow (doc-2) and Hugging Face dataset management into dual_head_tone_asr_plan.md. Formulated the lightweight label-only synchronization strategy to skip 462MB audio re-uploading by preserving git-tracked CSVs during remote extraction. Added Section 6 outlining all 6 pre-rental prerequisites, the mandatory 5-step local MPS smoke test gate, GPU selection requirements, and remote execution with automatic Hub backups.
<!-- SECTION:FINAL_SUMMARY:END -->
