---
id: TASK-408
title: Create Model Promotion and Post-Training Artifact Recomputation Guide
status: Done
assignee:
  - '@self'
created_date: '2026-11-08 15:52'
updated_date: '2026-11-08 15:55'
labels: []
dependencies: []
modified_files:
  - AGENTS.md
  - >-
    backlog/docs/guides/post_training_recomputation/doc-24 -
    Post-Model-Training-Lifecycle-and-Artifact-Recomputation-Guide.md
ordinal: 441300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Document the end-to-end operational procedure for promoting a newly trained Cherokee ASR model (single-head or dual-head) and recomputing all downstream artifacts. Review recent lessons from feature/rebuild-artifacts (TASK-391..TASK-394) and dual-head integration (TASK-405..TASK-407). Establish long-lived documentation in backlog/docs/ and update AGENTS.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Review feature/rebuild-artifacts and catalog all required artifact recomputations
- [x] #2 Create comprehensive long-lived guide in backlog/docs/ detailing step-by-step procedures, commands, artifact paths, and dual-head considerations
- [x] #3 Update AGENTS.md with reference to model promotion and recomputation workflow
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Review git history and tickets on feature/rebuild-artifacts (TASK-391..TASK-394) and dual-head branch (TASK-405..TASK-407) to catalog all dependent artifacts and operational steps.
2. Create a comprehensive long-lived guide in backlog/docs/guides/post_training_recomputation/ using backlog doc CLI.
3. Update AGENTS.md to link to and summarize the post-training recomputation procedure.
4. Verify documentation links, file paths, and syntax.
5. Check acceptance criteria and finalize task.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reviewed git history and prior tickets on feature/rebuild-artifacts (TASK-391..TASK-394) and dual-head integration (TASK-405..TASK-407). Created comprehensive long-lived guide in doc-24 (backlog/docs/guides/post_training_recomputation/doc-24 - Post-Model-Training-Lifecycle-and-Artifact-Recomputation-Guide.md) covering the post-model-training lifecycle, artifact dependency graph, step-by-step execution runbook, script arguments, output locations, codebase wiring, and dual-head multi-stream considerations. Synchronized AGENTS.md with the protocol and doc link. Verified zero pyright type errors and 100% test pass rate across unit tests.
<!-- SECTION:FINAL_SUMMARY:END -->
