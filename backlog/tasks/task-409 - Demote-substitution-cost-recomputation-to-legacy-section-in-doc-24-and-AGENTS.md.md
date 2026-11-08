---
id: TASK-409
title: >-
  Demote substitution cost recomputation to legacy section in doc-24 and
  AGENTS.md
status: Done
assignee:
  - '@self'
created_date: '2026-11-08 16:00'
updated_date: '2026-11-08 16:01'
labels: []
dependencies: []
modified_files:
  - AGENTS.md
  - >-
    backlog/docs/guides/post_training_recomputation/doc-24 -
    Post-Model-Training-Lifecycle-and-Artifact-Recomputation-Guide.md
ordinal: 442300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User clarified that empirical substitution costs (run_noisy_eval.py / PhonologicalConfusionCostMetric / NeedlemanWunschWordAligner) are low priority, rarely used, and legacy. Restructure doc-24 to separate them into their own Legacy / Secondary Workflows section with low priority, and update AGENTS.md accordingly.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Restructure doc-24 to isolate substitution costs into a dedicated low-priority legacy section
- [x] #2 Update AGENTS.md post-training protocol to designate substitution costs as optional legacy
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update doc-24 content via backlog doc update doc-24 to categorize substitution costs as low-priority legacy, moving it into a dedicated "Legacy / Secondary Workflows" section.
2. Update AGENTS.md to denote step 5 (run_noisy_eval.py) as optional legacy.
3. Verify pyright and pytest.
4. Check ACs and complete task.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Restructured doc-24 to separate empirical substitution costs into a dedicated low-priority legacy section (Section 6), clarifying that production alignment exclusively uses phonotactic CTC segmentation. Updated the recomputation matrix and operational checklist in doc-24 to designate substitution costs as P4 / optional legacy. Updated AGENTS.md post-training protocol accordingly.
<!-- SECTION:FINAL_SUMMARY:END -->
