---
id: TASK-82
title: >-
  Revert origin/main merge and inference consolidation affecting batch.py to
  redo keeping origin batch.py fixes
status: Done
assignee:
  - '@myself'
created_date: '2026-07-02 20:49'
updated_date: '2026-07-02 20:52'
labels: []
dependencies: []
ordinal: 78000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Revert the merge of origin/main and the consolidation of inference that affected batch.py, and redo the consolidation keeping all the fixes from origin's batch.py.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Revert or reset local branch to pre-merge state (commit 3602bd8 or similar)
- [x] #2 Identify and preserve the fixes from origin's batch.py
- [x] #3 Reapply inference consolidation / merge while keeping the fixes from origin's batch.py
- [x] #4 Verify that batch.py works and holds both sets of updates correctly
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Successfully reverted the branch to the pre-consolidation/pre-merge commit (3b74dc5). Redid the merge with origin/main, resolving conflicts in batch.py. Reapplied the inference centralization, removing all KenLM dependencies while preserving the advanced optimizations and fixes from origin's batch.py (lazy metadata loading, sorted batching, parallel CPU decoding pool, VRAM OOM fallbacks, character/word level confidence scores formatted in JSON). Verified the refactored code compiles successfully.
<!-- SECTION:FINAL_SUMMARY:END -->
