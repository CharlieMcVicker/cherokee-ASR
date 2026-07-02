---
id: TASK-85
title: Refactor train.py main function
status: Done
assignee:
  - '@myself'
created_date: '2026-07-02 21:06'
updated_date: '2026-07-02 21:07'
labels: []
dependencies: []
ordinal: 81000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The main function in train.py is too long. Break things into shorter functions and run them in sequence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Identify logical blocks in main function
- [x] #2 Extract blocks into separate, well-named functions
- [x] #3 Call new functions sequentially in main
- [x] #4 Ensure training workflow behaves identically
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extract command line arguments parsing and configuration update.
2. Extract directory setup logic.
3. Extract CSV loading and path resolution.
4. Extract vocabulary building and tokenizer/processor initialization.
5. Extract dataset conversion and prep mapping.
6. Extract model loading and trainer creation.
7. Extract checkpoint resumption resolution.
8. Extract post-training checkpoint evaluation.
9. Extract best checkpoint promotion logic.
10. Extract results/summary saving.
11. Reassemble main function to invoke these functions sequentially.
12. Verify correctness by running python syntax checks.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Refactored main function in train.py by extracting logical blocks into distinct, modular functions and executing them sequentially in main.
<!-- SECTION:FINAL_SUMMARY:END -->
