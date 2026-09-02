---
id: TASK-269.9
title: Implement CLI driver in scripts/run_noisy_eval.py
status: Done
assignee:
  - '@pipeline-engineer'
created_date: '2026-09-02 16:47'
updated_date: '2026-09-02 16:55'
labels:
  - evaluation
  - cli
dependencies: []
parent_task_id: TASK-269
priority: high
type: feature
ordinal: 280000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement unified CLI driver scripts/run_noisy_eval.py for running SNR evaluation sweeps, extracting confusion matrices, generating cost JSON artifacts, and rendering manifold visualizer plots.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CLI arguments for subset, SNR levels, noise types, and output directories
- [x] #2 Orchestrate evaluator, accumulator, cost engine, manifold analyzer, and visualizer
- [x] #3 Export eval records JSONL, confusion CSV, cost JSON, and visualization PNGs
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented scripts/run_noisy_eval.py CLI driver orchestrating end-to-end evaluation sweeps, cost artifact generation, and visualizations.
<!-- SECTION:FINAL_SUMMARY:END -->
