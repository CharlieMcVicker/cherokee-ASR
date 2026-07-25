---
id: TASK-169
title: Build evaluate_reconciliation.py Evaluation Framework
status: To Do
assignee: []
created_date: '2026-07-25 19:55'
labels: []
dependencies: []
ordinal: 165000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build end-to-end evaluation runner computing baseline raw CER vs. reconciled CER across train, validation, and test splits. Consumes cached alignment manifests from disk for rapid iteration and outputs summary metrics table plus eval_results.json.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Build evaluate_reconciliation.py orchestrating batch alignment loading and reconciliation
- [ ] #2 Compute Raw CER, Reconciled CER, and Relative Improvement delta CER
- [ ] #3 Print formatted metrics breakdown by split (train, validation, test, overall)
- [ ] #4 Save detailed evaluation artifact eval_results.json with per-line predictions and scores
<!-- AC:END -->
