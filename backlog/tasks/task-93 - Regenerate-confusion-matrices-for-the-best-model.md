---
id: TASK-93
title: Regenerate confusion matrices for the best model
status: Done
assignee:
  - '@agent'
created_date: '2026-07-04 15:14'
updated_date: '2026-07-04 15:17'
labels: []
dependencies: []
ordinal: 89000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Evaluate the best model against test data and all data, and generate confusion matrices.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Evaluate best model on test dataset and all data splits
- [x] #2 Compute character-level confusion matrix with insertions, deletions, substitutions, and correct characters
- [x] #3 Generate a report or heatmap of the confusion matrices
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Evaluated the best model (charliemcvicker/length-only-20260703-171218-asr-cherokee-extra-data) on the test split (186 items) and all splits combined (4122 items). Aligned references and predictions to construct character-level confusion matrices, capturing substitutions, insertions (<ins>), and deletions (<del>). Saved confusion matrices to data/results/ and generated matplotlib heatmap visualizations.
<!-- SECTION:FINAL_SUMMARY:END -->
