---
id: TASK-246
title: Delegate DTW chunk alignment cost calculation to Word Aligner
status: Done
assignee:
  - '@myself'
created_date: '2026-08-31 21:22'
updated_date: '2026-08-31 21:23'
labels: []
dependencies: []
ordinal: 248000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Refactor SlidingWindowDTWAligner so that it delegates cost/distance calculations to NeedlemanWunschWordAligner rather than maintaining duplicate distance_metric and preprocessor dependencies.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add compute_cost method to NeedlemanWunschWordAligner.
- [x] #2 Simplify SlidingWindowDTWAligner __init__ to accept word_aligner directly and delegate all scoring and preprocessing to it.
- [x] #3 Update any callers/tests and verify all tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add compute_cost(hypothesis: str, reference: str) -> float and normalize(text: str) -> str to NeedlemanWunschWordAligner.\n2. Update SlidingWindowDTWAligner to remove distance_metric and preprocessor constructor args and delegate cost computation to word_aligner.\n3. Update tests/callers if necessary.\n4. Run pytest to verify all tests pass.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Delegated chunk-level DTW distance scoring and text normalization in SlidingWindowDTWAligner to NeedlemanWunschWordAligner by adding compute_cost and normalize helper methods to the word aligner. Simplified SlidingWindowDTWAligner constructor and removed duplicate distance_metric and preprocessor fields.
<!-- SECTION:FINAL_SUMMARY:END -->
