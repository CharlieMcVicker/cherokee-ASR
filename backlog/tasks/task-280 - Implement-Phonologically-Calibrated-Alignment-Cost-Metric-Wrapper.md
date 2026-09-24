---
id: TASK-280
title: Implement Phonologically-Calibrated Alignment Cost Metric Wrapper
status: Done
assignee:
  - '@subagent-280'
created_date: '2026-09-10 18:20'
updated_date: '2026-09-10 18:22'
labels:
  - alignment
  - distance-metrics
  - phonology
dependencies: []
ordinal: 292000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a new module (calibrated_distance_metrics.py) with a wrapper class (PhonologicalConfusionCostMetric) that wraps ConfusionMatrixCostMetric without altering disk weight artifacts. Dynamically applies empirical character-specific vowel deletion penalties (i=0.46, v=0.60, o=0.63, a=0.68, u=0.87, e=0.89) and discounted aspiration insertion cost (h=0.10).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Create new module transcription/alignment/calibrated_distance_metrics.py with PhonologicalConfusionCostMetric wrapping DistanceMetric
- [x] #2 Apply character-specific empirical deletion costs for Cherokee vowels and low insertion cost for aspiration 'h'
- [x] #3 Ensure underlying disk weight files remain unmodified
- [x] #4 Add comprehensive unit tests verifying DP calculation and cost adjustments
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented PhonologicalConfusionCostMetric wrapping DistanceMetric in transcription/alignment/calibrated_distance_metrics.py. Calibrates empirical Cherokee vowel drop frequencies dynamically via probability_to_normalized_cost, applies discounted aspiration 'h' insertion costs (0.10), leaves disk weight artifacts unmodified, and exports across alignment package with full unit test coverage.
<!-- SECTION:FINAL_SUMMARY:END -->
