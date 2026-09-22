---
id: TASK-360.8
title: >-
  Phase 2B: Surface phonotactic constraints, syncope/intrusion masks & text
  preparer
status: Done
assignee:
  - '@phase-2b-implementor'
created_date: '2026-09-21 20:32'
updated_date: '2026-09-22 15:27'
labels: []
dependencies: []
parent_task_id: TASK-360
ordinal: 388200
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Consolidate Cherokee surface constraints (*HH, *C'), vowel syncope masks, intrusion site masks, prepare_cherokee_text into transcription.cherokee.phonotactics, and relocate PhonologicalConfusionCostMetric / ConfusionMatrixCostMetric to transcription.cherokee.distance. Retire normalizers.py in favor of direct convert_orthography calls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Surface phonotactic rules and transition masks reside in transcription.cherokee.phonotactics.phonotactics
- [x] #2 prepare_cherokee_text implements TextPreparerProtocol compatible with Tier 1 CTCSegmentationAligner
- [x] #3 PhonologicalConfusionCostMetric and ConfusionMatrixCostMetric reside in transcription.cherokee.distance as pluggable DistanceMetrics for the DP aligner
- [x] #4 Redundant normalizers.py retired in favor of direct convert_orthography calls
- [x] #5 Unit tests pass with pyright transcription reporting 0 errors
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Surface phonotactics into transcription/cherokee/phonotactics/phonotactics.py and __init__.py: PhonemeCategory, tokens, inventories, syncope masks, intrusion site masks, validity check, and prepare_cherokee_text conforming to TextPreparerProtocol.\n2. Relocate/consolidate Cherokee distance metrics into transcription/cherokee/distance.py: PhonologicalConfusionCostMetric, ConfusionMatrixCostMetric (and re-export LevenshteinDistanceMetric, etc.), conforming to DistanceMetric protocol, and re-export in transcription/cherokee/__init__.py.\n3. Retire normalizers.py by refactoring normalizer functions into direct convert_orthography calls and updating alignment modules to direct convert_orthography, while keeping backwards-compatible forwarding in normalizers.py and phonotactics.py.\n4. Write tests in transcription/cherokee/tests/test_cherokee_phonotactics.py and test_cherokee_distance.py.\n5. Verify test suite across all suites and 0 pyright errors, then format with black.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Consolidated Cherokee phonotactic rules, inventories, syncope/intrusion masks, and prepare_cherokee_text (conforming to TextPreparerProtocol) into transcription.cherokee.phonotactics. Relocated PhonologicalConfusionCostMetric and ConfusionMatrixCostMetric into transcription.cherokee.distance, re-exported across transcription.cherokee. Retired normalizers.py usages in favor of direct convert_orthography calls while preserving backward compatibility. Verified with all 282 tests passing on pytest, 0 pyright errors, and black formatting applied.
<!-- SECTION:FINAL_SUMMARY:END -->
