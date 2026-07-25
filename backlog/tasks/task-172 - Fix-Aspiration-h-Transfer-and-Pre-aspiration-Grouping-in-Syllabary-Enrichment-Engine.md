---
id: TASK-172
title: >-
  Fix Aspiration (h) Transfer and Pre-aspiration Grouping in Syllabary
  Enrichment Engine
status: Done
assignee:
  - '@agent'
created_date: '2026-07-25 20:28'
updated_date: '2026-07-25 20:29'
labels: []
dependencies: []
ordinal: 168000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix enrich_syllabary.py to preserve model-predicted aspiration (h) across vowel endings (e.g., yo + ASR yoh -> yoh, ha + ASR hah -> hah) and correctly group pre-aspiration h with s-clusters.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Preserve vocalic/post-vocalic aspiration h from ASR aligned window
- [x] #2 Group pre-aspiration h correctly with subsequent s-cluster syllables
- [x] #3 Verify CER improvement with inspect_pipeline.py and evaluate_reconciliation.py
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect enrich_syllabary.py and test_enrich_syllabary.py.\n2. Update _enrich_single_syllable() to preserve vocalic/post-vocalic aspiration h emitted by ASR (e.g., yo + ASR yoh -> yoh, ha + ASR hah -> hah).\n3. Fix pre-aspiration h grouping to attach h to pre-aspirated s-clusters and syllable boundaries.\n4. Run unit tests and verify CER improvement with evaluate_reconciliation.py.\n5. Mark ACs checked and set TASK-172 to Done.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented post-vocalic aspiration h preservation and pre-aspiration h grouping in enrich_syllabary.py. Added comprehensive unit tests in test_enrich_syllabary.py and verified CER improvement across evaluation datasets.
<!-- SECTION:FINAL_SUMMARY:END -->
