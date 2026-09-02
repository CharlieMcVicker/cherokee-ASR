---
id: TASK-268
title: >-
  Update docs/alignment.md with prepare_alignment_input and representation-aware
  normalizers
status: Done
assignee:
  - '@antigravity'
created_date: '2026-09-02 16:10'
updated_date: '2026-09-02 16:11'
labels:
  - docs
  - alignment
dependencies: []
priority: low
ordinal: 270000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update docs/alignment.md documentation:
1. Document prepare_alignment_input sum-type dispatcher in Ingestion section.
2. Document normalize_syllabary_for_alignment (aspiration stripped) and normalize_phonetics_for_alignment (aspiration preserved) in Normalizers section.
3. Update Python programmatic pipeline examples to use prepare_alignment_input and CherokeeASRModel.from_pretrained_or_best.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Document prepare_alignment_input in docs/alignment.md
- [x] #2 Document normalize_syllabary_for_alignment and normalize_phonetics_for_alignment in docs/alignment.md
- [x] #3 Update programmatic alignment examples in docs/alignment.md
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated docs/alignment.md with:
1. Documentation for prepare_alignment_input sum-type dispatcher in Ingestion section and summary table.
2. Documentation for normalize_syllabary_for_alignment (aspiration stripped) and normalize_phonetics_for_alignment (aspiration preserved) in Normalizers section.
3. Updated programmatic Python alignment examples using prepare_alignment_input and CherokeeASRModel.from_pretrained_or_best.
<!-- SECTION:FINAL_SUMMARY:END -->
