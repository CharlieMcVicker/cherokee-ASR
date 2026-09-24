---
id: TASK-355
title: >-
  Refactor alignment config wiring, ingestion normalizer builder, and
  phonotactic char_list
status: Done
assignee: []
created_date: '2026-09-21 17:04'
updated_date: '2026-09-21 17:10'
labels: []
dependencies: []
ordinal: 381000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Code review identified redundant parameter passing in get_logits_cached, duplicated normalizer closures in ingestion, and nested ternary logic in prepare_cherokee_text.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pass parameters via CTCAlignerConfig instance in get_logits_cached in transcription/alignment/ctc_aligner.py
- [x] #2 Unify normalizer factory _build_chunk_normalizer in transcription/alignment/ingestion.py
- [x] #3 Flatten char_list resolution in prepare_cherokee_text in transcription/alignment/phonotactics.py with explicit ValueError on missing char_list
- [x] #4 All existing tests pass with 0 errors
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Refactored alignment config instantiation via typed CTCAlignerConfig, unified normalizer factory in ingestion, and flattened char_list resolution.
<!-- SECTION:FINAL_SUMMARY:END -->
