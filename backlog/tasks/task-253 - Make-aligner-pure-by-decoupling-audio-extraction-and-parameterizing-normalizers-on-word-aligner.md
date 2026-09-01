---
id: TASK-253
title: >-
  Make aligner pure by decoupling audio extraction and parameterizing
  normalizers on word aligner
status: Done
assignee:
  - '@pure-aligner-subagent'
created_date: '2026-09-01 18:52'
updated_date: '2026-09-01 18:54'
labels: []
dependencies: []
ordinal: 255000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Refactor SlidingWindowDTWAligner and NeedlemanWunschWordAligner in aligner.py: 1. Remove audio/extractor dependency from aligner (caller passes emissions directly). 2. Remove helper function align_chunks. 3. Parameterize chunk_normalizer and emission_normalizer on NeedlemanWunschWordAligner defaulting to identity. 4. Update cli.py and all test suites.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 NeedlemanWunschWordAligner accepts optional chunk_normalizer and emission_normalizer (defaulting to identity) and applies them in compute_cost and align_words.
- [x] #2 SlidingWindowDTWAligner accepts word_aligner and max_skip args only, with align(emissions, chunks, source_id='') signature (no audio or extractor dependencies).
- [x] #3 Remove align_chunks helper function from aligner.py and __init__.py.
- [x] #4 Update cli.py to call extractor.extract(audio) before passing emissions to aligner.align.
- [x] #5 Update all test files and verify 100% of tests pass.
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Refactored aligner.py into pure functional alignment engine: parameterized chunk_normalizer and emission_normalizer (defaulting to identity) on NeedlemanWunschWordAligner, decoupled SlidingWindowDTWAligner from audio extraction by taking emissions Sequence directly and relying on word_aligner normalizers, removed align_chunks helper, updated cli.py to perform extraction prior to alignment, and updated test suite with 100% pass rate.
<!-- SECTION:FINAL_SUMMARY:END -->
