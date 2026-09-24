---
id: TASK-309
title: Fix intra-verse slice cumulative word emission bug in align_verse_slice
status: Done
assignee:
  - '@myself'
created_date: '2026-09-12 20:01'
updated_date: '2026-09-12 20:32'
labels: []
dependencies: []
ordinal: 325000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix character state window extraction in align_verse_slice to avoid accumulating previous word characters when tokens are unaligned, ensuring benchmark reports cleanly separated emitted words.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Constrain character state extraction in align_verse_slice strictly to the word interval
- [x] #2 Prevent unaligned words from accumulating state characters from previous words
- [x] #3 Verify benchmark comparison output shows proper word-level emitted tokens
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. In _extract_word_intervals, filter word character timings using non-blank tokens: timings[start_idx + 1:end_idx] (or guarding if start_idx + 1 >= end_idx).
2. Ensure start_f and end_f frame bounds are strictly bounded by raw_w_start and raw_w_end, preventing unaligned/previous-word state accumulation.
3. Verify unaligned words emit empty string and 0.0 confidence without reading frame 0.
4. Update unit tests in test_ctc_aligner.py to assert strict token slicing and prevent cumulative word emissions.
5. Run benchmark_ctc_segmentation_100_verses.py and test suite to verify word-level emitted tokens.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed character state window extraction and unvisited syncope token timing filtering in _extract_word_intervals, ensuring both align_verse_slice and continuous align strictly isolate word intervals, prevent unaligned/syncope-skipped tokens from accumulating preceding word states, and cleanly emit word-level tokens across benchmarks and continuous chapter manifests.
<!-- SECTION:FINAL_SUMMARY:END -->
