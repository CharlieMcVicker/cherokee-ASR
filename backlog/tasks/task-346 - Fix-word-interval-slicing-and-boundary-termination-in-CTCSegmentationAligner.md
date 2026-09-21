---
id: TASK-346
title: Fix word interval slicing and boundary termination in CTCSegmentationAligner
status: Done
assignee:
  - '@subagent'
created_date: '2026-09-17 16:54'
updated_date: '2026-09-17 16:58'
labels: []
dependencies: []
ordinal: 362000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix word interval end boundary calculation in _extract_word_intervals to capture trailing character acoustic durations up to the word delimiter boundary rather than prematurely truncating at the onset frame + index_duration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Word interval end timestamps include trailing character durations up to the word boundary delimiter
- [x] #2 Word interval end timestamps are strictly bounded by subsequent word start and total audio duration
- [x] #3 Viewer and alignment tests pass with zero regressions
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect _extract_word_intervals in transcription/alignment/ctc_aligner.py and check how timings[char_start_idx:end_idx] and state_list calculate the word end frame.
2. Update end_idx timing inclusion so trailing character acoustic frames / delimiter bounds are properly captured without premature cutoff.
3. Verify that w_end is bounded by next word start or total audio duration.
4. Run full test suite with pytest and ensure zero regressions.
5. Finalize task in Backlog.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated _extract_word_intervals in transcription/alignment/ctc_aligner.py so that word intervals include trailing delimiter timing and character acoustic frames up to the word boundary delimiter rather than prematurely truncating at max(w_timings) + index_duration. Ensured interval bounds are clamped strictly within [0.0, dur_sec] and verified all 276 tests in pytest pass cleanly with 0 type errors in pyright.
<!-- SECTION:FINAL_SUMMARY:END -->
