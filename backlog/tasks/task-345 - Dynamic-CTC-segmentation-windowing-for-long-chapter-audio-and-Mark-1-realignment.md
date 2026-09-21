---
id: TASK-345
title: >-
  Dynamic CTC segmentation windowing for long chapter audio and Mark 1
  realignment
status: Done
assignee:
  - '@myself'
created_date: '2026-09-16 18:09'
updated_date: '2026-09-16 18:59'
labels: []
dependencies: []
ordinal: 361000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Scale CTC segmentation window_size dynamically based on total audio frames (max(min_window_size, num_frames)) in CTCSegmentationAligner to eliminate search trellis truncation on long chapter recordings (>160s). Realign Mark Chapter 1 and verify accurate boundary timestamps for verses 40-45.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement dynamic window size scaling in CTCSegmentationAligner.align() based on audio frame count
- [x] #2 Verify Mark 1 verses 40-45 align with full duration up to 615+ seconds
- [x] #3 Realign Mark Chapter 1 and export updated alignment records, TextGrids, and HTML viewer
- [x] #4 Run pytest and pyright to ensure zero regressions
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update CTCSegmentationAligner.align() in transcription/alignment/ctc_aligner.py to dynamically scale min_window_size and max_window_size to max(self.min_window_size, lpz.shape[0]) and max(self.max_window_size, lpz.shape[0] * 2).
2. Update CTCSegmentationAligner.align_verse_slice() with equivalent dynamic window sizing.
3. Realign Mark Chapter 1 and entire Mark book via scripts/realign_bible.py.
4. Export updated HTML comparison viewer using scripts/view_ctc_comparison.py.
5. Verify verses 40-45 boundaries and word timings.
6. Run full pytest suite (275 tests) and pyright transcription.
7. Finalize task.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented dynamic window scaling win_size = max(self.min_window_size, min(20000, int(lpz.shape[0]))) in CTCSegmentationAligner.align() to eliminate search trellis truncation on long audio recordings (>160s). Realigned Mark Chapter 1 and exported updated alignment records and HTML viewer. Verified that verses 40-45 now align cleanly across 516s-616s (with verse 45 ending at 615.74s) and match exact spoken ASR emissions. All 275 pytest tests and pyright pass with 0 errors.
<!-- SECTION:FINAL_SUMMARY:END -->
