---
id: TASK-241
title: Remove audio_source from ChunkAlignmentEngine and SlidingWindowDTWAligner
status: Done
assignee:
  - '@antigravity'
created_date: '2026-08-31 20:50'
updated_date: '2026-08-31 20:51'
labels: []
dependencies: []
ordinal: 243000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove audio_source parameter from ChunkAlignmentEngine protocol and SlidingWindowDTWAligner.align_chunks. Let AlignmentPipeline set source_id on AlignmentOutput for outbound adapters.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Remove audio_source from ChunkAlignmentEngine.align_chunks in protocols.py
- [x] #2 Remove audio_source from SlidingWindowDTWAligner.align_chunks in sliding_window.py
- [x] #3 Update AlignmentPipeline.run to set alignment.source_id directly
- [x] #4 Update all test suites and verify pyright/pytest pass with 0 errors
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update protocols.py to remove audio_source from ChunkAlignmentEngine.align_chunks.
2. Update sliding_window.py to remove audio_source from align_chunks.
3. Update pipeline.py so AlignmentPipeline.run sets alignment.source_id.
4. Update test_core_aligner.py and test_pipeline.py.
5. Run pyright and pytest.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed audio_source parameter from ChunkAlignmentEngine.align_chunks port and SlidingWindowDTWAligner implementation, decoupling pure chunk alignment from audio source metadata. AlignmentPipeline now sets source_id on the resulting AlignmentOutput for outbound adapters. Verified 0 pyright errors and 86/86 pytest tests passing.
<!-- SECTION:FINAL_SUMMARY:END -->
