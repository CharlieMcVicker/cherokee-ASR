---
id: TASK-156
title: Support multi-token to multi-word ASR GT fusion in DP aligner
status: Done
assignee:
  - '@agent'
created_date: '2026-07-23 15:16'
updated_date: '2026-07-23 15:17'
labels: []
dependencies: []
ordinal: 152000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend Needleman-Wunsch DP alignment in transcription/timestamping/aligner.py to allow merging consecutive ASR emission tokens (M tokens) against GT words (K words) to handle word segmentation errors like 'word1wo ord2' -> 'word1 word2'.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 DP aligner evaluates M-token to K-word fusion transitions (up to M=3, K=3)
- [x] #2 Unit tests cover split word emission alignment cases
- [x] #3 Existing aligner tests pass cleanly
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Modify _align_words_char_range in transcription/timestamping/aligner.py to iterate over M consecutive ASR tokens (m=1..MAX_FUSE_ASR) and K consecutive GT words (k=1..MAX_FUSE_GT).\n2. Calculate string edit cost between concatenated ASR tokens (without spaces or with single space) and fused GT text.\n3. Add unit test to tests/test_timestamping.py or aligner tests validating 'word1wo ord2' fusion to 'word1 word2'.\n4. Run pytest to verify all alignment tests pass.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Supported multi-token ASR to multi-word GT fusion (M tokens to K GT words) in Needleman-Wunsch DP alignment engine. Updated sliding window candidate matching to consider unspaced concatenated tokens, and added unit tests in test_aligner.py.
<!-- SECTION:FINAL_SUMMARY:END -->
