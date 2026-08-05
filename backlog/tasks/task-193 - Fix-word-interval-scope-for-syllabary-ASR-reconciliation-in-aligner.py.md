---
id: TASK-193
title: Fix word interval scope for syllabary/ASR reconciliation in aligner.py
status: Done
assignee:
  - '@agent-k'
created_date: '2026-08-05 12:57'
updated_date: '2026-08-05 12:58'
labels: []
dependencies: []
ordinal: 189000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix bug where full verse emitted_text was passed into reconcile_phonetics for individual word intervals, causing word duplications in Praat TextGrid Tier 4.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pass matched word token emission slice into reconcile_phonetics per WordInterval
- [x] #2 Re-run Mark 1 alignment and confirm single word per interval in TextGrid Tier 4
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect aligner.py word interval creation and token mapping in _align_words_char_range\n2. Attach matched ASR tokens / emitted word slice to each WordInterval\n3. Update reconciliation block in align_tokens_to_verses to pass word-level ASR emitted text\n4. Re-run Mark 1 alignment and inspect Praat TextGrid Tier 4
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed word interval scoping bug in aligner.py by storing emitted_word slice per WordInterval during DP character alignment. Individual word intervals in Praat TextGrid Tier 4 now receive clean single-word phonological reconciliation without string concatenation or duplication.
<!-- SECTION:FINAL_SUMMARY:END -->
