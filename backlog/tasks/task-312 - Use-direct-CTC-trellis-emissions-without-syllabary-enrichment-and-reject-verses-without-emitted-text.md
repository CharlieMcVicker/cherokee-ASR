---
id: TASK-312
title: >-
  Use direct CTC trellis emissions without syllabary enrichment and reject
  verses without emitted text
status: Done
assignee:
  - '@myself'
created_date: '2026-09-12 20:55'
updated_date: '2026-09-12 20:57'
labels: []
dependencies: []
ordinal: 328000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Bypass legacy syllabary enrichment in CTC alignment pipeline (realign_bible.py and pipeline.py) to use direct syncope- and intrusion-aware trellis emissions. Eliminate phonetic fallback so verses without emitted text or with anomalies are excluded from high-quality training CSVs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Use chunk.emitted_text directly without reconcile_syllabary_asr in realign_bible.py and pipeline.py
- [x] #2 Exclude verses without emitted_text or with anomalies from training CSV export without phonetic fallback
- [x] #3 Update benchmark_ctc_segmentation_100_verses.py to use direct trellis emissions
- [x] #4 Run test suite and verify all tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect realign_bible.py, pipeline.py, and benchmark_ctc_segmentation_100_verses.py.
2. Update realign_bible.py to use direct chunk.emitted_text and strictly exclude verses without emitted_text or with anomalies from training CSVs without fallback to phonetic text.
3. Update pipeline.py to use direct word emissions when engine='ctc' without running legacy reconcile_alignment_words.
4. Update benchmark_ctc_segmentation_100_verses.py to evaluate direct trellis emissions.
5. Run test suite to verify tests pass and check for regressions.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Bypassed legacy heuristic syllabary enrichment in realign_bible.py, pipeline.py, and benchmark_ctc_segmentation_100_verses.py in favor of direct syncope- and intrusion-aware CTC trellis emissions. Enforced strict training CSV quality control: verses without emitted text or with flagged anomalies are excluded without fallback to phonetic text. Added unit test and verified 249/249 tests passing.
<!-- SECTION:FINAL_SUMMARY:END -->
