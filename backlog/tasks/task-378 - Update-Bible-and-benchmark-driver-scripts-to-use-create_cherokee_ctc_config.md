---
id: TASK-378
title: Update Bible and benchmark driver scripts to use create_cherokee_ctc_config
status: Done
assignee:
  - '@self'
created_date: '2026-11-02 21:13'
updated_date: '2026-11-02 21:16'
labels: []
dependencies: []
modified_files:
  - scripts/realign_bible.py
  - scripts/benchmark_ctc_segmentation_100_verses.py
  - scripts/realign_gs_mm_ctc.py
ordinal: 411300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update scripts/realign_bible.py, scripts/benchmark_ctc_segmentation_100_verses.py, and scripts/realign_gs_mm_ctc.py to instantiate aligner configs using create_cherokee_ctc_config rather than bare CTCAlignerConfig, ensuring Cherokee syncope classes and intrusive tokens are correctly inherited.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Update scripts/realign_bible.py to use create_cherokee_ctc_config in realign_book and main
- [x] #2 Update scripts/benchmark_ctc_segmentation_100_verses.py to use create_cherokee_ctc_config
- [x] #3 Update scripts/realign_gs_mm_ctc.py to use create_cherokee_ctc_config
- [x] #4 Verify syncope_tokens and intrusive_tokens are populated with Cherokee phonotactic defaults in instantiated configs
- [x] #5 Ensure pytest suite passes and pyright reports 0 errors
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Replace CTCAlignerConfig imports and instantiations with create_cherokee_ctc_config in scripts/realign_bible.py
2. Replace CTCAlignerConfig imports and instantiations with create_cherokee_ctc_config in scripts/benchmark_ctc_segmentation_100_verses.py and scripts/realign_gs_mm_ctc.py
3. Check scripts/tune_ctc_aligner_config.py or other scripts if applicable
4. Verify syncope_tokens are properly initialized across scripts
5. Run pytest and pyright to ensure zero regressions
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Replaced bare CTCAlignerConfig instantiations with create_cherokee_ctc_config across scripts/realign_bible.py, scripts/benchmark_ctc_segmentation_100_verses.py, and scripts/realign_gs_mm_ctc.py. Cleaned up obsolete kwargs (enforce_phonotactics) and stale imports. Verified that syncope_tokens and intrusive_tokens are populated with Cherokee phonotactic defaults and that full pytest suite (358 passed) and pyright (0 errors) pass cleanly.
<!-- SECTION:FINAL_SUMMARY:END -->
