---
id: TASK-337
title: >-
  Prevent illegal laryngeal intrusion before pre-aspirated sibilants (hs) in
  phonotactic masking
status: Done
assignee:
  - '@agent'
created_date: '2026-09-16 16:26'
updated_date: '2026-09-16 16:28'
labels: []
dependencies: []
ordinal: 353000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update transcription/alignment/phonotactics.py get_intrusion_mask so pre-aspirated sibilants ('hs') and tokens starting with 'h' are never marked as eligible for intrusive laryngeals, preventing spurious 'hhs' and 'hshs' clusters. Update tests and verify impact on dataset error taxonomy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Update get_intrusion_mask in phonotactics.py to disallow intrusion before 'hs' and tokens starting with 'h'
- [x] #2 Ensure phonotactic constraints strictly prohibit *HH and *hhs clusters
- [x] #3 Verify all phonotactics and alignment unit tests pass with pytest
- [x] #4 Rescore dataset using cached LPZ to evaluate reduction in spurious h errors
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect phonotactics.py get_intrusion_mask, is_valid_phonotactic_sequence, and token classifications.\n2. Update get_intrusion_mask so that:\n   - Tokens starting with 'h' (e.g. 'hs', 'hsk', 'hskw', 'hsl') or voiceless sonorants ('lh', 'nh', 'wh', 'yh') are never marked as eligible for pre-consonantal intrusion.\n   - Post-consonantal aspiration is disallowed if the consonant already ends with 'h' or is 'hs'.\n3. Update test_phonotactics.py to test 'hs', 'hsk', and 's' intrusion gating.\n4. Run pytest across the whole test suite.\n5. Run scripts/rescore_syllabary_dataset.py and analyze the reduction in error categories.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated get_intrusion_mask in transcription/alignment/phonotactics.py to strictly prohibit intrusive laryngeals before 'hs' and tokens starting/ending with 'h'. Added unit tests in test_phonotactics.py and verified all 274 tests pass. Rescored dataset: spurious 'h' errors plummeted by 82.6% (from 879 to 153), clean CER dropped to 5.10% across the dataset (test CER 5.58%), and overall clean WER dropped by 10.29% to 32.42%.
<!-- SECTION:FINAL_SUMMARY:END -->
