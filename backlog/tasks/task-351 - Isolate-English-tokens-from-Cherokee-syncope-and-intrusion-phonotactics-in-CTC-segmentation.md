---
id: TASK-351
title: >-
  Isolate English tokens from Cherokee syncope and intrusion phonotactics in CTC
  segmentation
status: Done
assignee:
  - '@antigravity'
created_date: '2026-09-18 18:35'
updated_date: '2026-09-18 18:42'
labels: []
dependencies: []
ordinal: 377000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure that English loanwords and stems in code-switched transcripts are not marked with Cherokee vowel syncope or intrusive glottal/aspiration masks during CTC Trellis preparation, while native Cherokee Syllabary retains full phonotactic awareness.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CodeSwitchedToken and CodeSwitchedLineResult provide per-token syncope and intrusion phonotactic masks (all False for pure English, segmented for compound clitics, Cherokee phonotactics for pure Syllabary)
- [x] #2 prepare_cherokee_text consumes token/character masks to populate is_syncope_token and is_intrusive_site without evaluating English words as Cherokee
- [x] #3 CTCSegmentationAligner and align_syllabary_ctc pass code-switched token metadata through to prepare_cherokee_text when code_switched=True
- [x] #4 Comprehensive unit and integration tests pass verifying zero-syncope on English words and correct phonotactics on Cherokee words
- [x] #5 gs_mm CTC alignment executes cleanly with multi-tier Praat TextGrid and JSON manifest output
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add syncope_mask and intrusion_mask to CodeSwitchedToken in codeswitched_preparer.py (zeroed for English stems, phonotactic for Syllabary).
2. Update prepare_cherokee_text in phonotactics.py to accept token phonotactic masks or tokens and respect provenance without evaluating English loanwords as Cherokee morphemes.
3. Wire token phonotactic metadata from source_lookup through CTCSegmentationAligner and align_syllabary_ctc.
4. Add comprehensive unit and integration tests in test_codeswitched_preparer.py, test_phonotactics.py, and test_ctc_aligner.py.
5. Create and run CTC alignment script for gs_mm interview in saving-the-voices and verify Praat TextGrid and JSON manifest.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented token-level phonotactic provenance isolation for code-switched alignment. CodeSwitchedToken now computes per-token syncope and intrusion masks (strictly zeroed for English loanwords and stems, segmented for compound clitics, and phonotactically aware for Cherokee Syllabary). prepare_cherokee_text and CTCSegmentationAligner forward these masks into CtcSegmentationParameters. Verified with 243 passing unit and integration tests and successfully realigned the 56-minute gs_mm interview with CTC segmentation into saving-the-voices/output_ctc/.
<!-- SECTION:FINAL_SUMMARY:END -->
