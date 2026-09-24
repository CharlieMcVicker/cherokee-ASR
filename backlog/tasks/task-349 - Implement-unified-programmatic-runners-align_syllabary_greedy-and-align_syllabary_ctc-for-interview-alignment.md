---
id: TASK-349
title: >-
  Implement unified programmatic runners align_syllabary_greedy and
  align_syllabary_ctc for interview alignment
status: Done
assignee:
  - '@agent'
created_date: '2026-09-18 13:43'
updated_date: '2026-09-18 13:46'
labels: []
dependencies: []
ordinal: 368000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provide clean, direct Python interfaces to align Cherokee syllabary transcripts (with potential code-switching) against audio using both the Greedy+Reconciliation pipeline and the Guided CTC Segmentation pipeline, exporting Praat TextGrids and JSON manifests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement align_syllabary_greedy in transcription.alignment accepting audio, syllabary transcript, and exporting Praat TextGrid & JSON manifest
- [x] #2 Implement align_syllabary_ctc in transcription.alignment accepting audio, syllabary transcript, with syncope/intrusion-aware CTC segmentation and exporting Praat TextGrid & JSON manifest
- [x] #3 Support robust syllabary text ingestion (plain text string, list of strings, .txt path, or chunk list) with graceful handling of code-switched English words
- [x] #4 Ensure both functions output multi-tier Praat TextGrids with Ground Truth (Syllabary), Emitted (ASR), and Reconciled words
- [x] #5 Comprehensive unit tests verifying both runners with mock/real audio and mixed syllabary text
- [x] #6 Export functions in transcription.alignment __all__ and document usage in docs/alignment.md
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Implement robust transcript ingestion helper load_syllabary_transcript supporting plain text string, list of strings, .txt file path, or JSON chunk lists, converting Syllabary to TTH phonetic ground-truth while preserving original Syllabary tokens and handling code-switched English words.
2. Implement align_syllabary_greedy in transcription.alignment.pipeline (or transcription.alignment) executing greedy ASR emissions extraction, DTW / Needleman-Wunsch alignment, syllabary reconciliation, and multi-tier Praat TextGrid + JSON manifest export.
3. Implement align_syllabary_ctc in transcription.alignment executing forward acoustic logit extraction (with caching), syncope- and intrusion-aware CTC trellis segmentation, backtracked word emissions, multi-tier Praat TextGrid, and JSON manifest export.
4. Export align_syllabary_greedy and align_syllabary_ctc in transcription.alignment.__all__.
5. Add comprehensive unit tests covering both functions with mock and real audio fixtures and mixed Cherokee/English code-switched text.
6. Update docs/alignment.md to document the two functions and their comparative usage.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented unified syllabary interview alignment runners align_syllabary_greedy and align_syllabary_ctc along with load_syllabary_transcript in transcription.alignment. Both functions support raw syllabary strings, text files, JSON chunk lists, and mixed English code-switching, producing multi-tier Praat TextGrids and JSON manifests. Verified with 6 dedicated unit tests and full 145-test alignment suite passing with 0 pyright errors.
<!-- SECTION:FINAL_SUMMARY:END -->
