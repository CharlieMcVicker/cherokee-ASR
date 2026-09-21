---
id: TASK-350.6
title: >-
  Build Code-Switching Ground Truth Preparer for Mixed Syllabary and English
  Text
status: Done
assignee:
  - '@supervisor'
created_date: '2026-09-18 14:49'
updated_date: '2026-09-18 16:04'
labels:
  - alignment
  - code-switching
  - syllabary
  - normalization
dependencies:
  - TASK-350.5
parent_task_id: TASK-350
priority: high
type: feature
ordinal: 375000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement create_groundtruth_for_code_switched_syllabary to prepare mixed Cherokee Syllabary and English code-switched text for alignment without double conversion. Discriminates tokens into pure Cherokee Syllabary, English Latin words, and compound clitics (e.g. JayᎢ -> Jay + Ꭲ). Converts Cherokee Syllabary directly to canonical TTH phonetics while passing English tokens strictly through the ARPAbet synthetic target projector, avoiding Cherokee DG-to-TTH consonant mutation on English words.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement create_groundtruth_for_code_switched_syllabary with script-level token discrimination
- [x] #2 Prevent double-conversion by isolating English tokens from Cherokee DG-to-TTH consonant mutation
- [x] #3 Correctly segment and project compound tokens with English stems and attached Syllabary clitics (e.g. JayᎢ)
- [x] #4 Emit unified canonical TTH target sequence while preserving multi-tier word metadata (Syllabary, English, Reconciled)
- [x] #5 Add unit tests in tests/alignment/test_codeswitched_preparer.py covering sample sentences from saving-the-voices/gs_mm.txt
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Build transcription/alignment/arpabet/codeswitched_preparer.py with CodeSwitchedToken, TokenType, and CodeSwitchedLineResult immutable types.
2. Implement create_groundtruth_for_code_switched_syllabary with script-level token discrimination (pure Cherokee Syllabary, pure English, and compound Latin stem + Syllabary clitic like JayᎢ).
3. Ensure zero double-conversion: Syllabary converts directly to TTH; English tokens project via SyntheticTargetProjector without passing through Cherokee DG consonant mutations.
4. Wire into transcription.alignment.ingestion load_syllabary_transcript and load_interview_transcript when code_switched=True.
5. Create comprehensive unit tests in transcription/alignment/tests/test_codeswitched_preparer.py covering sample sentences directly from saving-the-voices/gs_mm.txt.
6. Verify test suite with pytest and static type analysis with pyright.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented codeswitched_preparer.py with immutable models, token discrimination, compound clitic segmentation, and zero double-conversion. Integrated into ingestion pipeline. 22 dedicated unit tests and 238 full alignment tests passing with 0 pyright errors.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented create_groundtruth_for_code_switched_syllabary in transcription.alignment.arpabet.codeswitched_preparer to prepare mixed Cherokee Syllabary and English code-switched transcripts for ASR alignment with zero double conversion. Discriminates tokens into pure Syllabary, English, compound clitics (e.g. JayᎢ, WellingᏛ), and punctuation. Correctly segments compound clitics, converts Syllabary directly to canonical TTH, and projects English stems strictly via SyntheticTargetProjector. Wired into load_syllabary_transcript and load_interview_transcript. Verified with 22 dedicated unit tests in test_codeswitched_preparer.py and full 238-test alignment suite passing with 0 pyright errors.
<!-- SECTION:FINAL_SUMMARY:END -->
