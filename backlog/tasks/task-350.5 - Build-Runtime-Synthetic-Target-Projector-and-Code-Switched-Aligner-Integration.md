---
id: TASK-350.5
title: Build Runtime Synthetic Target Projector and Code-Switched Aligner Integration
status: Done
assignee:
  - '@supervisor'
created_date: '2026-09-18 14:44'
updated_date: '2026-09-18 15:54'
labels:
  - alignment
  - projector
  - runtime
  - integration
dependencies:
  - TASK-350.4
parent_task_id: TASK-350
priority: high
type: feature
ordinal: 374000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build runtime projector translating English words in code-switched transcripts into synthetic Cherokee phonetic targets using g2p_en and the calibrated acoustic confusion matrix. Precompute an O(1) static dictionary for frequent English loanwords. Integrate projector into transcription.alignment.ingestion and prepare_alignment_input so English words are projected and aligned by CTCSegmentationAligner against Cherokee ASR emissions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement pure functional projector mapping English text -> ARPAbet -> synthetic Cherokee TTH target
- [x] #2 Generate and support static memoized dictionary for top frequent English words for O(1) runtime lookup
- [x] #3 Integrate into transcription.alignment.ingestion and prepare_alignment_input to support mixed Cherokee/English transcripts
- [x] #4 Add unit and integration tests verifying alignment accuracy on code-switched Cherokee/English sentences
- [x] #5 Update docs/alignment.md and AGENTS.md documenting code-switched alignment architecture and CLI options
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Build transcription/alignment/arpabet/projector.py implementing SyntheticTargetProjectorProtocol with g2p_en + AcousticConfusionMatrix argmax projection.
2. Build static memoized dictionary generator and O(1) lookup cache for frequent English words at data/arpabet_alignment/dictionaries/english_loanwords_tth.json.
3. Integrate projector into transcription.alignment.ingestion (prepare_alignment_input and load_syllabary_transcript) to project English words into synthetic Cherokee TTH.
4. Add comprehensive unit and integration tests in transcription/alignment/tests/test_arpabet_projector.py verifying projection accuracy and end-to-end alignment preparation.
5. Update docs/alignment.md and AGENTS.md documenting code-switched alignment architecture and CLI parameters.
6. Run pytest and pyright to verify zero regressions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented SyntheticTargetProjector and code-switched alignment integration. Generated static dictionary with 3,662 English loanwords at data/arpabet_alignment/dictionaries/english_loanwords_tth.json. Integrated into prepare_alignment_input, load_syllabary_transcript, load_interview_transcript, align_syllabary_greedy, align_syllabary_ctc, and cli.py. Verified with 11 dedicated tests in test_arpabet_projector.py, 216 alignment tests, 354 total repository tests, and zero pyright errors.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented runtime SyntheticTargetProjector complying with SyntheticTargetProjectorProtocol for pure functional translation of English text and ARPAbet sequences into canonical Cherokee TTH phonetics. Generated static memoized dictionary for 3,662+ English loanwords at data/arpabet_alignment/dictionaries/english_loanwords_tth.json for O(1) runtime lookup, backed by dynamic G2P and calibrated confusion matrix argmax mapping. Integrated code-switched alignment into prepare_alignment_input, load_syllabary_transcript, load_interview_transcript, align_syllabary_greedy, align_syllabary_ctc, and align-cherokee CLI (--code-switched, --transcript). Updated docs/alignment.md and AGENTS.md. Verified with 11 unit/integration tests in test_arpabet_projector.py, 216 alignment tests, and all 354 repository tests with zero pyright type check errors.
<!-- SECTION:FINAL_SUMMARY:END -->
