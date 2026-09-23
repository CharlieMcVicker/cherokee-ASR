---
id: TASK-360.10
title: 'Phase 3B: Build code-switched dialogue alignment pipeline'
status: Done
assignee:
  - '@phase-3b-implementor'
created_date: '2026-09-21 20:33'
updated_date: '2026-09-22 19:52'
labels: []
dependencies: []
parent_task_id: TASK-360
ordinal: 389200
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Consolidate multi-speaker dialogue text parsing, speaker prefix handling, code-switched clitic projection, Silero VAD soft-masking, and 7-tier Praat TextGrid generation into transcription.pipelines.dialogue (replacing transcription/alignment/pipeline.py).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 DialogueAlignmentPipeline implemented in transcription.pipelines.dialogue orchestrating dialogue alignment
- [x] #2 Token discrimination isolates English tokens from Cherokee syncope/intrusion masks
- [x] #3 Assembles and exports 7-tier Praat TextGrids (Turn, Speaker, Syllabary, English, Reconciled, CTC Word, Phoneme) and JSON manifests
- [x] #4 Dialogue realignment tests in test_interview_realignment.py pass under new imports
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Implement DialogueAlignmentPipeline and helper procedures in transcription/pipelines/dialogue/pipeline.py with:
- Dialogue line and turn ingestion supporting speaker prefixes (e.g. 'Speaker 1: ...') and chunk timestamps/IDs.
- Token discrimination isolating English tokens from Cherokee syncope/intrusion masks using CodeSwitchedPreparer from transcription.cherokee.codeswitching.
- Model inference via CherokeeASRModel returning ModelOutput.
- Optional Silero VAD soft-masking using transcription.core.audio.masking.
- CTC segmentation alignment via transcription.core.alignment.ctc.CTCSegmentationAligner (and optional greedy DTW fallback for test/baseline parity).
- Phonetic and syllabary reconciliation using transcription.cherokee.enrichment.
- 7-tier Praat TextGrid generation (Turn, Speaker, Syllabary, English, Reconciled, CTC Word, Phoneme) + JSON manifest export using transcription.core.exporters.
- Turnkey runners align_dialogue, align_syllabary_ctc, align_syllabary_greedy.
2. Create transcription/pipelines/dialogue/__init__.py and export in transcription/pipelines/__init__.py.
3. Update transcription/alignment/pipeline.py as a backwards-compatible shim forwarding to transcription.pipelines.dialogue.pipeline.
4. Create comprehensive unit tests in transcription/pipelines/dialogue/tests/test_dialogue_pipeline.py covering all 7 tiers, token discrimination, speaker handling, and export formats.
5. Verify test_interview_realignment.py passes under both new and legacy imports.
6. Run full test suite and pyright static type checks; format with black.
7. Finalize task in Backlog with ACs checked and final summary.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented DialogueAlignmentPipeline in transcription.pipelines.dialogue. Consolidated multi-speaker dialogue text parsing, speaker prefix extraction, code-switched token discrimination isolating English tokens/clitics from Cherokee syncope/intrusion masks, Silero VAD soft-masking, CTC segmentation, phonetic syllabary reconciliation, and 7-tier Praat TextGrid (Turn, Speaker, Syllabary, English, Reconciled, CTC Word, Phoneme) + JSON manifest generation. Updated transcription/alignment/pipeline.py as backwards-compatible forwarding shim. Verified with 16 targeted tests in test_dialogue_pipeline.py, test_interview_realignment.py, and test_syllabary_runners.py, passing full 474-test suite, 0 pyright errors, and black formatting.
<!-- SECTION:FINAL_SUMMARY:END -->
