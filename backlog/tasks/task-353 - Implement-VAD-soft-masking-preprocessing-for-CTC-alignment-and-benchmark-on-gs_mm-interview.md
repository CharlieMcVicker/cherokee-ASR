---
id: TASK-353
title: >-
  Implement VAD soft-masking preprocessing for CTC alignment and benchmark on
  gs_mm interview
status: Done
assignee:
  - '@agent-k'
created_date: '2026-09-21 16:19'
updated_date: '2026-09-21 16:24'
labels: []
dependencies: []
ordinal: 379000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Surrounding laughter and non-speech events cause CTC segmentation to drag word characters across non-speech frames, inflating word durations and corrupting emitted word transcriptions. Implement Silero VAD soft-masking logit preprocessing to blend non-speech frames towards blank emissions and evaluate against scripts/realign_gs_mm_ctc.py.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement mask_non_speech_logits with Silero VAD continuous speech probability soft-blending
- [x] #2 Add optional non-speech soft-masking configuration to CTCAlignerConfig and CTCSegmentationAligner
- [x] #3 Run realign_gs_mm_ctc with VAD soft-masking enabled and output comparative TextGrid/manifest
- [x] #4 Unit tests verify mask_non_speech_logits behavior and CTCSegmentationAligner integration
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Silero VAD soft-masking in transcription/audio/non_speech_masking.py with convex posterior blending. Wired configuration into CTCAlignerConfig and CTCSegmentationAligner. Ran pytest suite (244 tests passing) and executed scripts/realign_gs_mm_ctc.py to generate updated gs_mm_ctc.TextGrid and gs_mm_ctc_manifest.json.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Silero VAD continuous soft-masking preprocessing for CTC alignment log-probabilities, parameterized through CTCAlignerConfig(enable_vad_soft_masking=True), verified across full unit test suite, and executed realign_gs_mm_ctc.py generating fresh TextGrid and JSON manifest.
<!-- SECTION:FINAL_SUMMARY:END -->
