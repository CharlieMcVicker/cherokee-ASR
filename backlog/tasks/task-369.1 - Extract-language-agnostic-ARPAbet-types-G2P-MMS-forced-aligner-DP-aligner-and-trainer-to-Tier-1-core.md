---
id: TASK-369.1
title: >-
  Extract language-agnostic ARPAbet types, G2P, MMS forced aligner, DP aligner,
  and trainer to Tier 1 core
status: Done
assignee:
  - '@subagent'
created_date: '2026-09-24 14:52'
updated_date: '2026-09-24 14:59'
labels: []
dependencies: []
parent_task_id: TASK-369
ordinal: 400300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Relocate MMSForcedAligner, WagnerFischerAligner, ctc_prefix_beam_search, bucket_by_duration, G2P lookup, AcousticConfusionMatrix, and ConfusionMatrixTrainer to transcription.core with zero Cherokee-specific assumptions. Build generic SyntheticTargetProjector with Viterbi joint-ngram projection.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Relocate MMSForcedAligner and AlignedWordSpan to transcription.core.alignment.forced_aligner
- [x] #2 Relocate WagnerFischerAligner to transcription.core.alignment.dp
- [x] #3 Implement language-agnostic AcousticConfusionMatrix, G2PEngine, and ConfusionMatrixTrainer in transcription.core.codeswitching
- [x] #4 Implement generic SyntheticTargetProjector in transcription.core.codeswitching supporting static dictionary lookup and dynamic Viterbi joint-ngram fallback
- [x] #5 Add unit tests in transcription.core for all Tier 1 codeswitching primitives
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extract MMSForcedAligner to transcription/core/alignment/forced_aligner.py and WagnerFischerAligner to transcription/core/alignment/dp.py
2. Implement transcription/core/codeswitching/ (types.py, g2p.py, matrix.py, trainer.py, projector.py)
3. Move core unit tests from transcription/alignment/tests/ to transcription/core/codeswitching/tests/ and transcription/core/alignment/tests/
4. Verify tests and typechecking
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Relocated MMSForcedAligner and AlignedWordSpan to transcription.core.alignment.forced_aligner. Relocated WagnerFischerAligner to transcription.core.alignment.dp. Implemented language-agnostic types, G2PEngine, AcousticConfusionMatrix, ConfusionMatrixTrainer, and SyntheticTargetProjector in transcription.core.codeswitching. Verified all 415 unit tests pass and pyright reports 0 errors.
<!-- SECTION:FINAL_SUMMARY:END -->
