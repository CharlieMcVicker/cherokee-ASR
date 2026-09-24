---
id: TASK-369.1
title: >-
  Extract language-agnostic ARPAbet types, G2P, MMS forced aligner, DP aligner,
  and trainer to Tier 1 core
status: To Do
assignee: []
created_date: '2026-09-24 14:52'
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
- [ ] #1 Relocate MMSForcedAligner and AlignedWordSpan to transcription.core.alignment.forced_aligner
- [ ] #2 Relocate WagnerFischerAligner to transcription.core.alignment.dp
- [ ] #3 Implement language-agnostic AcousticConfusionMatrix, G2PEngine, and ConfusionMatrixTrainer in transcription.core.codeswitching
- [ ] #4 Implement generic SyntheticTargetProjector in transcription.core.codeswitching supporting static dictionary lookup and dynamic Viterbi joint-ngram fallback
- [ ] #5 Add unit tests in transcription.core for all Tier 1 codeswitching primitives
<!-- AC:END -->
