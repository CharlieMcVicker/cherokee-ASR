---
id: TASK-360.5
title: 'Phase 1B: Core audio chunking, energy profiling & VAD logit soft-masking'
status: In Progress
assignee:
  - '@phase-1b-implementor'
created_date: '2026-09-21 20:32'
updated_date: '2026-09-22 14:48'
labels: []
dependencies: []
parent_task_id: TASK-360
ordinal: 387200
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extract language-agnostic audio segmentation, energy profiling, and Silero VAD logit soft-masking into transcription.core.audio, eliminating hardcoded Cherokee paths or settings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 segment_long_audio, get_energy_profile, and AudioChunk reside in transcription.core.audio.segment
- [ ] #2 Silero VAD logit masking functions (apply_vad_soft_masking, mask_non_speech_logits, extract_vad_intervals) reside in transcription.core.audio.masking
- [ ] #3 Audio unit tests (test_audio_segmenter.py, test_non_speech_masking.py) pass under new imports and pyright reports 0 errors
<!-- AC:END -->
