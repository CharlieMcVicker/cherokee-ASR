---
id: TASK-200
title: Expose in-memory PCM inference entrypoint in transcription.inference.infer
status: Done
assignee:
  - '@agent'
created_date: '2026-08-12 21:16'
updated_date: '2026-08-12 21:20'
labels: []
dependencies: []
ordinal: 196000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose load_asr_model and infer_pcm_array in transcription.inference.infer so downstream applications can pass raw 1D float32 PCM numpy/bytes arrays directly from VAD buffers without needing intermediate files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 load_asr_model returns cached model and processor
- [x] #2 infer_pcm_array accepts 1D float32 numpy array or list/bytes and returns text and confidence
- [x] #3 Pyright and tests pass
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Exposed load_asr_model and infer_pcm_array in transcription/inference/infer.py with full Pyright type compliance and in-memory PCM float32 array processing.
<!-- SECTION:FINAL_SUMMARY:END -->
