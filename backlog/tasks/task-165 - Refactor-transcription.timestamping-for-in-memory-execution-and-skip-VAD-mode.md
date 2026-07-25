---
id: TASK-165
title: Refactor transcription.timestamping for in-memory execution and skip-VAD mode
status: Done
assignee:
  - '@agent'
created_date: '2026-07-25 19:55'
updated_date: '2026-07-25 20:00'
labels: []
dependencies: []
ordinal: 161000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Refactor transcription.timestamping to decouple ASR logits/emissions consumption and alignment logic from model execution. Expose lightweight CPU alignment functions (e.g. align_emissions_to_text) so batched GPU model outputs can be processed in memory without running VAD or file IO.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Expose callable Python function align_audio_segment() in timestamping module
- [x] #2 Add skip_vad boolean flag to bypass VAD audio segmentation
- [x] #3 Support optional debug export to disk while keeping align_cli.py as a wrapper
- [x] #4 Decouple ASR emissions consumption from model inference in timestamping module
- [x] #5 Expose lightweight CPU-based alignment functions for in-memory logits/emissions
- [x] #6 Add skip_vad flag for pre-cut audio segments
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect transcription/timestamping/aligner.py, align_cli.py, and exporter.py.\n2. Extract pure CPU alignment functions decoupling ASR logits/emissions consumption from PyTorch model loading and audio decoding.\n3. Add skip_vad parameter to allow running alignment on pre-cut audio segments.\n4. Ensure align_cli.py functions as a thin CLI wrapper around the refactored core.\n5. Run unit tests in transcription/timestamping to ensure no regressions.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Refactored transcription/timestamping module to decouple PyTorch ASR model execution and audio decoding from DTW alignment logic. Introduced lightweight CPU functions align_emissions_to_text and align_audio_segment for in-memory token/text alignment, added skip_vad boolean parameter/flag to bypass VAD audio segmentation when processing pre-cut audio clips, updated align_cli.py to act as a thin wrapper, and added unit tests covering all new in-memory and skip_vad workflows.
<!-- SECTION:FINAL_SUMMARY:END -->
