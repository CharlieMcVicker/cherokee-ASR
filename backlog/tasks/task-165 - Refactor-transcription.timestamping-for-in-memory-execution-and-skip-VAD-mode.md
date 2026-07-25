---
id: TASK-165
title: Refactor transcription.timestamping for in-memory execution and skip-VAD mode
status: To Do
assignee: []
created_date: '2026-07-25 19:55'
updated_date: '2026-07-25 19:57'
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
- [ ] #1 Expose callable Python function align_audio_segment() in timestamping module
- [ ] #2 Add skip_vad boolean flag to bypass VAD audio segmentation
- [ ] #3 Support optional debug export to disk while keeping align_cli.py as a wrapper
- [ ] #4 Decouple ASR emissions consumption from model inference in timestamping module
- [ ] #5 Expose lightweight CPU-based alignment functions for in-memory logits/emissions
- [ ] #6 Add skip_vad flag for pre-cut audio segments
<!-- AC:END -->
