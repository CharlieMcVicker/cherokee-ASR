---
id: TASK-348.1
title: >-
  Decode cached chapter logits directly for verse greedy hypothesis in
  realign_bible.py
status: To Do
assignee: []
created_date: '2026-09-17 18:30'
labels:
  - alignment
  - performance
  - realign_bible
dependencies: []
parent_task_id: TASK-348
priority: high
ordinal: 365000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In scripts/realign_bible.py, replace sequential per-verse CherokeeASRModel.transcribe(split_out_path) calls with direct decoding from the cached chapter acoustic log-probability matrix (lpz[start_frame:end_frame]) via model.decode(). This eliminates 40-80 redundant PyTorch neural network forward passes per chapter.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 realign_bible.py slices the in-memory/cached chapter lpz matrix by verse start/end frame indices instead of invoking model.transcribe on exported WAV files
- [ ] #2 greedy_hypothesis and greedy_confidence are computed via model.decode(verse_lpz) without re-running Wav2Vec2 forward inference
- [ ] #3 Existing alignment record schema and output values remain identical or equivalent
- [ ] #4 pytest and pyright pass with zero regressions
<!-- AC:END -->
