---
id: TASK-348.1
title: >-
  Decode cached chapter logits directly for verse greedy hypothesis in
  realign_bible.py
status: Done
assignee:
  - '@antigravity'
created_date: '2026-09-17 18:30'
updated_date: '2026-09-18 13:34'
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
- [x] #1 realign_bible.py slices the in-memory/cached chapter lpz matrix by verse start/end frame indices instead of invoking model.transcribe on exported WAV files
- [x] #2 greedy_hypothesis and greedy_confidence are computed via model.decode(verse_lpz) without re-running Wav2Vec2 forward inference
- [x] #3 Existing alignment record schema and output values remain identical or equivalent
- [x] #4 pytest and pyright pass with zero regressions
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. In realign_bible.py, obtain chapter lpz from ctc_aligner.get_logits_cached(audio_path)\n2. Compute start_frame and end_frame from chunk.start_sec and chunk.end_sec using ctc_aligner.index_duration\n3. Slice verse_lpz = lpz[start_frame:end_frame] and decode via ctc_aligner.model.decode(verse_lpz)\n4. Compute greedy_sentence and greedy_conf without neural forward passes\n5. Verify with pytest and pyright
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Replaced per-verse transcribe calls with in-memory slicing of cached chapter acoustic log-probabilities (lpz[start_f:end_f]) and direct decoding with model.decode(verse_lpz, compute_word_confidences=False). Verified with 277 passing pytest tests and 0 pyright errors.
<!-- SECTION:FINAL_SUMMARY:END -->
