---
id: TASK-350.3
title: Implement Batch Cherokee ASR Inference with Per-Model Emissions Cache
status: Done
assignee:
  - '@supervisor'
created_date: '2026-09-18 14:44'
updated_date: '2026-09-18 15:34'
labels:
  - alignment
  - inference
  - cache
  - asr
dependencies:
  - TASK-350.2
parent_task_id: TASK-350
priority: high
type: feature
ordinal: 372000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run batched inference of the 5,000 word clips through CherokeeASRModel on Apple Silicon (mps/CPU). Optimize execution by bucketing audio clips by duration to eliminate wasted padding. Extract greedy TTH tokens, per-token confidences, and Top-3 beam hypotheses with sequence scores. Persist results to a per-model cache data/arpabet_alignment/cache/{model_id}_emissions.json so forward passes execute only once per model checkpoint.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement duration-sorted batching for CherokeeASRModel forward pass on Apple Silicon (mps/CPU)
- [x] #2 Extract greedy TTH tokens and per-character confidence scores for each word clip
- [x] #3 Extract Top-3 beam search hypotheses with sequence logprob scores
- [x] #4 Persist and load cache at data/arpabet_alignment/cache/{model_id}_emissions.json, bypassing forward pass on subsequent runs
- [x] #5 Provide unit tests for batch runner, cache hit/miss logic, and emissions parsing
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Build transcription/alignment/arpabet/inference.py with duration-sorted batching (to minimize padding overhead on MPS/CPU) and Top-K beam decoding.
2. Implement Cherokee digraph grouping using tokenize_phonemes from phonotactics.py so multi-character phonemes (hs, th, kh, ts, tsh, tl, tlh, lh, etc.) are treated as atomic Cherokee tokens with merged confidences.
3. Implement per-model emissions cache persistence at data/arpabet_alignment/cache/{sanitized_model_id}_emissions.json adhering to InferenceCacheManifest.
4. Add automated unit tests in transcription/alignment/tests/test_arpabet_inference.py covering duration sorting, digraph grouping, beam hypothesis extraction, and cache hit/miss persistence.
5. Execute forward-pass inference over the 5,000 clips in data/arpabet_alignment/words/ using the best Cherokee ASR model on MPS to produce the cached emissions manifest.
6. Verify test suite with pytest and static analysis with pyright.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented batch inference runner in transcription/alignment/arpabet/inference.py with duration-sorted bucketing, digraph grouping, CTC prefix beam search Top-3 hypotheses, and durable InferenceCacheManifest caching. Executed over 5,000 word clips on Apple Silicon MPS in ~35s, populating data/arpabet_alignment/cache/charliemcvicker_length-only-20260805-231513-asr-bible_57a318660fd6dceab4b2747346a7aeda2922e294_emissions.json. All 14 tests in test_arpabet_inference.py and 194 alignment tests pass. Zero Pyright errors.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented batch Cherokee ASR inference runner with per-model emissions caching (TASK-350.3). Created transcription/alignment/arpabet/inference.py featuring: (1) sanitize_model_id for clean filesystem cache filenames; (2) bucket_by_duration for padding-free batched forward passes; (3) group_digraphs_with_confidences supporting all 11 canonical Cherokee digraphs/trigraphs (hs, th, kh, ts, tsh, tl, tlh, lh, nh, wh, yh) and decomposing sibilant clusters (hsk -> hs + k) with arithmetic mean confidence averaging; (4) ctc_prefix_beam_search and extract_top_k_hypotheses returning Top-3 beam search hypotheses with sequence logprobs; (5) run_model_inference_on_manifest with automated cache hit bypass and persistence to data/arpabet_alignment/cache/{sanitized_model_id}_emissions.json using InferenceCacheManifest. Executed forward pass across all 5,000 LibriSpeech word clips on Apple Silicon MPS in 35 seconds, populating the 15MB cache manifest. Verified with 14 unit tests in test_arpabet_inference.py, 194 passing alignment tests, and 0 Pyright errors.
<!-- SECTION:FINAL_SUMMARY:END -->
