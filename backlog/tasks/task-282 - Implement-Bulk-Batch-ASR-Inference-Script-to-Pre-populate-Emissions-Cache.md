---
id: TASK-282
title: Implement Bulk Batch ASR Inference Script to Pre-populate Emissions Cache
status: Done
assignee: []
created_date: '2026-09-10 20:06'
updated_date: '2026-09-10 20:28'
labels: []
dependencies: []
ordinal: 294000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build a high-throughput batch inference pipeline that processes entire collections of audio recordings (e.g., all chapters of Mark and Matthew) in batched GPU passes with dynamic padding, writing the resulting TokenEmission sequences directly into the CachedASREmissionsExtractor disk cache structure (runs/cache/emissions/) so downstream alignment runs instantaneously without on-demand ASR overhead.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement batch audio tensor collation and dynamic padding for multi-file ASR inference in CherokeeASRModel / CherokeeASRExtractor
- [x] #2 Write batch outputs directly to runs/cache/emissions/ matching CachedASREmissionsExtractor JSON schema and cache key hash format
- [x] #3 Provide CLI script scripts/batch_cache_emissions.py accepting audio directory and model revision flags
- [x] #4 Verify CachedASREmissionsExtractor automatically detects and reuses bulk-generated cache entries without running model inference
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add get_logits_batch to CherokeeASRModel with dynamic padding and MPS/CPU/CUDA fallback.
2. Add extract_batch to CherokeeASRExtractor for multi-input chunk collation.
3. Add populate_cache (and infer_bulk alias) to CachedASREmissionsExtractor to check cache hits and batch infer uncached items.
4. Create scripts/batch_cache_emissions.py CLI utility for bulk cache pre-population.
5. Add comprehensive unit tests in test_extractors.py.
6. Run bulk cache pre-population on Bible audio files using length-only colon model.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented bulk batch ASR emissions caching pipeline. Added get_logits_batch to CherokeeASRModel and extract_batch to CherokeeASRExtractor for collation and dynamic padding. Added populate_cache (and infer_bulk alias) to CachedASREmissionsExtractor to check disk cache hits and batch infer uncached items. Created CLI driver scripts/batch_cache_emissions.py. Populated runs/cache/emissions/ for all 52 New Testament audio files (28,316 token emissions) using toneless model charliemcvicker/length-only-20260704-155307-asr-cherokee-colon:76e62140955f4738abdab345ea34068b02d8d2a2. All unit tests pass.
<!-- SECTION:FINAL_SUMMARY:END -->
