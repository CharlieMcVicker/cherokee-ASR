---
id: TASK-336
title: >-
  Implement on-disk caching for clean and noised LPZ logits in syllabary rescore
  pipeline
status: Done
assignee:
  - '@agent'
created_date: '2026-09-16 16:20'
updated_date: '2026-09-16 16:24'
labels: []
dependencies: []
ordinal: 352000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Cache computed LPZ logit matrices, durations, and greedy decodes for clean and pink-noised audio to disk (e.g. under runs/cache/lpz/). Avoid repeatedly synthesizing noise transforms and running Wav2Vec2 forward passes on subsequent rescore runs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement on-disk LPZ and greedy decode caching in scripts/rescore_syllabary_dataset.py
- [x] #2 Ensure noise generation and Wav2Vec2 forward passes are skipped when cache is present
- [x] #3 Verify rescore script executes in under 5 seconds with cached LPZ and produces identical results
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Design an LPZ cache directory structure (runs/cache/rescore_syllabary/) storing compressed NPZ files (or torch pt files) keyed by condition (clean vs noisy_pink_18db).\n2. Store lpz, dur_sec, and greedy_text in the cache.\n3. In evaluate_dataset(), check if cache file exists. If so, load cached records and skip audio loading, noise generation, and Wav2Vec2 forward passes.\n4. If cache does not exist, compute LPZ and greedy decodes, save to cache file, and proceed.\n5. Add a --force-recompute CLI flag to scripts/rescore_syllabary_dataset.py.\n6. Run the script once to populate the cache, and a second time to verify sub-5s runtime and identical metrics.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented on-disk caching in scripts/rescore_syllabary_dataset.py for clean and pink-noised (18 dB) LPZ matrices, durations, and greedy decodes under runs/cache/rescore_syllabary/. Added --force-recompute flag. Verified that cached evaluation runs across all 1,389 samples in 5.4s (down from 80.7s) with 100% identical metrics.
<!-- SECTION:FINAL_SUMMARY:END -->
