---
id: TASK-282
title: Implement Bulk Batch ASR Inference Script to Pre-populate Emissions Cache
status: To Do
assignee: []
created_date: '2026-09-10 20:06'
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
- [ ] #1 Implement batch audio tensor collation and dynamic padding for multi-file ASR inference in CherokeeASRModel / CherokeeASRExtractor
- [ ] #2 Write batch outputs directly to runs/cache/emissions/ matching CachedASREmissionsExtractor JSON schema and cache key hash format
- [ ] #3 Provide CLI script scripts/batch_cache_emissions.py accepting audio directory and model revision flags
- [ ] #4 Verify CachedASREmissionsExtractor automatically detects and reuses bulk-generated cache entries without running model inference
<!-- AC:END -->
