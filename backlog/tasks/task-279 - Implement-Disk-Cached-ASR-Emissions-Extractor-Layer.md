---
id: TASK-279
title: Implement Disk-Cached ASR Emissions Extractor Layer
status: Done
assignee:
  - '@subagent-279'
created_date: '2026-09-10 17:52'
updated_date: '2026-09-10 17:55'
labels:
  - alignment
  - cache
  - emissions
dependencies: []
ordinal: 291000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a simple caching wrapper layer (CachedASREmissionsExtractor) conforming to ASREmissionsExtractor protocol that caches token emissions outputs to disk by model identifier and input audio path/hash. This must be implemented prior to TASK-276 and used during Bible realignment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement CachedASREmissionsExtractor conforming to ASREmissionsExtractor protocol with disk persistence
- [x] #2 Generate deterministic cache keys based on model identity (name/revision) and audio file path or audio content hash
- [x] #3 Add unit tests verifying cache hits bypass model inference and write to disk on misses
- [x] #4 Ensure full backwards compatibility and zero signature breaking changes
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented `CachedASREmissionsExtractor` conforming to `ASREmissionsExtractor` protocol in `transcription/alignment/extractors.py`. Provides disk caching with deterministic cache key generation based on model/prefix identity and audio file paths (including mtime/size invalidation) or audio content hash (for AudioSegment, ndarray, raw bytes). Exposed `CachedASREmissionsExtractor` in `transcription/alignment/__init__.py` and added full unit test coverage in `transcription/alignment/tests/test_extractors.py`.
<!-- SECTION:FINAL_SUMMARY:END -->
