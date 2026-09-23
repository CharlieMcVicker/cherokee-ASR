---
id: TASK-361
title: Prune ASREmissionsExtractor hierarchy and extractors.py
status: To Do
assignee: []
created_date: '2026-09-23 15:55'
labels: []
dependencies: []
ordinal: 391300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Legacy ASREmissionsExtractor hierarchy, CherokeeASRExtractor, CachedASREmissionsExtractor, and extractors.py remain in transcription/alignment/ as backward-compatibility shims despite ModelOutput and on-model .npz caching being established in Tier 1. Remove extractors.py, its unit tests, and update callers (scripture pipeline, apps/cli.py, batch_cache_emissions.py) to use ModelOutput directly.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 extractors.py and test_extractors.py deleted
- [ ] #2 transcription.apps.cli and transcription.pipelines.scripture updated to use ModelOutput
- [ ] #3 All tests pass and pyright reports 0 errors
<!-- AC:END -->
