---
id: TASK-363
title: Prune legacy transcription/audio and transcription/new_testament shims
status: To Do
assignee: []
created_date: '2026-09-23 15:55'
labels: []
dependencies: []
ordinal: 393300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
transcription/audio/non_speech_masking.py and transcription/new_testament/ (pipeline.py) exist solely as forwarding shims to transcription.core.audio and transcription.pipelines.scripture. Callers and test fixtures should be updated to point directly to Tier 1 core.audio and Tier 3 pipelines.scripture, and the legacy directories removed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 transcription/new_testament/ package deleted and tests updated to test_scripture_pipeline.py
- [ ] #2 transcription/audio/ forwarding shims pruned in favor of transcription.core.audio
- [ ] #3 All tests pass and pyright reports 0 errors
<!-- AC:END -->
