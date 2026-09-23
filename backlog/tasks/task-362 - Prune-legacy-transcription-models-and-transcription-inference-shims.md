---
id: TASK-362
title: Prune legacy transcription/models and transcription/inference shims
status: To Do
assignee: []
created_date: '2026-09-23 15:55'
labels: []
dependencies: []
ordinal: 392300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
transcription/models/asr_model.py, WordConfidence, and transcription/inference/ shims were preserved for backward compatibility. Call sites across tests, training, and evaluation should import directly from transcription.cherokee.models.loader, transcription.core.models, and transcription.core.audio. Delete the legacy modules and update callers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 transcription/models/ directory and legacy shims deleted
- [ ] #2 Call sites in training/ and evaluation/ updated to transcription.cherokee.models and transcription.core
- [ ] #3 All tests pass and pyright reports 0 errors
<!-- AC:END -->
