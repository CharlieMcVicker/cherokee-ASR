---
id: TASK-366
title: Prune legacy transcription/utils compatibility shims
status: To Do
assignee: []
created_date: '2026-09-23 15:55'
labels: []
dependencies: []
ordinal: 396300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
transcription/utils/ (orthography.py, syllabary_map.py, tone_normalization.py) forward to transcription.cherokee.orthography. Call sites across training, evaluation, and tests should be updated to import from transcription.cherokee.orthography directly and the forwarding wrappers deleted per Clean Break Protocol.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 transcription/utils/ orthography and syllabary compatibility shims deleted
- [ ] #2 Call sites in training/, evaluation/, and utils updated to transcription.cherokee.orthography
- [ ] #3 All tests pass and pyright reports 0 errors
<!-- AC:END -->
