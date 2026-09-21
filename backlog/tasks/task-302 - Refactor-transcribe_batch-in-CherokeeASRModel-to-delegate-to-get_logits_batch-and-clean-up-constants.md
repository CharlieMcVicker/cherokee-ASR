---
id: TASK-302
title: >-
  Refactor transcribe_batch in CherokeeASRModel to delegate to get_logits_batch
  and clean up constants
status: Done
assignee: []
created_date: '2026-09-12 19:36'
updated_date: '2026-09-12 19:40'
labels: []
dependencies: []
ordinal: 318000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Streamline CherokeeASRModel.transcribe_batch to delegate directly to get_logits_batch, eliminating redundant code, and define FRAME_DURATION_SEC constant.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Refactor transcribe_batch to reuse get_logits_batch and decode results cleanly
- [x] #2 Define and use FRAME_DURATION_SEC constant in sliding-window inference
<!-- AC:END -->
