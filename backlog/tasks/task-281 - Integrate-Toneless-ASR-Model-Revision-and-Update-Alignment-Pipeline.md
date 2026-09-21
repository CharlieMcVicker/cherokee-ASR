---
id: TASK-281
title: Integrate Toneless ASR Model Revision and Update Alignment Pipeline
status: To Do
assignee: []
created_date: '2026-09-10 20:04'
labels: []
dependencies: []
ordinal: 293000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pull a later Cherokee ASR model revision (or checkpoint) that does not output numeric tone digits, and adapt the emissions caching, reconciliation, and alignment pipelines to consume toneless hypotheses cleanly.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Identify and evaluate candidate toneless Cherokee ASR model revision from HuggingFace Hub
- [ ] #2 Update CachedASREmissionsExtractor cache keys for the toneless model checkpoint
- [ ] #3 Realign Bible chapters (Mark and Matthew) without numeric tone interference in reconciliation
- [ ] #4 Verify syllabary reconciliation formatting and whitespace integrity
<!-- AC:END -->
