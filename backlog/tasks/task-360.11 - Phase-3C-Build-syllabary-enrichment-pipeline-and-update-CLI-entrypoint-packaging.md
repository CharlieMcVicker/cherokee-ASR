---
id: TASK-360.11
title: >-
  Phase 3C: Build syllabary enrichment pipeline and update CLI entrypoint &
  packaging
status: To Do
assignee: []
created_date: '2026-09-21 20:33'
labels: []
dependencies: []
parent_task_id: TASK-360
ordinal: 389300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build EnrichmentPipeline in transcription.pipelines.enrichment, update transcription.apps.cli (align-cherokee), and update pyproject.toml entrypoint.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 EnrichmentPipeline implemented in transcription.pipelines.enrichment
- [ ] #2 align-cherokee CLI relocated to transcription.apps.cli delegating to domain pipelines
- [ ] #3 pyproject.toml [project.scripts] updated to align-cherokee = "transcription.apps.cli:main"
- [ ] #4 CLI unit tests in test_cli.py pass
<!-- AC:END -->
