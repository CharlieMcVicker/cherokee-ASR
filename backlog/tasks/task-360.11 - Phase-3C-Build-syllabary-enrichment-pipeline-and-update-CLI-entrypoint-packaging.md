---
id: TASK-360.11
title: >-
  Phase 3C: Build syllabary enrichment pipeline and update CLI entrypoint &
  packaging
status: Done
assignee:
  - '@phase-3c-implementor'
created_date: '2026-09-21 20:33'
updated_date: '2026-09-22 20:21'
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
- [x] #1 EnrichmentPipeline implemented in transcription.pipelines.enrichment
- [x] #2 align-cherokee CLI relocated to transcription.apps.cli delegating to domain pipelines
- [x] #3 pyproject.toml [project.scripts] updated to align-cherokee = "transcription.apps.cli:main"
- [x] #4 CLI unit tests in test_cli.py pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Implement EnrichmentPipeline in transcription/pipelines/enrichment/pipeline.py and re-export in transcription/pipelines/enrichment/__init__.py and transcription/pipelines/__init__.py.
2. Relocate align-cherokee CLI to transcription/apps/cli.py delegating cleanly to domain pipelines (ScripturePipeline, DialogueAlignmentPipeline, EnrichmentPipeline) and create transcription/apps/__init__.py.
3. Update pyproject.toml [project.scripts] to point align-cherokee to transcription.apps.cli:main and keep transcription/alignment/cli.py as backwards-compatible forwarding shim.
4. Update and add unit tests in transcription/alignment/tests/test_cli.py and transcription/pipelines/enrichment/tests/ to verify CLI delegation and enrichment pipeline functionality.
5. Verify with pytest, pyright, and black.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented EnrichmentPipeline in transcription.pipelines.enrichment, relocated align-cherokee CLI to transcription.apps.cli delegating directly to domain pipelines (ScripturePipeline, DialogueAlignmentPipeline, EnrichmentPipeline), updated pyproject.toml [project.scripts] to transcription.apps.cli:main while preserving backward-compatible shim in transcription.alignment.cli, and validated comprehensive CLI and enrichment unit tests with 100% test pass rate and zero pyright errors.
<!-- SECTION:FINAL_SUMMARY:END -->
