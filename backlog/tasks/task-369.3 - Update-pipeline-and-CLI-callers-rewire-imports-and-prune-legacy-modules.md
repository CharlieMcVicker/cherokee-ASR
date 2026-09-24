---
id: TASK-369.3
title: 'Update pipeline and CLI callers, rewire imports, and prune legacy modules'
status: To Do
assignee: []
created_date: '2026-09-24 14:52'
labels: []
dependencies: []
parent_task_id: TASK-369
ordinal: 402300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update all call sites in transcription.pipelines.dialogue, transcription.pipelines.enrichment, and transcription.apps.cli to import from canonical Tier 1 and Tier 2 modules. Prune obsolete transcription/cherokee/arpabet files and verify zero regressions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Update dialogue and enrichment pipeline imports to use Tier 1 core and Tier 2 Cherokee factories
- [ ] #2 Update transcription.apps.cli and scripts to canonical import paths
- [ ] #3 Delete redundant files and obsolete test fixtures
- [ ] #4 Verify pytest (100% pass) and pyright transcription (0 errors)
<!-- AC:END -->
