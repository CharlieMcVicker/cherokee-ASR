---
id: TASK-369.3
title: 'Update pipeline and CLI callers, rewire imports, and prune legacy modules'
status: Done
assignee:
  - '@supervisor'
created_date: '2026-09-24 14:52'
updated_date: '2026-09-24 15:21'
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
- [x] #1 Update dialogue and enrichment pipeline imports to use Tier 1 core and Tier 2 Cherokee factories
- [x] #2 Update transcription.apps.cli and scripts to canonical import paths
- [x] #3 Delete redundant files and obsolete test fixtures
- [x] #4 Verify pytest (100% pass) and pyright transcription (0 errors)
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated pipeline, CLI, and script callers to canonical Tier 1 / Tier 2 modules, rewired all tests and documentation, pruned obsolete legacy directory transcription/cherokee/arpabet/, and verified 0 pyright errors and 358 passed pytest tests.
<!-- SECTION:FINAL_SUMMARY:END -->
