---
id: TASK-179
title: Add precommit hook for pyright on transcription module
status: Done
assignee:
  - '@agent'
created_date: '2026-07-27 17:48'
updated_date: '2026-07-27 17:50'
labels: []
dependencies: []
ordinal: 175000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Configure pyright in pre-commit hook to check for type errors and issues in the transcription module.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 pyright added to .pre-commit-config.yaml for transcription module
- [x] #2 pyright configured/installed and pre-commit passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect current .pre-commit-config.yaml and pyright setup.
2. Update .pre-commit-config.yaml to add pyright hook targeting the transcription module.
3. Test running pre-commit hook / pyright to verify it works and resolves any issues.
4. Mark AC/DoD and set TASK-179 status to Done.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added Pyright hook targeting the transcription module to .pre-commit-config.yaml and created pyrightconfig.json configured to use the cherokee-asr conda environment.
<!-- SECTION:FINAL_SUMMARY:END -->
