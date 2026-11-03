---
id: TASK-390
title: Fix pre-commit hook for digohwelisgi module structure
status: Done
assignee:
  - '@self'
created_date: '2026-11-03 20:35'
updated_date: '2026-11-03 20:36'
labels: []
dependencies: []
modified_files:
  - .pre-commit-config.yaml
ordinal: 423300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update .pre-commit-config.yaml to target digohwelisgi instead of deprecated transcription module path and clean up stale exclude paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Update pyright hook in .pre-commit-config.yaml to match digohwelisgi
- [x] #2 Remove stale transcription paths from black exclude
- [x] #3 Verify pre-commit run on digohwelisgi passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update .pre-commit-config.yaml to change pyright hook name and files regex from ^transcription/ to ^digohwelisgi/.
2. Clean up black exclude regex in .pre-commit-config.yaml to remove defunct transcription/ path.
3. Run pre-commit run --all-files to verify pyright and black hooks succeed on all digohwelisgi files.
4. Check off acceptance criteria and close TASK-390.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated .pre-commit-config.yaml to point the pyright hook to ^digohwelisgi/ and renamed it from `pyright (transcription)` to `pyright (digohwelisgi)`. Removed stale transcription/ paths from the black exclude regex. Reinstalled pre-commit git hooks to reference the active conda environment and verified `pre-commit run --all-files` and `pre-commit run --files digohwelisgi/...` pass with zero errors.
<!-- SECTION:FINAL_SUMMARY:END -->
