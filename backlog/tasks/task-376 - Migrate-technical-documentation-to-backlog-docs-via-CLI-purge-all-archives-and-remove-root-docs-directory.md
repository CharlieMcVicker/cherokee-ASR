---
id: TASK-376
title: >-
  Migrate technical documentation to backlog/docs via CLI, purge all archives,
  and remove root docs directory
status: Done
assignee:
  - '@agent'
created_date: '2026-09-24 17:30'
updated_date: '2026-09-24 17:32'
labels: []
dependencies: []
ordinal: 409300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
All project documentation should reside in backlog/docs to provide a single, unified source of truth searchable via Backlog.md CLI. Dead archive folders (data/archive, docs/archive) should be deleted rather than preserved as clutter. Top-level docs/ should be removed and README.md / AGENTS.md updated.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All dead archive folders (data/archive/, docs/archive/) are permanently removed
- [x] #2 All active architecture guides and technical specs are migrated into backlog/docs using the Backlog CLI
- [x] #3 The root docs/ directory is removed
- [x] #4 README.md and AGENTS.md references are updated to point to backlog/docs
- [x] #5 Pytest suite and type checker continue to pass cleanly
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Delete data/archive/ and docs/archive/\n2. Migrate active architecture guides into backlog/docs/architecture/ using backlog doc create/update\n3. Migrate active specs into backlog/docs/specs/ using backlog doc create/update\n4. Remove root docs/ directory\n5. Update README.md and AGENTS.md documentation links\n6. Run pytest and pyright to verify everything remains green
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Purged dead archive directories (data/archive/, docs/archive/). Migrated all active subsystem architecture guides, technical specifications, and analyses into backlog/docs/ via Backlog CLI (creating doc-9 through doc-23). Removed the top-level docs/ directory. Updated README.md and AGENTS.md documentation links. Verified complete test suite (358 passed) and pyright static type checks (0 errors).
<!-- SECTION:FINAL_SUMMARY:END -->
