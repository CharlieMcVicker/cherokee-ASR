---
id: TASK-158
title: Support separate --bible-metadata and --chunk-list arguments in CLI
status: Done
assignee:
  - '@agent'
created_date: '2026-07-23 16:16'
updated_date: '2026-07-23 16:16'
labels: []
dependencies: []
ordinal: 154000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update prepare_ground_truth.py and align_cli.py to support distinct CLI arguments for ingesting Bible metadata vs standard segment chunk list JSON files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CLI accepts --bible-metadata for key-value Bible metadata JSON
- [x] #2 CLI accepts --chunk-list for array of segment chunk dicts
- [x] #3 Unit tests cover both ingest modes
- [x] #4 All timestamping tests pass
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added parse_chunk_list in prepare_ground_truth.py to ingest array-formatted chunk JSON files. Updated align_cli.py argument parser with mutually exclusive --bible-metadata and --chunk-list kwargs, preserving --metadata as backward-compatible alias. Added unit tests.
<!-- SECTION:FINAL_SUMMARY:END -->
