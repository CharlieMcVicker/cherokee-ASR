---
id: TASK-387
title: Optimize Dockerfile and add remote dataset fetch script
status: Done
assignee:
  - '@self'
created_date: '2026-11-03 19:45'
updated_date: '2026-11-03 20:03'
labels: []
dependencies:
  - TASK-386
modified_files:
  - scripts/fetch_training_data.py
  - Dockerfile
  - backlog/docs/guides/vast_ai_training/doc-2 - Vast.ai-Training-Guide.md
  - digohwelisgi/cherokee/training/tests/test_package_and_fetch.py
ordinal: 420300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove bulky local data COPY layers from the Dockerfile and introduce an idempotent fetch script (scripts/fetch_training_data.py) using snapshot_download from Hugging Face Hub for fast container startup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Create scripts/fetch_training_data.py supporting idempotent download and cache checking
- [x] #2 Remove COPY data/training and COPY data/projects from Dockerfile
- [x] #3 Update Vast.ai training documentation and instructions for downloading data at startup
- [x] #4 Verify training script executes successfully in container with fetched data
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Dockerfile builds without large audio layers and fetch script successfully downloads dataset
<!-- DOD:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create scripts/fetch_training_data.py supporting single-stream archive download and snapshot download fallback with idempotency.
2. Remove bulky COPY data/ layers from Dockerfile and copy scripts/ directory instead.
3. Update Vast.ai Training Guide (doc-2) with the new automated fetch workflow.
4. Write unit tests for packaging and fetch script and verify with pytest and pyright.
5. Verify that fetched data paths resolve directly in training scripts.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Optimized Dockerfile and implemented remote dataset fetcher.
- Created scripts/fetch_training_data.py supporting fast single-request archive extraction (cherokee_asr_data.tar.gz) and snapshot_download fallback, with full idempotency on container restarts.
- Removed bulky COPY data/ layers from Dockerfile and added scripts/ directory, dropping image bloat by ~1.6 GB.
- Updated Vast.ai training guide (doc-2) with remote fetching instructions.
- Added comprehensive unit tests and verified with pytest (14 passed) and pyright (0 errors).
<!-- SECTION:FINAL_SUMMARY:END -->
