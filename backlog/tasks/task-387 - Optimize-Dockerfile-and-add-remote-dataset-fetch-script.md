---
id: TASK-387
title: Optimize Dockerfile and add remote dataset fetch script
status: To Do
assignee: []
created_date: '2026-11-03 19:45'
labels: []
dependencies:
  - TASK-386
ordinal: 420300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove bulky local data COPY layers from the Dockerfile and introduce an idempotent fetch script (scripts/fetch_training_data.py) using snapshot_download from Hugging Face Hub for fast container startup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Create scripts/fetch_training_data.py supporting idempotent download and cache checking
- [ ] #2 Remove COPY data/training and COPY data/projects from Dockerfile
- [ ] #3 Update Vast.ai training documentation and instructions for downloading data at startup
- [ ] #4 Verify training script executes successfully in container with fetched data
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Dockerfile builds without large audio layers and fetch script successfully downloads dataset
<!-- DOD:END -->
