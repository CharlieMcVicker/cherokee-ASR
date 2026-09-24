---
id: TASK-374
title: >-
  Migrate training_data to data/training and update Dockerfile and training
  pipelines
status: Done
assignee:
  - '@agent'
created_date: '2026-09-24 16:28'
updated_date: '2026-09-24 16:35'
labels: []
dependencies: []
ordinal: 407300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The training_data directory sits at the root while other datasets reside in data/. Migrating training_data to data/training unifies the data directory structure. All consumers in digohwelisgi/training, Dockerfile, scripts, and distance metrics must be updated to prevent build and runtime regressions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 training_data is moved to data/training
- [x] #2 Dockerfile paths are updated (COPY data/training and COPY digohwelisgi)
- [x] #3 Training scripts and evaluation modules in digohwelisgi/training and scripts/ are updated to point to data/training
- [x] #4 Cherokee phonological distance module referencing training sentence CSVs is updated
- [x] #5 All training and model evaluation tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Move training_data to data/training\n2. Update Dockerfile to copy data/training, data/projects/cherokee_new_testament, and digohwelisgi\n3. Update training scripts, evaluation modules, split scripts, and distance metric references\n4. Update .gitignore if needed\n5. Run pytest and verify clean test execution
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated training_data to data/training. Updated Dockerfile to copy data/training, data/projects/cherokee_new_testament, and digohwelisgi. Updated all training dataset preparation and evaluation paths in digohwelisgi/training, scripts/, and phonological distance references. Verified with pytest suite (358 passed).
<!-- SECTION:FINAL_SUMMARY:END -->
