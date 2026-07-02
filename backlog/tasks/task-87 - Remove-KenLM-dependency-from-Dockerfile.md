---
id: TASK-87
title: Remove KenLM dependency from Dockerfile
status: Done
assignee:
  - '@agent'
created_date: '2026-07-02 21:40'
updated_date: '2026-07-02 21:40'
labels: []
dependencies: []
ordinal: 83000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove KenLM installation, build dependencies, and related libraries from the project's Dockerfile since we are no longer relying on KenLM.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Remove KenLM compilation stage and path configuration from Dockerfile
- [x] #2 Remove KenLM build libraries from apt-get install
- [x] #3 Remove pyctcdecode and kenlm python library from pip install list
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed KenLM system libraries (cmake, zlib, boost, eigen), the git-cloning/compilation steps, the PATH variable modifications, and deleted pyctcdecode/kenlm from the pip requirements in the Dockerfile.
<!-- SECTION:FINAL_SUMMARY:END -->
