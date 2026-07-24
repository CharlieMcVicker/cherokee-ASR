---
id: TASK-163
title: Add align-cherokee CLI entry point to pyproject.toml
status: Done
assignee:
  - '@agent'
created_date: '2026-07-24 15:20'
updated_date: '2026-07-24 15:21'
labels: []
dependencies: []
ordinal: 159000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add entry point console script for align_cli main function so it can be installed using pip install -e .
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 pyproject.toml includes [project.scripts] align-cherokee entrypoint
- [x] #2 pip install -e . installs align-cherokee executable
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added align-cherokee script entry point to pyproject.toml and verified editable installation.
<!-- SECTION:FINAL_SUMMARY:END -->
