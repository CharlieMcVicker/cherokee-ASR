---
id: TASK-228
title: Move project requirements into pyproject.toml
status: Done
assignee:
  - '@antigravity'
created_date: '2026-08-27 13:46'
updated_date: '2026-08-27 13:51'
labels: []
dependencies: []
ordinal: 219000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate all project requirements into pyproject.toml dependencies and optional-dependencies to allow proper package installation via pip/setuptools
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add project dependencies from requirements.txt to dependencies in pyproject.toml
- [x] #2 Include missing runtime dependencies (e.g. pydub, tqdm, huggingface_hub) if needed
- [x] #3 Configure optional/dev dependencies if appropriate (e.g. black, pre-commit)
- [x] #4 Verify package can be installed and imports resolve cleanly
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect pyproject.toml, requirements.txt, and repository dependencies.\n2. Update pyproject.toml dependencies array with project requirements (core and dev/optional dependencies).\n3. Check package installation using pip install -e . in conda cherokee-asr environment.\n4. Verify importability and CLI entrypoints.\n5. Mark task complete with summary and tests.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated all project requirements into pyproject.toml PEP 621 dependencies array and optional dev dependencies. Updated package discovery to find both transcription and syllabary_transcriber packages. Updated requirements.txt to point to editable install with dev extras (-e .[dev]). Verified installation with pip install -e . and confirmed pytest test passes.
<!-- SECTION:FINAL_SUMMARY:END -->
