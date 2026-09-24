---
id: TASK-375
title: >-
  Purge obsolete frontend and build artifacts, update .gitignore, and clean
  stale test files
status: Done
assignee:
  - '@agent'
created_date: '2026-09-24 16:28'
updated_date: '2026-09-24 16:31'
labels: []
dependencies: []
ordinal: 408300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The repository root contains obsolete legacy frontend directories (frontend/, package.json, package-lock.json), build directories (transcriber_build, transcriber_dist, workshop_transcription.egg-info), stale root artifacts (vocab.json, run01.txt, steps-isolated-cuda-asr.txt), redundant virtualenvs (venv, .venv), and outdated pre-refactor test files in digohwelisgi/alignment/tests/. Purging these unblocks clean pytest discovery and removes dead code.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 frontend/ and root package.json/package-lock.json are removed in favor of syllabary_transcriber/ui
- [x] #2 transcriber_build, transcriber_dist, and .egg-info artifacts are removed from tracking and added to .gitignore
- [x] #3 Stale root scratch text files and corrupt root vocab.json are pruned
- [x] #4 Pre-refactor duplicate test files in digohwelisgi/alignment/tests/ are removed or migrated to proper 4-tier module test folders
- [x] #5 pytest runs cleanly without collection errors across the entire codebase
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Remove legacy frontend/, package.json, package-lock.json\n2. Remove stale root files (vocab.json, run01.txt, steps-isolated-cuda-asr.txt, output_w2v2, remote_output_w2v2, venv, .venv)\n3. Remove build artifacts (transcriber_build, transcriber_dist, workshop_transcription.egg-info) and update .gitignore\n4. Remove stale pre-refactor test files in digohwelisgi/alignment/tests/\n5. Run pytest across the repository to verify clean discovery and passing tests
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Purged obsolete legacy frontend directory, root package.json and package-lock.json, stale scratch files (vocab.json, run01.txt, steps-isolated-cuda-asr.txt), old outputs (output_w2v2, remote_output_w2v2), virtual environments (.venv, venv), and build directories (transcriber_build, transcriber_dist, .egg-info). Configured pytest pythonpath in pyproject.toml and updated .gitignore. Verified clean pytest suite run with all 358 tests passing.
<!-- SECTION:FINAL_SUMMARY:END -->
