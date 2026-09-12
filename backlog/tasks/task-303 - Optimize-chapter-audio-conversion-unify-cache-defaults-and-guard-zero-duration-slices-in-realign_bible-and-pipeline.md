---
id: TASK-303
title: >-
  Optimize chapter audio conversion, unify cache defaults, and guard
  zero-duration slices in realign_bible and pipeline
status: Done
assignee: []
created_date: '2026-09-12 19:36'
updated_date: '2026-09-12 19:40'
labels: []
dependencies: []
ordinal: 319000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pre-convert chapter audio to 16kHz mono once per chapter in realign_bible.py, guard zero-duration audio slice exports, unify DEFAULT_CACHE_DIR across pipeline.py, and update docstrings in ingestion.py.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pre-convert audio_seg to 16kHz mono once per chapter before verse slicing in realign_bible.py
- [x] #2 Add duration guard for empty audio slices in realign_bible.py
- [x] #3 Unify default cache directory in new_testament pipeline with DEFAULT_CACHE_DIR from ctc_aligner.py
- [x] #4 Fix normalizer docstring in ingestion.py
<!-- AC:END -->
