---
id: TASK-317
title: >-
  Integrate RAG-derived phonotactic rules into
  transcription/alignment/phonotactics.py
status: To Do
assignee: []
created_date: '2026-09-14 13:09'
labels: []
dependencies: []
ordinal: 333000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Once the RAG phonotactics output is received, implement transcription/alignment/phonotactics.py with functions for generating phonotactic intrusion site masks and syncope masks in the /t/ /th/ and /k/ /kh/ /kw/ /khw/ consonant system. Add dedicated unit tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Implement phonotactics.py with Cherokee rule parser
- [ ] #2 Generate accurate syncope_mask and intrusion_site_mask
- [ ] #3 Handle Cherokee digraphs (kw, khw, th, kh, tsh, tlh, lh, nh, wh, yh)
- [ ] #4 Add unit tests in test_phonotactics.py
<!-- AC:END -->
