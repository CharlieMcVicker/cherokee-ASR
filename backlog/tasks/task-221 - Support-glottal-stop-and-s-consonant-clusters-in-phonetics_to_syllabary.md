---
id: TASK-221
title: Support glottal stop and s-consonant clusters in phonetics_to_syllabary
status: To Do
assignee:
  - '@agent'
created_date: '2026-08-12 22:12'
updated_date: '2026-09-14 13:26'
labels: []
dependencies: []
ordinal: 212000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix kanolv'vhska -> ᎦᏃᎸᎥᏍᎦ conversion by handling glottal stops (') and post-vocalic s-consonant cluster segmentation (hsk -> h, s, k).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Fix kanolv'vhska -> ᎦᏃᎸᎥᏍᎦ test case in test_syllabary_map.py
- [ ] #2 Verify unittest suite passes
<!-- AC:END -->
