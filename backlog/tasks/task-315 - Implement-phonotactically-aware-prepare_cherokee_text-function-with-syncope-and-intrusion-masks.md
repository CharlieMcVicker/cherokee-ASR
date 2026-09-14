---
id: TASK-315
title: >-
  Implement phonotactically-aware prepare_cherokee_text function with syncope
  and intrusion masks
status: To Do
assignee: []
created_date: '2026-09-12 21:22'
labels: []
dependencies: []
ordinal: 331000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement a specialized prepare_cherokee_text function in transcription/alignment/ that analyzes Cherokee syllabary and phonetic words to generate ground truth matrices, syncope_masks (for valid vowel syncopation sites), and intrusion_site_masks (for valid pre-aspiration, laryngeal alternation, and glottal stop sites). Integrate with CTCSegmentationAligner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Implement prepare_cherokee_text() supporting Cherokee phonotactic rules
- [ ] #2 Generate accurate syncope_mask and intrusion_mask distinguishing vowel loss from intrusive detours
- [ ] #3 Integrate custom text preparation into CTCSegmentationAligner
- [ ] #4 Add comprehensive unit tests for phonotactic text preparation
<!-- AC:END -->
