---
id: TASK-316
title: >-
  Tune per-character intrusion penalties and phonotactic site masks in
  CTCSegmentationAligner
status: To Do
assignee: []
created_date: '2026-09-12 21:22'
labels: []
dependencies: []
ordinal: 332000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Once ctc-segmentation implements per-token intrusive penalties (e.g. intrusive_penalties={'h': 4.0, '\'': 0.8}) and is_intrusive_site masks, integrate the updated ctc-segmentation release into CTCSegmentationAligner, calibrate the parameters across Cherokee New Testament chapters, and benchmark against baseline alignment records.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Update CTCSegmentationAligner to accept and pass intrusive_penalties (dict) and is_intrusive_site masks
- [ ] #2 Calibrate penalty values (syncope_penalty, /h/ intrusion penalty, /'/ glottal stop penalty) on Mark and Matthew
- [ ] #3 Run benchmark on 100 verses and verify appropriate /h/ and glottal stop insertion without spurious noise
- [ ] #4 Verify test suite passes with updated parameters
<!-- AC:END -->
