---
id: TASK-364
title: Prune legacy transcription/alignment and arpabet forwarding shims
status: To Do
assignee: []
created_date: '2026-09-23 15:55'
labels: []
dependencies: []
ordinal: 394300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
transcription/alignment/arpabet/ (projector.py, codeswitched_preparer.py), transcription/alignment/calibrated_distance_metrics.py, normalizers.py, and pipeline.py remain as backward-compatibility forwarding wrappers. Clean break requires call sites to import directly from Tier 2 transcription.cherokee and Tier 3 transcription.pipelines, and removing the forwarding wrappers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 transcription/alignment/arpabet/ forwarding shims pruned
- [ ] #2 transcription/alignment/ normalizers.py, calibrated_distance_metrics.py, and pipeline.py deleted
- [ ] #3 All tests pass and pyright reports 0 errors
<!-- AC:END -->
