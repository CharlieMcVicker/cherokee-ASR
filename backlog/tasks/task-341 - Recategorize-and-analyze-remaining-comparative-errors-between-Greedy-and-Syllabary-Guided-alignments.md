---
id: TASK-341
title: >-
  Recategorize and analyze remaining comparative errors between Greedy and
  Syllabary-Guided alignments
status: To Do
assignee: []
created_date: '2026-09-16 17:10'
labels: []
dependencies: []
documentation:
  - >-
    backlog/docs/evaluation/doc-8 -
    Syllabary-Guided-vs-Greedy-CTC-Segmentation-Progress-Tracker.md
priority: high
type: spike
ordinal: 357000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Perform a comprehensive 3-way comparative error analysis across the 1,389 Cherokee Syllabary dataset samples (Ground Truth vs. Greedy ASR vs. Syllabary-Guided CTC Segmentation) following the resolution of lateral deaffrication and laryngeal masking. Classify all remaining degradation events into structured buckets (e.g. voiceless sonorant aspiration, spelling inconsistencies in ground truth, hiatus glottal stops, minor medial syncope) to guide the next phase of aligner enhancements.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Generate an exhaustive 3-way comparative error breakdown script and report across clean and noisy evaluation conditions
- [ ] #2 Classify remaining mismatch events into discrete phonological and transcript categories with exact counts and examples
- [ ] #3 Update doc-8 with the revised error taxonomy and next targeted interventions
<!-- AC:END -->
