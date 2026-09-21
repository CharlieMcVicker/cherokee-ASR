---
id: TASK-278
title: >-
  Filter Bible Datasets, Generate Multi-Tier Manifests, and Package
  Retranscription Audio
status: To Do
assignee: []
created_date: '2026-09-10 17:50'
labels:
  - dataset
  - new-testament
  - export
dependencies: []
ordinal: 290000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Using the identified optimal cost threshold T*, partition the realigned Mark and Matthew datasets into clean training CSVs, full manifests with cost/status columns, and package rejected verses and audio slices into cherokee_new_testament/needs_retranscription/ with rejected_verses.csv.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Export clean training CSVs (mark_clean.csv, matthew_clean.csv) containing only verses with cost <= T*
- [ ] #2 Export omnibus manifest CSVs (mark_manifest.csv, matthew_manifest.csv) with cost, status, start_sec, end_sec, reference_sentence, asr_hypothesis
- [ ] #3 Extract and stage rejected verse audio clips into cherokee_new_testament/needs_retranscription/
- [ ] #4 Generate rejected_verses.csv with full metadata for review and retranscription
<!-- AC:END -->
