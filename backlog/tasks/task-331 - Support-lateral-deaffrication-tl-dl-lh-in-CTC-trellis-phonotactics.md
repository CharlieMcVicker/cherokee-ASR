---
id: TASK-331
title: Support lateral deaffrication (tl -> lh) in CTC trellis phonotactics
status: To Do
assignee: []
created_date: '2026-09-14 20:03'
updated_date: '2026-09-14 20:27'
labels: []
dependencies: []
ordinal: 347000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In Cherokee speech, written lateral affricate tl (e.g. from citation syllabary ᎣᏝ 'otla') frequently undergoes lateral deaffrication to lateral fricative/approximant lh ('olha'). To prevent the CTC segmentation trellis from penalizing spoken 'olha' or dropping alignment confidence, prepare_cherokee_text in transcription/alignment/phonotactics.py should license optional deletion/skip of the leading stop 't' in 'tl' clusters via the syncope/optional token mask.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 prepare_cherokee_text marks leading stop 't' in 'tl' clusters as eligible for optional deletion in syncope_mask
- [ ] #2 Unit tests in test_phonotactics.py verify tl deaffrication masking behavior
- [ ] #3 CTC aligner successfully aligns deaffricated speech (e.g. ᎣᏝ -> spoken olha) with high confidence
- [ ] #4 All pytest tests and pyright static typing pass with zero regressions
<!-- AC:END -->
