---
id: TASK-360.10
title: 'Phase 3B: Build code-switched dialogue alignment pipeline'
status: To Do
assignee: []
created_date: '2026-09-21 20:33'
labels: []
dependencies: []
parent_task_id: TASK-360
ordinal: 389200
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Consolidate multi-speaker dialogue text parsing, speaker prefix handling, code-switched clitic projection, Silero VAD soft-masking, and 7-tier Praat TextGrid generation into transcription.pipelines.dialogue (replacing transcription/alignment/pipeline.py).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 DialogueAlignmentPipeline implemented in transcription.pipelines.dialogue orchestrating dialogue alignment
- [ ] #2 Token discrimination isolates English tokens from Cherokee syncope/intrusion masks
- [ ] #3 Assembles and exports 7-tier Praat TextGrids (Turn, Speaker, Syllabary, English, Reconciled, CTC Word, Phoneme) and JSON manifests
- [ ] #4 Dialogue realignment tests in test_interview_realignment.py pass under new imports
<!-- AC:END -->
