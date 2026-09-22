---
id: TASK-360.3
title: 'Phase 3A: Build Scripture chapter & verse slicing pipeline'
status: To Do
assignee: []
created_date: '2026-09-21 20:24'
updated_date: '2026-09-21 20:47'
labels: []
dependencies: []
parent_task_id: TASK-360
ordinal: 389100
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Consolidate New Testament Bible chapter ingestion, DG-to-TTH normalization, continuous chapter CTC segmentation, and verse slicing using existing AudioChunk / AlignedChunk domain models into transcription.pipelines.scripture (replacing transcription/new_testament/).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 ScripturePipeline implemented in transcription.pipelines.scripture composing Tier 1 CTCSegmentationAligner and Tier 2 cherokee.phonotactics
- [ ] #2 Ingestion of Bible JSON/TSV chapters supports chapter-level continuous audio alignment
- [ ] #3 Verse boundary partitioning slices clean 16kHz WAV clips using existing AudioChunk/AlignedChunk domain models
- [ ] #4 Integration tests in test_pipeline.py pass under new imports
<!-- AC:END -->
