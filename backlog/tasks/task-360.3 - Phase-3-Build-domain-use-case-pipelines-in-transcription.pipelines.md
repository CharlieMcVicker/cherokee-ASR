---
id: TASK-360.3
title: 'Phase 3A: Build Scripture chapter & verse slicing pipeline'
status: Done
assignee:
  - '@phase-3a-implementor'
created_date: '2026-09-21 20:24'
updated_date: '2026-09-22 19:35'
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
- [x] #1 ScripturePipeline implemented in transcription.pipelines.scripture composing Tier 1 CTCSegmentationAligner and Tier 2 cherokee.phonotactics
- [x] #2 Ingestion of Bible JSON/TSV chapters supports chapter-level continuous audio alignment
- [x] #3 Verse boundary partitioning slices clean 16kHz WAV clips using existing AudioChunk/AlignedChunk domain models
- [x] #4 Integration tests in test_pipeline.py pass under new imports
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Build transcription/pipelines/scripture/ingestion.py supporting Bible JSON/TSV loading into TextChunks, verse metadata dictionaries, and phonetic normalizer integration.
2. Build transcription/pipelines/scripture/pipeline.py with ScripturePipeline (and align_chapter procedural adapter) composing Tier 1 CTCSegmentationAligner (transcription.core.alignment.ctc), Tier 2 prepare_cherokee_text (transcription.cherokee.phonotactics), audio loading/inference via ModelOutput, AudioChunk/AlignedChunk domain models, verse boundary slicing to 16kHz WAV, and export to TextGrid/manifest.
3. Expose ScripturePipeline in transcription/pipelines/scripture/__init__.py and transcription/pipelines/__init__.py.
4. Update transcription/new_testament/ to act as backwards-compatible delegation shims pointing to transcription.pipelines.scripture.
5. Create comprehensive tests in transcription/pipelines/scripture/tests/test_pipeline.py and verify new_testament existing tests pass.
6. Verify pyright and pytest pass cleanly.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented ScripturePipeline in transcription.pipelines.scripture composing Tier 1 CTCSegmentationAligner and Tier 2 cherokee.phonotactics. Built chapter ingestion supporting continuous audio alignment, AudioChunk/AlignedChunk domain models, clean 16kHz WAV verse boundary partitioning, and backwards-compatible facades in transcription.new_testament. All 469 unit and integration tests passing with 0 pyright errors.
<!-- SECTION:FINAL_SUMMARY:END -->
