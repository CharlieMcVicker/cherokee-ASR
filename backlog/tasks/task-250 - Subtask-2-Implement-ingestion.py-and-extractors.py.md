---
id: TASK-250
title: 'Subtask 2: Implement ingestion.py and extractors.py'
status: Done
assignee:
  - '@agent-subtask2'
created_date: '2026-09-01 16:37'
updated_date: '2026-09-01 16:40'
labels: []
dependencies: []
ordinal: 252000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement data ingestion functions returning (chunks, metadata_lookup) and audio/ASR token extraction classes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement ingestion.py with load_bible_chunks and load_generic_chunks.
- [x] #2 Implement extractors.py with CherokeeASRExtractor, CallbackEmissionsExtractor, PrecomputedEmissionsExtractor.
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented transcription/alignment/ingestion.py with load_bible_chunks and load_generic_chunks returning (chunks, source_lookup), and transcription/alignment/extractors.py with prepare_audio_chunks helper and CherokeeASRExtractor, CallbackEmissionsExtractor, and PrecomputedEmissionsExtractor classes. Added comprehensive unit tests in transcription/alignment/tests/test_ingestion_and_extractors.py.
<!-- SECTION:FINAL_SUMMARY:END -->
