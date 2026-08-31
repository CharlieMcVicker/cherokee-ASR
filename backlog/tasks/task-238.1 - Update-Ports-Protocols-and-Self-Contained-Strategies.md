---
id: TASK-238.1
title: 'Update Ports, Protocols, and Self-Contained Strategies'
status: Done
assignee:
  - '@subagent'
created_date: '2026-08-31 20:38'
updated_date: '2026-08-31 20:39'
labels: []
dependencies: []
parent_task_id: TASK-238
ordinal: 238000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Declare InboundChunkAdapter and OutboundAlignmentAdapter protocols in transcription.alignment.ports.protocols; add load_chunks to inbound adapters; implement native phonetic normalization in CherokeePhoneticPreprocessor; and ensure AudioChunk / segment_long_audio are canonicalized in transcription.audio.segment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Declare InboundChunkAdapter and OutboundAlignmentAdapter in protocols.py
- [x] #2 Ensure BibleMetadataVerseAdapter and GenericChunkListAdapter implement InboundChunkAdapter with load_chunks()
- [x] #3 Make CherokeePhoneticPreprocessor self-contained without importing from timestamping
- [x] #4 Ensure CherokeeASRExtractor and audio segmentation import from transcription.audio.segment
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Declared InboundChunkAdapter and OutboundAlignmentAdapter in protocols.py; updated BibleMetadataVerseAdapter and GenericChunkListAdapter with path constructor argument and load_chunks() implementation; made CherokeePhoneticPreprocessor self-contained with direct normalize_text_for_alignment implementation; migrated AudioChunk and segment_long_audio into transcription.audio.segment with backwards compatibility in audio_segmenter.py; added adapter tests and verified all tests pass in transcription/.
<!-- SECTION:FINAL_SUMMARY:END -->
