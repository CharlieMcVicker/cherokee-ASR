---
id: TASK-237
title: >-
  Enforce strict ASREmissionsExtractor protocol without union types in aligner
  APIs
status: Done
assignee:
  - '@myself'
created_date: '2026-08-30 22:59'
updated_date: '2026-08-30 23:00'
labels: []
dependencies: []
ordinal: 236000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove Union[CherokeeASRModel, ASREmissionsExtractor] in create_audio_aligner and align_audio_segment, strictly requiring ASREmissionsExtractor. Remove redundant skip_vad from create_audio_aligner since VAD configuration belongs to the extractor.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Strictly type extractor parameter as ASREmissionsExtractor across create_audio_aligner and align_audio_segment
- [x] #2 Remove redundant skip_vad parameter from create_audio_aligner and align_audio_segment
- [x] #3 Update align_cli and test callers to pass concrete ASREmissionsExtractor instances
- [x] #4 All tests pass
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Enforced strict ASREmissionsExtractor protocol without union types in create_audio_aligner and align_audio_segment. Removed redundant skip_vad argument since VAD chunking is configured on the extractor itself. All 96 tests pass.
<!-- SECTION:FINAL_SUMMARY:END -->
