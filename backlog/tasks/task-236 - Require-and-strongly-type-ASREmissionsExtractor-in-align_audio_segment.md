---
id: TASK-236
title: Require and strongly type ASREmissionsExtractor in align_audio_segment
status: Done
assignee:
  - '@subagent'
created_date: '2026-08-30 22:55'
updated_date: '2026-08-30 22:58'
labels: []
dependencies: []
ordinal: 235000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace loose optional model_or_fn / processor parameters in align_audio_segment with a required, strongly-typed ASREmissionsExtractor. Introduce concrete extractor adapters (CherokeeASRExtractor, CallbackEmissionsExtractor, PrecomputedEmissionsExtractor) in transcription.alignment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement ASREmissionsExtractor implementations (CherokeeASRExtractor, CallbackEmissionsExtractor, PrecomputedEmissionsExtractor) in transcription.alignment.strategies
- [x] #2 Update align_audio_segment signature to require non-optional extractor parameter with strict typing
- [x] #3 Update callers in align_cli.py and test_aligner.py to supply typed extractor instances
- [x] #4 All tests in test suite pass cleanly
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented ASREmissionsExtractor strategies (CherokeeASRExtractor, CallbackEmissionsExtractor, PrecomputedEmissionsExtractor), exported them in transcription.alignment packages, refactored align_audio_segment to require a strongly-typed extractor parameter, updated callers in align_cli.py and test_aligner.py, and added comprehensive unit tests.
<!-- SECTION:FINAL_SUMMARY:END -->
