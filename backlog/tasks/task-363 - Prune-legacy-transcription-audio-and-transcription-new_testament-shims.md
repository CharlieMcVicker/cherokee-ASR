---
id: TASK-363
title: Prune legacy transcription/audio and transcription/new_testament shims
status: Done
assignee:
  - '@subagent'
created_date: '2026-09-23 15:55'
updated_date: '2026-09-23 16:17'
labels: []
dependencies: []
modified_files:
  - transcription/audio/__init__.py
  - transcription/audio/extract.py
  - transcription/audio/non_speech_masking.py
  - transcription/audio/segment.py
  - transcription/audio/test_audio_segmenter.py
  - transcription/audio/test_non_speech_masking.py
  - transcription/new_testament/__init__.py
  - transcription/new_testament/pipeline.py
  - transcription/new_testament/tests/test_pipeline.py
  - transcription/core/audio/tests/test_audio_segmenter.py
  - transcription/core/audio/tests/test_non_speech_masking.py
  - transcription/pipelines/scripture/tests/test_scripture_pipeline.py
  - transcription/alignment/ctc_aligner.py
  - transcription/alignment/tests/test_cli.py
  - transcription/alignment/tests/test_interview_realignment.py
  - transcription/alignment/tests/test_syllabary_runners.py
  - transcription/pipelines/dialogue/tests/test_dialogue_pipeline.py
  - server.py
  - scripts/process_interviews.py
  - scripts/benchmark_ctc_segmentation_100_verses.py
  - scripts/calibrate_intrusion_penalties.py
  - docs/audio_segmentation.md
ordinal: 393300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
transcription/audio/non_speech_masking.py and transcription/new_testament/ (pipeline.py) exist solely as forwarding shims to transcription.core.audio and transcription.pipelines.scripture. Callers and test fixtures should be updated to point directly to Tier 1 core.audio and Tier 3 pipelines.scripture, and the legacy directories removed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 transcription/new_testament/ package deleted and tests updated to test_scripture_pipeline.py
- [x] #2 transcription/audio/ forwarding shims pruned in favor of transcription.core.audio
- [x] #3 All tests pass and pyright reports 0 errors
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Consolidate new_testament tests into transcription/pipelines/scripture/tests/test_scripture_pipeline.py and core audio tests into transcription/core/audio/tests/\n2. Update all imports referencing transcription.audio and transcription.new_testament to transcription.core.audio and transcription.pipelines.scripture\n3. Delete transcription/audio/ and transcription/new_testament/ legacy directories\n4. Update documentation referencing legacy transcription.audio modules\n5. Run pytest and pyright transcription to verify zero regressions\n6. Commit changes incrementally and finalize task
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Pruned legacy transcription/audio and transcription/new_testament forwarding shims. Relocated unit tests to transcription/core/audio/tests/ and consolidated scripture tests into transcription/pipelines/scripture/tests/test_scripture_pipeline.py. Updated all callers across server, scripts, tests, and documentation to consume Tier 1 transcription.core.audio and Tier 3 transcription.pipelines.scripture. Verified 100% test pass rate across all 440 tests with 0 pyright errors.
<!-- SECTION:FINAL_SUMMARY:END -->
