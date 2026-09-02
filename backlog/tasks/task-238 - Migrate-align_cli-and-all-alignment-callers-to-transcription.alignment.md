---
id: TASK-238
title: Migrate align_cli and all alignment callers to transcription.alignment
status: Done
assignee:
  - '@antigravity'
created_date: '2026-08-31 20:33'
updated_date: '2026-08-31 20:43'
labels: []
dependencies: []
ordinal: 237000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the align-cherokee CLI runner, new_testament pipeline, and all alignment callers from the legacy timestamping facade to natively consume transcription.alignment domain models, SlidingWindowDTWAligner, and inbound/outbound adapters.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement native CLI runner in transcription.alignment.cli with support for Bible metadata, chunk lists, VAD toggles, Praat TextGrid and manifest export, debug exports, and metrics reporting
- [x] #2 Update pyproject.toml align-cherokee entrypoint to transcription.alignment.cli:main
- [x] #3 Update transcription.new_testament.pipeline and other callers to consume transcription.alignment natively
- [x] #4 Ensure backward compatibility facade in transcription.timestamping re-exports or delegates cleanly without regressions
- [x] #5 Ensure all unit tests pass across transcription test suite
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Enhance transcription.alignment.ports.protocols to declare InboundChunkAdapter and OutboundAlignmentAdapter protocols.
2. Build the abstracted runtime orchestrator transcription.alignment.pipeline (AlignmentPipeline) that takes injected inbound chunk adapter, ASR extractor, alignment engine, strategies, and outbound adapters, and executes the alignment lifecycle.
3. Implement the lightweight CLI wrapper transcription.alignment.cli that configures and injects the selected adapters into AlignmentPipeline and triggers execution.
4. Move VAD audio segmentation (segment_long_audio, AudioChunk) into transcription.audio.segment and normalize preprocessors directly in transcription.alignment.strategies.preprocessors.
5. Remove dead code in transcription.timestamping, eliminating obsolete shims and duplicate implementations.
6. Update pyproject.toml and transcription.new_testament.pipeline to use transcription.alignment directly.
7. Update test suites to test transcription.alignment, pipeline, and CLI natively, verifying all tests pass.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Successfully completed full migration of alignment architecture to transcription.alignment:
- Implemented abstracted runtime orchestrator AlignmentPipeline in transcription.alignment.pipeline.
- Built lightweight CLI wrapper transcription.alignment.cli with main() and run_alignment_pipeline.
- Added InboundChunkAdapter and OutboundAlignmentAdapter protocols.
- Updated pyproject.toml align-cherokee entrypoint and new_testament pipeline.
- Canonicalized VAD segmentation in transcription.audio.segment.
- Fully removed obsolete dead code in transcription/timestamping.
- All 86 unit tests pass cleanly.
<!-- SECTION:FINAL_SUMMARY:END -->
