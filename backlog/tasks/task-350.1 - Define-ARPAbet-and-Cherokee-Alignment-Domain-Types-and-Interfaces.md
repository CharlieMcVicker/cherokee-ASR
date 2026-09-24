---
id: TASK-350.1
title: Define ARPAbet and Cherokee Alignment Domain Types and Interfaces
status: Done
assignee:
  - '@supervisor'
created_date: '2026-09-18 14:44'
updated_date: '2026-09-18 14:56'
labels:
  - alignment
  - types
  - arpabet
dependencies: []
modified_files:
  - transcription/alignment/arpabet/__init__.py
  - transcription/alignment/arpabet/types.py
  - transcription/alignment/tests/test_arpabet_types.py
  - AGENTS.md
parent_task_id: TASK-350
priority: high
type: feature
ordinal: 370000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define strict, immutable data structures and pure functional transformation interfaces for the ARPAbet-to-Cherokee phonetic mapping system under transcription.alignment.arpabet.types following our Types and Maps architectural principles.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Define immutable domain models: ArpabetToken, CherokeeToken, WordManifestEntry, TopKHypothesis, WordInferenceCacheEntry, AcousticConfusionMatrix, and SyntheticCherokeeTarget
- [x] #2 Define pure mapping protocols for G2P extraction, DP traceback accumulation, and synthetic target projection
- [x] #3 Support lossless JSON serialization and deserialization across all cache and matrix data shapes
- [x] #4 Pass static type checks with Pyright and add unit tests in tests/alignment/test_arpabet_types.py
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect existing alignment models in transcription.alignment.models and orthography conventions.
2. Create transcription/alignment/arpabet package and define domain models in transcription/alignment/arpabet/types.py (ArpabetToken, CherokeeToken, WordManifestEntry, TopKHypothesis, WordInferenceCacheEntry, InferenceCacheManifest, AcousticConfusionMatrix, SyntheticCherokeeTarget).
3. Define pure mapping protocols (EnglishToArpabetProtocol, TracebackAlignerProtocol, SyntheticTargetProjectorProtocol).
4. Implement JSON serialization/deserialization helpers with roundtrip validation.
5. Create comprehensive unit tests in transcription/alignment/tests/test_arpabet_types.py and verify with pytest and pyright.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented strict immutable domain models and pure transformation protocols under transcription/alignment/arpabet/types.py adhering to Types and Maps architectural principles. Exported all models and protocols in __init__.py. Added 19 comprehensive unit tests in transcription/alignment/tests/test_arpabet_types.py covering immutability, normalizations, epsilon tokens, query methods, log-cost calculations, and lossless JSON serialization roundtrips. Verified 100% pass on pytest (19/19 arpabet tests, 165/165 full alignment suite) and 0 errors with Pyright.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented TASK-350.1: Defined immutable domain models (ArpabetToken, CherokeeToken, WordManifestEntry, TopKHypothesis, WordInferenceCacheEntry, InferenceCacheManifest, AcousticConfusionMatrix, SyntheticCherokeeTarget, AlignedTokenPair, TracebackAlignmentResult) and pure mapping protocols (EnglishToArpabetProtocol, G2PExtractorProtocol, TracebackAlignerProtocol, DPTracebackAccumulatorProtocol, SyntheticTargetProjectorProtocol) under transcription/alignment/arpabet. Provided full lossless JSON serialization/deserialization across all models. Verified with 19 comprehensive unit tests and zero errors from Pyright.
<!-- SECTION:FINAL_SUMMARY:END -->
