---
id: TASK-242
title: >-
  Enforce explicit required parameters on constructors and add make_default_*
  factory functions
status: Done
assignee:
  - '@antigravity'
created_date: '2026-08-31 20:55'
updated_date: '2026-08-31 20:59'
labels: []
dependencies: []
ordinal: 244000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove default argument fallbacks from __init__ constructors across transcription.alignment classes to require explicit dependency injection, and provide make_default_* factory functions for standard construction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Make all initialization arguments with defaults required in core aligners, adapters, pipeline, and strategies
- [x] #2 Implement make_default_* factory functions for word aligner, sliding window aligner, pipeline, adapters, preprocessors, metrics, and extractors
- [x] #3 Update cli.py and all test suites to use explicit constructors or factory functions
- [x] #4 Verify pyright and pytest pass with 0 errors
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Make all __init__ arguments without defaults required on alignment classes.
2. Add @classmethod make_default(...) factory methods on each class (NeedlemanWunschWordAligner, SlidingWindowDTWAligner, AlignmentPipeline, BibleMetadataVerseAdapter, GenericChunkListAdapter, PraatTextGridAdapter, ManifestJsonAdapter, SyllabaryToPhoneticPreprocessor, PhonologicalDistanceMetric, CherokeeASRExtractor, CallbackEmissionsExtractor).
3. Update cli.py, test_core_aligner.py, test_pipeline.py, test_cli.py, test_adapters.py, test_domain_and_strategies.py to use explicit constructors or ClassName.make_default(...).
4. Verify pyright and pytest.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Made all constructor arguments without defaults strictly required across alignment classes and added @classmethod make_default(...) factory methods on each class. Updated cli.py and all test suites. Verified 0 pyright errors and 86/86 pytest tests passing.
<!-- SECTION:FINAL_SUMMARY:END -->
