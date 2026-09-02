---
id: TASK-243
title: Remove redundant make_default from 0-arg classes
status: Done
assignee:
  - '@antigravity'
created_date: '2026-08-31 20:59'
updated_date: '2026-08-31 21:00'
labels: []
dependencies: []
ordinal: 245000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove make_default classmethods from stateless/0-arg classes (DefaultCERDistanceMetric, CherokeePhoneticPreprocessor, ManifestJsonAdapter) where standard construction requires no arguments.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Remove make_default from DefaultCERDistanceMetric, CherokeePhoneticPreprocessor, and ManifestJsonAdapter
- [x] #2 Update callers to instantiate 0-arg classes directly
- [x] #3 Verify pyright and pytest pass with 0 errors
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Remove make_default from DefaultCERDistanceMetric, CherokeePhoneticPreprocessor, and ManifestJsonAdapter.
2. Update callers across pipeline.py, preprocessors.py, sliding_window.py, distance_metrics.py, and test files to instantiate them with ClassName().
3. Run pyright and pytest.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed redundant make_default classmethods from 0-arg/stateless classes (DefaultCERDistanceMetric, CherokeePhoneticPreprocessor, ManifestJsonAdapter). Updated callers and tests to construct them directly via ClassName(). Verified 0 pyright errors and 86/86 pytest tests pass.
<!-- SECTION:FINAL_SUMMARY:END -->
