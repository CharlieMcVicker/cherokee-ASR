---
id: TASK-385
title: Implement language-agnostic audio dataset packager in digohwelisgi.core
status: Done
assignee:
  - '@self'
created_date: '2026-11-03 19:45'
updated_date: '2026-11-03 19:50'
labels: []
dependencies:
  - TASK-384
modified_files:
  - digohwelisgi/core/datasets/__init__.py
  - digohwelisgi/core/datasets/types.py
  - digohwelisgi/core/datasets/packaging.py
  - digohwelisgi/core/datasets/tests/test_packaging.py
  - digohwelisgi/core/__init__.py
  - AGENTS.md
ordinal: 418300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement a generic, pure-function audio dataset packager and staging engine in Tier 1 that scans split CSVs, verifies 100% audio existence and integrity, stages only referenced audio into target structures, and writes dataset manifests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Define immutable domain types (DatasetSplit, PackagedAudioRecord, PackagingManifest) in Tier 1
- [x] #2 Implement pure package_dataset procedure to isolate only referenced audio and generate manifests
- [x] #3 Include optional Hugging Face Hub upload capability via huggingface_hub
- [x] #4 Add unit tests verifying exact audio extraction, manifest generation, and missing-file rejection
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Unit tests pass and pyright digohwelisgi passes cleanly
<!-- DOD:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create digohwelisgi/core/datasets module with types.py and packaging.py.
2. Define immutable models: DatasetSplit, PackagedAudioRecord, PackagingManifest, PackagingResult.
3. Implement pure package_dataset procedure: scanning splits, resolving audio, computing sha256 checksums, copying/staging files, and creating manifests.
4. Add push_to_hf_hub procedure with huggingface_hub integration.
5. Write unit tests in digohwelisgi/core/datasets/tests/test_packaging.py.
6. Verify with pytest and pyright digohwelisgi.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented language-agnostic audio dataset packager in digohwelisgi.core.datasets.
- Defined immutable models: DatasetSplit, PackagedAudioRecord, PackagingManifest, PackagingConfig, PackagingResult.
- Implemented pure package_audio_dataset procedure with audio isolation, SHA256 checksumming, header inspection, and manifest serialization.
- Implemented upload_to_huggingface_hub for automated dataset repository management.
- Added comprehensive unit tests and verified with pyright (0 errors) and pytest.
<!-- SECTION:FINAL_SUMMARY:END -->
