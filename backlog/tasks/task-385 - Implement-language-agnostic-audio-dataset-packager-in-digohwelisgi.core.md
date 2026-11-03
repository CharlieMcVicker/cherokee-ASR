---
id: TASK-385
title: Implement language-agnostic audio dataset packager in digohwelisgi.core
status: To Do
assignee: []
created_date: '2026-11-03 19:45'
labels: []
dependencies:
  - TASK-384
ordinal: 418300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement a generic, pure-function audio dataset packager and staging engine in Tier 1 that scans split CSVs, verifies 100% audio existence and integrity, stages only referenced audio into target structures, and writes dataset manifests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Define immutable domain types (DatasetSplit, PackagedAudioRecord, PackagingManifest) in Tier 1
- [ ] #2 Implement pure package_dataset procedure to isolate only referenced audio and generate manifests
- [ ] #3 Include optional Hugging Face Hub upload capability via huggingface_hub
- [ ] #4 Add unit tests verifying exact audio extraction, manifest generation, and missing-file rejection
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Unit tests pass and pyright digohwelisgi passes cleanly
<!-- DOD:END -->
