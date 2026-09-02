---
id: TASK-267
title: >-
  Persist multi-tier reconciled words and phonetic alignments in JSON alignment
  manifest
status: Done
assignee:
  - '@agent-manifest-export'
created_date: '2026-09-02 15:33'
updated_date: '2026-09-02 15:36'
labels:
  - alignment
  - exporters
  - manifest
dependencies: []
priority: medium
ordinal: 269000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend export_manifest in transcription/alignment/exporters.py to accept optional additional_word_tiers (including Reconciled Words) and persist reconciled word timestamps and syllabary text in alignment_manifest.json alongside standard words. Update CLI to pass additional_word_tiers to export_manifest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Extend export_manifest signature and serialization to accept additional_word_tiers
- [x] #2 Serialize reconciled words into alignment_manifest.json
- [x] #3 Update cli.py to pass additional_word_tiers to export_manifest
- [x] #4 Add unit tests in test_exporters.py and test_cli.py
- [x] #5 Verify all unit tests pass and pyright reports 0 errors
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Extended export_manifest with additional_word_tiers support and unit tests
<!-- SECTION:FINAL_SUMMARY:END -->
