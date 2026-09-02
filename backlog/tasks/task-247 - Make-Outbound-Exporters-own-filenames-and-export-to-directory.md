---
id: TASK-247
title: Make Outbound Exporters own filenames and export to directory
status: Done
assignee:
  - '@myself'
created_date: '2026-08-31 21:29'
updated_date: '2026-08-31 21:30'
labels: []
dependencies: []
ordinal: 249000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Refactor OutboundAlignmentAdapter protocol and all adapters (PraatTextGridAdapter, ManifestJsonAdapter, DebugJsonAdapter) to accept output_dir in export(alignment, output_dir) and own their filenames. Simplify AlignmentPipeline to accept exporters: Sequence[OutboundAlignmentAdapter].
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Update OutboundAlignmentAdapter protocol export signature to take output_dir.
- [x] #2 Update PraatTextGridAdapter, ManifestJsonAdapter, DebugJsonAdapter to own filename in __init__ and write to output_dir/filename.
- [x] #3 Update AlignmentPipeline to accept a flat sequence of exporters.
- [x] #4 Update cli.py and test suites, verifying all tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update OutboundAlignmentAdapter Protocol in protocols.py to export(alignment, output_dir).\n2. Update PraatTextGridAdapter, ManifestJsonAdapter, DebugJsonAdapter in outbound.py to accept optional filename in __init__ and write to output_dir/filename in export.\n3. Update AlignmentPipeline in pipeline.py to accept Sequence[OutboundAlignmentAdapter] and call exporter.export(alignment, output_dir).\n4. Update cli.py and all test files in transcription/alignment/tests/.\n5. Run pytest in cherokee-asr conda environment to verify.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Refactored OutboundAlignmentAdapter protocol and all adapters (PraatTextGridAdapter, ManifestJsonAdapter, DebugJsonAdapter) to accept output_dir in export(alignment, output_dir) and own their filenames at initialization. Updated AlignmentPipeline to accept a clean flat Sequence[OutboundAlignmentAdapter]. Updated cli.py and all test suites, verifying that all 86 unit tests pass.
<!-- SECTION:FINAL_SUMMARY:END -->
