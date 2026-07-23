---
id: TASK-144
title: Build unified CLI entrypoint for ground-truth timestamping
status: Done
assignee:
  - '@agent-k'
created_date: '2026-07-23 13:30'
updated_date: '2026-07-23 13:35'
labels: []
dependencies: []
ordinal: 140000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create transcription/alignment/align_cli.py orchestrating prepare_ground_truth, aligner, and exporter into a runnable CLI pipeline.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Runs full pipeline end-to-end from single terminal command
- [x] #2 Exports Praat TextGrid and alignment_manifest.json to specified output directory
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create transcription/timestamping/align_cli.py\n2. Integrate audio_segmenter, prepare_ground_truth, aligner, and exporter\n3. Provide argparse CLI interface for --audio, --metadata, --output-dir, and --export-praat\n4. Test CLI execution on mark_01_metadata.json
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created transcription/timestamping/align_cli.py orchestrating audio_segmenter, prepare_ground_truth, aligner, and exporter. Verified end-to-end execution producing alignment_manifest.json and alignment.TextGrid.
<!-- SECTION:FINAL_SUMMARY:END -->
