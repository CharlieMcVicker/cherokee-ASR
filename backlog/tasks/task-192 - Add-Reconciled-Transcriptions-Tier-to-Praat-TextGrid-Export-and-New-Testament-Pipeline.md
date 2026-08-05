---
id: TASK-192
title: >-
  Add Reconciled Transcriptions Tier to Praat TextGrid Export and New Testament
  Pipeline
status: Done
assignee:
  - '@agent-k'
created_date: '2026-08-05 12:49'
updated_date: '2026-08-05 12:53'
labels: []
dependencies: []
ordinal: 188000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Integrate phonological syllabary/ASR reconciliation into align_cli and export_praat_textgrid, and verify by running on the Book of Mark and launching Praat with audio.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add --reconcile flag to align_cli.py and run_alignment_pipeline
- [x] #2 Update export_praat_textgrid to output Reconciled Words tier when available
- [x] #3 Update transcription.new_testament.pipeline to enable reconciliation
- [x] #4 Run alignment on Book of Mark audio and launch Praat with audio script
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect aligner.py, exporter.py, align_cli.py, and new_testament/pipeline.py\n2. Add reconciled word/segment fields to AlignmentResult in aligner.py\n3. Update aligner / align_cli to compute reconciled phonetics when reconcile=True\n4. Update exporter.py to output 'Reconciled Words' tier in Praat TextGrid\n5. Update transcription/new_testament/pipeline.py to pass reconcile=True\n6. Locate Book of Mark audio and metadata, run alignment pipeline\n7. Launch Praat script/app to view audio + TextGrid
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Integrated phonological reconciliation directly into alignment pipeline (align_cli.py, aligner.py, exporter.py, and new_testament/pipeline.py). Added 'Reconciled Words' tier (Tier 4) to output Praat TextGrids. Successfully ran alignment on Mark Chapter 1 and loaded audio and TextGrid in Praat.
<!-- SECTION:FINAL_SUMMARY:END -->
