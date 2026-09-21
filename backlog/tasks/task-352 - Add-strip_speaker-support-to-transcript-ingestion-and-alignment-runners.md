---
id: TASK-352
title: Add strip_speaker support to transcript ingestion and alignment runners
status: Done
assignee:
  - '@antigravity'
created_date: '2026-09-18 18:53'
updated_date: '2026-09-18 18:56'
labels: []
dependencies: []
ordinal: 378000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enable optional stripping of speaker prefix labels (e.g. Guy Soldier:, Mary:, ᏣᎵ:) in load_syllabary_transcript, align_syllabary_ctc, and align_syllabary_greedy so speaker tags are recorded in metadata without polluting acoustic alignment targets.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 load_syllabary_transcript accepts strip_speaker parameter and extracts speaker metadata while aligning spoken text
- [x] #2 align_syllabary_ctc and align_syllabary_greedy forward strip_speaker parameter
- [x] #3 realign_gs_mm_ctc.py runs with strip_speaker=True utilizing cached logits to produce clean alignment without speaker names
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update load_syllabary_transcript in ingestion.py to accept strip_speaker parameter and strip speaker labels from alignment targets while preserving them in source metadata.
2. Forward strip_speaker in align_syllabary_ctc and align_syllabary_greedy in pipeline.py.
3. Update realign_gs_mm_ctc.py with strip_speaker=True and re-run alignment using cached logprobs.
4. Verify tests pass and inspect updated manifest.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added strip_speaker option to load_syllabary_transcript, align_syllabary_ctc, and align_syllabary_greedy. Realigned the 56-minute gs_mm interview in 37 seconds using cached logits, cleanly stripping speaker tags from the alignment target while preserving speaker labels in turn metadata.
<!-- SECTION:FINAL_SUMMARY:END -->
