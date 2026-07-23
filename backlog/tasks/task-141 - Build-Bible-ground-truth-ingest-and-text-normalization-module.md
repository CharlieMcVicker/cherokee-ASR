---
id: TASK-141
title: Build Bible ground truth ingest and text normalization module
status: Done
assignee:
  - '@agent-k'
created_date: '2026-07-23 13:30'
updated_date: '2026-07-23 13:34'
labels: []
dependencies: []
ordinal: 137000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create transcription/alignment/prepare_ground_truth.py to parse mark_01_metadata.json, strip hyphens, apply consonant respelling via tone_normalization.py, drop /h/s and punctuation, and output clean normalized verse structures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Parses mark_01_metadata.json into structured verse entries
- [x] #2 Applies respell_consonants to map t->th, d->t, qu->gw
- [x] #3 Strips hyphens, /h/ sound markers, and punctuation for ASR matching
- [x] #4 Outputs normalized ground-truth JSON schema for aligner
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create transcription/timestamping/prepare_ground_truth.py\n2. Parse Bible metadata JSON (e.g. mark_01_metadata.json)\n3. Implement normalize_text_for_alignment: hyphen stripping, respell_consonants (t->th, d->t, qu->gw), drop /h/s, and strip punctuation\n4. Write unit tests in test_prepare_ground_truth.py
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created transcription/timestamping/prepare_ground_truth.py module with parse_bible_metadata and normalize_text_for_alignment. Tested in test_prepare_ground_truth.py.
<!-- SECTION:FINAL_SUMMARY:END -->
