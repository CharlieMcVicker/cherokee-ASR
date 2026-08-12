---
id: TASK-219
title: Map ASR output text to Cherokee Syllabary in syllabary_transcriber backend
status: Done
assignee:
  - '@agent'
created_date: '2026-08-12 22:06'
updated_date: '2026-08-12 22:08'
labels: []
dependencies: []
ordinal: 210000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure infer_pcm_array enriches ASR phonetics into Cherokee Syllabary using transcription.syllabary_enrichment or syllabary mapper, returning both transcription and syllabary keys.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Enrich ASR phonetic output into Cherokee Syllabary in infer_pcm_array
- [x] #2 Return syllabary key in TranscribeResult dict
- [x] #3 Verify front-end receives and appends Cherokee Syllabary characters
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added phonetics_to_syllabary mapping engine in syllabary_map.py that splits phonetic text on vowel boundaries (a,e,i,o,u,v) using k/kh (ga/ka) and t/th conventions. Added automatic fallback to drop 'h' (e.g. thv -> tv -> Ꮫ). Attached syllabary field to inference output.
<!-- SECTION:FINAL_SUMMARY:END -->
