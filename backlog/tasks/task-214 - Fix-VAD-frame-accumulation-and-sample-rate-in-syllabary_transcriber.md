---
id: TASK-214
title: Fix VAD frame accumulation and sample rate in syllabary_transcriber
status: Done
assignee:
  - '@agent'
created_date: '2026-08-12 21:58'
updated_date: '2026-08-12 21:59'
labels: []
dependencies: []
ordinal: 205000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix issue where frames before speech_start are dropped, minimum speech duration check is missing, and AudioContext sample rate fallback needs explicit handling for 16kHz ASR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Buffer audio leading up to speech start in vad-processor.js
- [x] #2 Send recorded float32 PCM frames to transcribe_pcm when speech ends
- [x] #3 Enforce 16kHz sampling rate across AudioContext and Python backend
- [x] #4 Verify transcription UI receives and renders output
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed VAD frame retention, sample rate configuration, and PCM frame transmission for transcription.
<!-- SECTION:FINAL_SUMMARY:END -->
