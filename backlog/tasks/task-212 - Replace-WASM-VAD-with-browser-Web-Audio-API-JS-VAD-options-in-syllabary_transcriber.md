---
id: TASK-212
title: >-
  Replace WASM VAD with browser Web Audio API JS VAD options in
  syllabary_transcriber
status: Done
assignee:
  - '@agent-k'
created_date: '2026-08-12 21:54'
updated_date: '2026-08-12 21:54'
labels: []
dependencies: []
ordinal: 203000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move from WASM VAD in syllabary_transcriber frontend to pure JS Web Audio API VAD options (Hark.js volume thresholding and/or Custom AudioWorkletProcessor VAD with energy/zero-crossing rates).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Replace WASM VAD implementation in syllabary_transcriber frontend
- [x] #2 Implement Hark or Custom AudioWorklet VAD for audio stream processing
- [x] #3 Verify voice activity detection works in browser without WebAssembly
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Replaced WASM Silero/ONNX VAD with pure JavaScript Web Audio API AudioWorklet processor (vad-processor.js). Removed ONNX/VAD WebAssembly NPM dependencies.
<!-- SECTION:FINAL_SUMMARY:END -->
