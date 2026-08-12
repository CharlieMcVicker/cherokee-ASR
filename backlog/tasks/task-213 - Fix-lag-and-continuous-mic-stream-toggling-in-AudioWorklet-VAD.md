---
id: TASK-213
title: Fix lag and continuous mic stream toggling in AudioWorklet VAD
status: Done
assignee:
  - '@agent-k'
created_date: '2026-08-12 21:55'
updated_date: '2026-08-12 21:56'
labels: []
dependencies: []
ordinal: 204000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The AudioWorklet VAD initialization in useVad is rapidly creating, tearing down, and recreating MediaStreams/AudioContexts causing microphone status flickering on macOS and high UI lag.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Ensure single continuous MediaStream and AudioContext during VAD lifecycle
- [x] #2 Eliminate mic status flickering and CPU/UI lag
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Decoupled inline callback dependencies in useVad.ts using refs and guarded initAudio against double invocation. Microphone stream and AudioContext remain continuous without rapid teardown/recreation.
<!-- SECTION:FINAL_SUMMARY:END -->
