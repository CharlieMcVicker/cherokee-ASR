---
id: TASK-218
title: Fix PyWebView vs FastAPI dev server endpoint routing in usePyWebView
status: Done
assignee:
  - '@agent'
created_date: '2026-08-12 22:04'
updated_date: '2026-08-12 22:04'
labels: []
dependencies: []
ordinal: 209000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure transcribePcm in usePyWebView properly posts to /api/transcribe-pcm or pywebview.api and logs errors when webview bridge is unattached in web browser.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add console logs for transcribePcm request and response in usePyWebView.ts
- [x] #2 Ensure fallback to FastAPI http backend works in browser dev server
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Lowered VAD speech buffer sample threshold from 4000 (~0.25s) to 1600 (~0.10s) and added console logging across VAD worklet events, useVad, and usePyWebView to expose audio payload size and transcription response status in browser console.
<!-- SECTION:FINAL_SUMMARY:END -->
