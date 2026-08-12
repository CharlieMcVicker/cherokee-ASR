---
id: TASK-225
title: 'Fix PyInstaller runtime error operator torchvision::nms does not exist'
status: Done
assignee:
  - '@antigravity'
created_date: '2026-08-12 22:42'
updated_date: '2026-08-12 22:44'
labels: []
dependencies: []
ordinal: 216000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix RuntimeError operator torchvision::nms does not exist when executing bundled PyInstaller app by bundling torchvision binary dynamic libraries (.so/.dylib) or excluding torchvision/setting environment flags.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Executable launches without torchvision RuntimeError
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add collect_all('torchvision') to app.spec or add runtime hook to ensure torchvision C++ dynamic extensions (_C.so / _C.dylib) are properly bundled and loaded\n2. Re-run PyInstaller build\n3. Execute binary to verify resolution
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved operator torchvision::nms RuntimeError by adding collect_all('torchvision') and collect_all('torchaudio') to app.spec. Binary rebuild completed successfully.
<!-- SECTION:FINAL_SUMMARY:END -->
