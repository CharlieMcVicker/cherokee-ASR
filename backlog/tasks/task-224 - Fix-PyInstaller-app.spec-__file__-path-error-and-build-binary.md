---
id: TASK-224
title: Fix PyInstaller app.spec __file__ path error and build binary
status: Done
assignee:
  - '@antigravity'
created_date: '2026-08-12 22:37'
updated_date: '2026-08-12 22:40'
labels: []
dependencies: []
ordinal: 215000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix NameError __file__ in PyInstaller app.spec when running pyinstaller command, bundle React static assets and Info.plist entitlements, and build executable binary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 app.spec runs without NameError
- [x] #2 PyInstaller builds executable without errors in dist
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Fix NameError in app.spec by replacing __file__ with SPECPATH provided by PyInstaller\n2. Add BUNDLE section to app.spec for macOS .app bundle output with Info.plist entitlements\n3. Execute pyinstaller build command and verify binary generation
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed NameError __file__ in PyInstaller app.spec by using PyInstaller SPECPATH. Built macOS app bundle and standalone executable successfully in transcriber_dist.
<!-- SECTION:FINAL_SUMMARY:END -->
