---
id: TASK-206
title: Build desktop application main entrypoint and PyInstaller packaging spec
status: Done
assignee: []
created_date: '2026-08-12 21:17'
updated_date: '2026-08-12 21:27'
labels: []
dependencies: []
ordinal: 202000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create syllabary_transcriber/__main__.py desktop launcher that loads built React dist assets into pywebview window, and configure PyInstaller packaging spec for standalone binary distribution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 python -m syllabary_transcriber launches pywebview desktop window with bundled React app
- [x] #2 PyInstaller spec file provided under syllabary_transcriber/packaging/ specifying assets, PyWebView binaries, and macOS Info.plist permissions
- [x] #3 App builds and executes standalone desktop window
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created __main__.py desktop application launcher and PyInstaller packaging spec with macOS microphone entitlement plist settings.
<!-- SECTION:FINAL_SUMMARY:END -->
