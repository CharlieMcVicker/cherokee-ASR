---
id: TASK-226
title: Fix PyWebView WKWebView secure context restriction for navigator.mediaDevices
status: In Progress
assignee:
  - '@antigravity'
created_date: '2026-08-12 22:47'
updated_date: '2026-08-12 22:47'
labels: []
dependencies: []
ordinal: 217000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix undefined navigator.mediaDevices in PyWebView on macOS WKWebView due to non-secure localhost/file origin by configuring PyWebView webview instance, custom scheme, or secure origin header.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 navigator.mediaDevices is defined in pywebview
<!-- AC:END -->
