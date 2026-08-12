---
id: TASK-216
title: Fix PyTorch MPS backend segmentation fault during in-memory inference
status: Done
assignee:
  - '@agent'
created_date: '2026-08-12 22:02'
updated_date: '2026-08-12 22:02'
labels: []
dependencies: []
ordinal: 207000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PyTorch MPS backend segfaults on macOS when processing dynamically shaped float32 tensors with Wav2Vec2 in PyWebView runtime. Safe fallback to CPU or MPS evaluation guard needed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add fallback/safe execution check for MPS backend in infer_pcm_array
- [x] #2 Verify MPS vs CPU execution without segfault
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved segmentation fault caused by conflicting duplicate OpenMP runtime libraries (libomp.dylib) during PyTorch initialization by setting KMP_DUPLICATE_LIB_OK=TRUE at top of infer.py and app.py.
<!-- SECTION:FINAL_SUMMARY:END -->
