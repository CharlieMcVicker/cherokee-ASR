---
id: TASK-217
title: >-
  Disable CUDA and set PyTorch device flags before inference imports to prevent
  segfault in syllabary_transcriber
status: Done
assignee:
  - '@agent'
created_date: '2026-08-12 22:03'
updated_date: '2026-08-12 22:03'
labels: []
dependencies: []
ordinal: 208000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Disable CUDA/MPS via env vars prior to PyTorch imports to prevent C++ runtime thread segfaults when launched via Python or PyWebView.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Set CUDA_VISIBLE_DEVICES='' and PyTorch OMP env vars in __main__.py and app.py
- [x] #2 Verify python -m syllabary_transcriber runs without segfault
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Disabled CUDA (CUDA_VISIBLE_DEVICES='') and single-threaded OpenMP/MKL (OMP_NUM_THREADS=1, MKL_NUM_THREADS=1) at top of __main__.py, app.py, and infer.py before PyTorch imports to prevent segfaults when launching python -m syllabary_transcriber.
<!-- SECTION:FINAL_SUMMARY:END -->
