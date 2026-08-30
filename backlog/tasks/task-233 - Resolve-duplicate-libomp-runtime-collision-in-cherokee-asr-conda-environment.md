---
id: TASK-233
title: Resolve duplicate libomp runtime collision in cherokee-asr conda environment
status: Done
assignee:
  - '@antigravity'
created_date: '2026-08-30 22:05'
updated_date: '2026-08-30 22:06'
labels: []
dependencies: []
ordinal: 227000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix libomp.dylib duplicate initialization error by removing conflicting conda ffmpeg and llvm-openmp packages and switching OpenBLAS to the pthreads build while keeping host FFmpeg 8.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Remove conflicting conda ffmpeg and llvm-openmp packages
- [x] #2 Install libopenblas pthreads build to prevent duplicate libomp initialization
- [x] #3 Verify torch and ffmpeg work without libomp duplicate runtime crash
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed conflicting conda-forge ffmpeg and llvm-openmp packages from cherokee-asr conda environment and switched libopenblas to the thread-safe pthreads build. Verified PyTorch, Torchaudio, Torchvision, NumPy, and FFmpeg 8.1.2 operate without any duplicate libomp initialization errors.
<!-- SECTION:FINAL_SUMMARY:END -->
