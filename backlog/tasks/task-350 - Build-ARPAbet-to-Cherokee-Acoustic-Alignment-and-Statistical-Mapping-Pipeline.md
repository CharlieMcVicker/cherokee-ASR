---
id: TASK-350
title: Build ARPAbet-to-Cherokee Acoustic Alignment and Statistical Mapping Pipeline
status: Done
assignee: []
created_date: '2026-09-18 14:44'
updated_date: '2026-09-18 18:27'
labels:
  - alignment
  - phonetics
  - arpabet
  - asr
dependencies: []
documentation:
  - docs/spec_arpabet_alignment_plan.md
priority: high
type: feature
ordinal: 369000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build an acoustic alignment and empirical statistical mapping pipeline that enables our Cherokee ASR engine to align code-switched transcripts containing English words. Slices 5,000 phonetically balanced 16kHz mono English word clips from LibriSpeech, feeds them through CherokeeASRModel to record emitted Cherokee tokens and Top-3 hypotheses, learns an empirical transition matrix P(Cherokee | ARPAbet) via Expectation-Maximization and DP traceback, and projects English words into synthetic Cherokee phonetic targets at runtime.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Parent tracking epic for all 5 ARPAbet alignment subtasks
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Delivered end-to-end ARPAbet-to-Cherokee acoustic alignment and statistical mapping pipeline across subtasks TASK-350.1 through TASK-350.7. Extracted 5,000 phonetically-balanced 16kHz mono word clips from LibriSpeech; performed batch forward-pass inference on the toneless pre-Bible model checkpoint; trained the empirical confusion matrix via confidence-weighted Numba Wagner-Fischer EM; built the runtime SyntheticTargetProjector with O(1) loanword memoization and dynamic G2P fallback; built the script-discriminating code-switched ground truth preparer preventing double-conversion; and realigned the 56.5-minute saving-the-voices/gs_mm interview achieving 100% matched chunk ratio (+35 chunks recovered over baseline) and +2,891 aligned characters. All 240 tests pass with 0 Pyright errors.
<!-- SECTION:FINAL_SUMMARY:END -->
