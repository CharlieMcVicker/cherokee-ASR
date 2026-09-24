---
id: TASK-369
title: >-
  Modularize ARPAbet cross-lingual loanword projection into Tier 1 generic
  engine and Tier 2 Cherokee factory
status: Done
assignee:
  - '@supervisor'
created_date: '2026-09-24 14:52'
updated_date: '2026-09-24 15:21'
labels: []
dependencies: []
ordinal: 399300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extract the language-agnostic cross-lingual loanword projection pipeline (English G2P, MMS forced aligner, dynamic programming aligners, acoustic confusion matrix trainer, and Viterbi joint-ngram projector) into Tier 1 core. Tier 2 cherokee provides thin factory functions parameterized with Cherokee T/TH acoustic artifacts, enabling any downstream language model to reuse the cross-lingual projection engine.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Tier 1 owns language-agnostic ARPAbet types, English G2P, MMS forced aligner, DP aligners, matrix trainer, and generic SyntheticTargetProjector
- [x] #2 Tier 2 provides make_cherokee_projector factory and retains Cherokee compound clitic segmentation
- [x] #3 Prune legacy transcription/cherokee/arpabet and update all callers and pipelines
- [x] #4 All tests pass and pyright returns 0 errors
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Successfully modularized the ARPAbet cross-lingual loanword projection engine into Tier 1 (transcription.core.codeswitching & transcription.core.alignment) and Tier 2 (transcription.cherokee.codeswitching). Pruned legacy transcription/cherokee/arpabet/, rewired all pipeline/CLI call sites, and verified 358 pytest unit tests pass and 0 pyright errors.
<!-- SECTION:FINAL_SUMMARY:END -->
