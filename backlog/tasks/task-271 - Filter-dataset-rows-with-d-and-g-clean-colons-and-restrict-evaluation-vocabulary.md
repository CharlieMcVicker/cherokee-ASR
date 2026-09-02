---
id: TASK-271
title: >-
  Filter dataset rows with d and g, clean colons, and restrict evaluation
  vocabulary
status: Done
assignee:
  - '@antigravity'
created_date: '2026-09-02 18:20'
updated_date: '2026-09-02 18:46'
labels:
  - evaluation
  - data-cleaning
  - normalization
dependencies: []
priority: high
type: bug
ordinal: 283000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Incorporate training loop text normalization (removing colons and punctuation via normalize_text) during evaluation dataset loading, filter out any dataset rows containing 'd' or 'g', and restrict ConfusionAccumulator vocabulary to the dataset's ground-truth alphabet so out-of-alphabet model tokens (e.g. d, g, ~, q) and colons do not pollute confusion matrix rows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Apply canonical normalize_text() to reference transcriptions in scripts/run_noisy_eval.py and evaluator.py
- [x] #2 Filter out any dataset rows containing 'd' or 'g' in reference transcriptions as bad data
- [x] #3 Restrict ConfusionAccumulator and cost engine vocabulary strictly to the active ground-truth dataset alphabet
- [x] #4 Filter empty tokens and out-of-alphabet tokens from top-K candidate accumulation
- [x] #5 Verify all evaluation tests pass and re-run evaluation sweep on clean dataset
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 normalize_text applied to all dataset references removing colons and punctuation
- [x] #2 Rows with [dg] in reference filtered out
- [x] #3 Confusion matrix vocabulary restricted to dataset alphabet
- [x] #4 All unit tests pass and clean evaluation sweep completes
<!-- DOD:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update load_dataset in scripts/run_noisy_eval.py to apply normalize_text() and filter out references containing 'd' or 'g'.
2. In extract_peaks_and_topk (evaluator.py), ensure blank / empty string tokens are omitted.
3. In ConfusionAccumulator (confusion.py), when a vocabulary is specified or derived, filter top-K updates to only tokens in vocab (or unk), and strictly use the reference vocabulary for the matrix.
4. Pass the cleaned dataset vocabulary to ConfusionAccumulator in run_noisy_eval.py.
5. Verify unit tests in transcription/evaluation/tests.
6. Re-run scripts/run_noisy_eval.py and verify clean outputs.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Applied canonical normalize_text() stripping all colons and punctuation from reference text. Filtered dataset rows containing 'd' or 'g'. Restricted ConfusionAccumulator and cost engine to the clean 17-character dataset alphabet, filtering blank and out-of-vocabulary model tokens from top-K candidates. Full training sweep (37,490 evaluations) completed with 0.88% white / 0.21% pink baseline CER at 25dB, and clean confusion matrix and cost artifacts generated.
<!-- SECTION:FINAL_SUMMARY:END -->
