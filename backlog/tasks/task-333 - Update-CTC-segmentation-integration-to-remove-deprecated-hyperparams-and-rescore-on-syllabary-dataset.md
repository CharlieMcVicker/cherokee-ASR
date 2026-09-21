---
id: TASK-333
title: >-
  Update CTC segmentation integration to remove deprecated hyperparams and
  rescore on syllabary dataset
status: Done
assignee:
  - '@agent'
created_date: '2026-09-16 15:43'
updated_date: '2026-09-16 15:51'
labels: []
dependencies: []
ordinal: 349000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update CTCAlignerConfig and CTCSegmentationAligner to remove tuned hyperparameters (syncope_penalty, intrusive_penalties, intrusive_min_logprobs) following ctc-segmentation PR #7. Rescore clean and noisy performance against the dataset with syllabary comparing greedy inference vs syllabary-guided alignment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Remove deprecated hyperparameters (syncope_penalty, intrusive_penalty, intrusive_penalties, intrusive_min_logprobs) from CTCAlignerConfig and CTCSegmentationAligner
- [x] #2 Update unit tests and alignment callers to work cleanly with the updated ctc_segmentation API
- [x] #3 Run evaluation comparing clean and noised audio performance using greedy ASR inference vs syllabary-guided CTC segmentation on the dataset
- [x] #4 Verify all unit and regression tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update CTCAlignerConfig and CTCSegmentationAligner to remove deprecated hyperparameters (syncope_penalty, intrusive_penalty, intrusive_penalties, intrusive_min_logprobs).
2. Update unit tests and callers across transcription/alignment and tests.
3. Build and run evaluation script on split_audio_syl_target.csv comparing clean and pink-noise perturbed (18 dB SNR) audio for greedy ASR inference vs syllabary-guided CTC segmentation.
4. Verify pytest suite and pyright type checks pass.
5. Summarize evaluation metrics and performance comparison.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated CTCAlignerConfig and CTCSegmentationAligner to remove deprecated tuning hyperparameters (syncope_penalty, intrusive_penalty, intrusive_penalties, intrusive_min_logprobs) in accordance with ctc-segmentation PR #7 Relative Contrastive Gating. Created scripts/rescore_syllabary_dataset.py and evaluated all 1,389 samples across test, valid, train, and full dataset for clean and pink-noise perturbed (18 dB SNR) audio. Verified all 273 pytest tests and pyright type checking pass.
<!-- SECTION:FINAL_SUMMARY:END -->
