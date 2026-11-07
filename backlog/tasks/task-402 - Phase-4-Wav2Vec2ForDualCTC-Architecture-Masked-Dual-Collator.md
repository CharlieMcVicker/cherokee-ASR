---
id: TASK-402
title: 'Phase 4: Wav2Vec2ForDualCTC Architecture & Masked Dual Collator'
status: Done
assignee:
  - '@self'
created_date: '2026-11-07 21:25'
updated_date: '2026-11-07 22:13'
labels: []
dependencies:
  - TASK-401
modified_files:
  - digohwelisgi/cherokee/models/dual_head.py
  - digohwelisgi/cherokee/models/__init__.py
  - digohwelisgi/cherokee/models/tests/test_dual_head_model.py
ordinal: 435300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the split-top Y-Network Wav2Vec2ForDualCTC model with configurable branch depth (default final 2 layers cloned from facebook/wav2vec2-large-xlsr-53), masked tone CTC loss, and DualDataCollatorCTCWithPadding.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Wav2Vec2ForDualCTC forks the final num_branch_layers (default 2) into independent CV and Moraic Tone transformer branches and linear CTC heads
- [x] #2 Forward pass computes CV CTC loss on all samples and masked Tone CTC loss strictly on tone-annotated samples
- [x] #3 DualDataCollatorCTCWithPadding dynamically pads input_values, labels (-100), and tone_labels (-100) with optional live audio augmentation
- [x] #4 Synthetic forward/backward tests verify non-zero gradients in trunk and both branches and zero tone loss contribution from masked Bible samples
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Implement Wav2Vec2DualCTCOutput, Wav2Vec2DualConfig, and Wav2Vec2ForDualCTC in digohwelisgi/cherokee/models/dual_head.py
2. Implement DualDataCollatorCTCWithPadding with dynamic padding for input_values, labels (-100), and tone_labels (-100) plus live audio augmentation
3. Export dual-head symbols in digohwelisgi/cherokee/models/__init__.py
4. Add unit tests in digohwelisgi/cherokee/models/tests/test_dual_head_model.py and run Integration Gate 4 verification
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Split-Layer Wav2Vec2ForDualCTC, Wav2Vec2DualCTCOutput, and DualDataCollatorCTCWithPadding in digohwelisgi/cherokee/models/dual_head.py and verified Gate 4 with pytest and pyright.
<!-- SECTION:FINAL_SUMMARY:END -->
