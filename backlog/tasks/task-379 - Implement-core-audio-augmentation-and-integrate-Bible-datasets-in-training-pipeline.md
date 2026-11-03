---
id: TASK-379
title: >-
  Implement core audio augmentation and integrate Bible datasets in training
  pipeline
status: Done
assignee:
  - '@self'
created_date: '2026-11-02 21:25'
updated_date: '2026-11-02 21:31'
labels: []
dependencies: []
modified_files:
  - digohwelisgi/core/audio/augmentation.py
  - digohwelisgi/core/audio/__init__.py
  - digohwelisgi/core/audio/tests/test_audio_augmentation.py
  - digohwelisgi/training/train.py
  - digohwelisgi/training/tests/test_train_pipeline.py
  - pyproject.toml
  - Dockerfile
  - uv.lock
  - AGENTS.md
ordinal: 412300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement language-agnostic live data augmentation (TimeStretch, PitchShift, AddGaussianNoise) using audiomentations in digohwelisgi.core.audio.augmentation, integrate with CTCTrainer and DataCollatorCTCWithPadding in digohwelisgi.training.train, resolve multi-domain audio paths for Bible datasets, port PyTorch 2.6+ safe globals, and update project dependencies and Dockerfile.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement AudioAugmentationConfig and AudioAugmenter in digohwelisgi.core.audio.augmentation with TimeStretch, PitchShift, and AddGaussianNoise
- [x] #2 Add unit tests in digohwelisgi/core/audio/tests/test_audio_augmentation.py verifying transforms and toggle stability
- [x] #3 Update train.py with CTCTrainer, DataCollatorCTCWithPadding augmentation toggle, CLI/CONFIG flags, and PyTorch 2.6+ safe globals
- [x] #4 Enhance _resolve_audio_path in train.py to resolve both original and Bible dataset audio paths
- [x] #5 Update pyproject.toml and Dockerfile to add audiomentations
- [x] #6 Verify all tests pass with pytest and static analysis passes with pyright digohwelisgi
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create digohwelisgi/core/audio/augmentation.py defining AudioAugmentationConfig and AudioAugmenter using audiomentations (TimeStretch, PitchShift, AddGaussianNoise)
2. Export augmentation components from digohwelisgi.core.audio
3. Write unit tests in digohwelisgi/core/audio/tests/test_audio_augmentation.py
4. Update digohwelisgi/training/train.py: DataCollatorCTCWithPadding with live augmentation, CTCTrainer with train/eval toggle, parse_args/CONFIG CLI options, PyTorch 2.6+ safe globals, and robust _resolve_audio_path supporting Bible audio
5. Add audiomentations to pyproject.toml and Dockerfile
6. Run pytest and pyright digohwelisgi to verify tests and typing
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented language-agnostic audio data augmentation layer in core and integrated multi-domain Bible datasets into the Wav2Vec2 training pipeline:
1. Created digohwelisgi/core/audio/augmentation.py containing AudioAugmentationConfig and AudioAugmenter using audiomentations (TimeStretch, PitchShift, AddGaussianNoise) and exported them via digohwelisgi.core.audio.
2. Created digohwelisgi/core/audio/tests/test_audio_augmentation.py with unit tests for default config, no-op passthrough, Gaussian noise injection, and TimeStretch/PitchShift waveform modification.
3. Enhanced digohwelisgi/training/train.py:
   - Added PyTorch 2.6+ safe globals for weights_only=True checkpoint resumption.
   - Refactored DataCollatorCTCWithPadding to support live on-the-fly audio augmentation using AudioAugmenter.
   - Introduced CTCTrainer(Trainer) subclass to automatically enable augmentation during get_train_dataloader() and disable during get_eval_dataloader().
   - Added CLI arguments (--augment / --no-augment, probabilities, min/max ranges) and synced them with CONFIG.
   - Enhanced _resolve_audio_path to seamlessly resolve paths across data/projects/cherokee_new_testament/, legacy training_data/, and data/training/processed/sentence_audio (verified 0 missing across all 5,231 samples across all 6 splits).
   - Ensured evaluate_checkpoints_dual explicitly disables collator augmentation for deterministic evaluation.
4. Created digohwelisgi/training/tests/test_train_pipeline.py testing multi-domain audio path resolution, DataCollatorCTCWithPadding augment mode toggle, and CTCTrainer train/eval dataloader hooks.
5. Added audiomentations to pyproject.toml, Dockerfile, and synchronized uv.lock.
6. Updated AGENTS.md Tier 1 core audio documentation.
7. Verified full pytest suite passes (366 passed) and pyright static analysis passes with 0 errors.
<!-- SECTION:FINAL_SUMMARY:END -->
