---
id: TASK-393
title: >-
  Recompute ARPAbet-to-Cherokee acoustic mapping with checkpoint-8000 and
  visualize differences
status: Done
assignee:
  - '@self'
created_date: '2026-11-04 16:13'
updated_date: '2026-11-04 16:20'
labels: []
dependencies: []
modified_files:
  - scripts/recompute_arpabet_matrix.py
  - >-
    data/arpabet_alignment/matrices/charliemcvicker_bible-augmentation-checkpoint-8000-wav2vec2-large-xlsr_1dd927526fd11671ee41890ae393fa6a7c527414_confusion_matrix.json
  - >-
    data/arpabet_alignment/visualizations/arpabet_comparison_prebible_vs_ckpt8000.png
ordinal: 426300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run batch inference over the 5,000 LibriSpeech word clips using checkpoint-8000, estimate the updated EM acoustic confusion matrix, and generate comparative visualizations contrasting pre-Bible vs checkpoint-8000 mappings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Compute emissions cache and train confusion matrix for checkpoint-8000
- [x] #2 Generate comparative visualizations contrasting phoneme mappings and probabilities against pre-Bible baseline
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Recomputed the cross-lingual ARPAbet-to-Cherokee acoustic confusion matrix using checkpoint-8000 via batch inference on 5,000 LibriSpeech clips (208 clips/sec on MPS) and 4 iterations of Expectation-Maximization. Generated comparative visualization figure at data/arpabet_alignment/visualizations/arpabet_comparison_prebible_vs_ckpt8000.png contrasting mappings, top-target probabilities, confidence shifts, and substitution distributions against the pre-Bible baseline.
<!-- SECTION:FINAL_SUMMARY:END -->
