---
id: TASK-358
title: >-
  Implement joint 1,2-gram ARPAbet-to-Cherokee phonetic confusion matrix and
  compare CER against unigram baseline
status: Done
assignee:
  - '@antigravity'
created_date: '2026-09-21 19:46'
updated_date: '2026-09-21 19:52'
labels: []
dependencies: []
ordinal: 384000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
1-to-1 unigram phonetic mapping cannot capture English loanword diphthongs (AY, AW, OW, OY, EY) and consonant clusters (T-SH, N-G, S-T). Extending AcousticConfusionMatrix and SyntheticTargetProjector to support joint 1,2-gram transitions with EM frequency accumulation and Viterbi DP projection improves alignment accuracy against Cherokee speech emissions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Extend AcousticConfusionMatrix and matrix estimation in transcription/alignment/arpabet/matrix.py to support 1-to-2, 2-to-1, and 2-to-2 transitions
- [x] #2 Update SyntheticTargetProjector in transcription/alignment/arpabet/projector.py to perform Viterbi / DP tiling over multi-gram transitions
- [x] #3 Ensure backward compatibility with existing serializations while supporting multigram tables
- [x] #4 Add comprehensive unit tests for 1,2-gram alignment, EM convergence, and projection
- [x] #5 Evaluate CER comparison between unigram and 1,2-gram models on the sample dataset and document results
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Domain Modeling: Update AcousticConfusionMatrix in types.py to support 1-gram and 2-gram source and target phone mappings with backward-compatible JSON serialization.
2. Seed & Alignment Estimation: Extend seed tables and DP traceback / EM accumulator in matrix.py to evaluate 1-to-1, 1-to-2, 2-to-1, and 2-to-2 transitions.
3. Viterbi DP Projection: Implement dynamic programming tiling in SyntheticTargetProjector.project_arpabet to select optimal multi-gram mappings.
4. Testing: Add unit tests for 2-gram alignment, EM estimation, and Viterbi projection; verify with pytest and pyright.
5. Evaluation: Train 1,2-gram matrix on 5,000-word dataset, measure CER vs unigram baseline, and verify improvement on diphthongs/digraphs.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented 1,2-gram joint multi-token support in AcousticConfusionMatrix, Wagner-Fischer DP traceback aligner, EM frequency accumulator, and SyntheticTargetProjector Viterbi dynamic programming. Verified with 246 passing unit tests, 0 pyright errors, and benchmarked CER across 4,992 words showing substantial accuracy improvements on diphthongs and consonant clusters.
<!-- SECTION:FINAL_SUMMARY:END -->
