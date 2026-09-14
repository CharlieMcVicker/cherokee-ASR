---
id: TASK-321
title: >-
  Implement Phonotactic CTC Segmentation, Character Anomaly Detection, and
  Calibration
status: To Do
assignee: []
created_date: '2026-09-14 13:29'
labels: []
dependencies: []
documentation:
  - docs/spec_per_token_intrusion_and_site_masks.md
  - docs/alignment.md
modified_files:
  - transcription/alignment/ctc_aligner.py
  - transcription/alignment/pipeline.py
  - transcription/alignment/realign_bible.py
  - transcription/alignment/phonotactics.py
  - tests/test_ctc_aligner.py
  - tests/test_pipeline.py
priority: high
type: feature
ordinal: 337000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
### Objective & Context
Upgrade the Cherokee CTC segmentation alignment pipeline from coarse global scalar penalties to phonotactically-aware per-token penalties and site masks, add character-level minimum acoustic confidence anomaly detection, and calibrate optimal parameters on Mark Chapter 1 without running full New Testament realignment (deferred to TASK-320).

### Key Architecture & Components
1. **Phonotactics Engine**: `transcription/alignment/phonotactics.py` (`prepare_cherokee_text`) applies Cherokee phonological rules, site-specific intrusion masks (`is_intrusive_site`), and syncope masks (`is_syncope_token`).
2. **CTC Segmentation Integration**: `transcription/alignment/ctc_aligner.py` and `transcription/alignment/pipeline.py` configure `ctc_segmentation.CtcSegmentationParameters` with `intrusive_penalties`, `intrusive_min_logprobs`, `is_intrusive_site`, and `is_syncope_token`.
3. **Character-Level Anomaly Detection**: Upgrade `CTCSegmentationAligner` anomaly detection to evaluate per-character minimum acoustic probabilities alongside geometric mean confidence to reliably catch isolated typo corruptions (e.g., Mark 1:1 `yihstv` typo).
4. **Calibration & Benchmarking**: Use `transcription/alignment/benchmark_ctc_segmentation_100_verses.py` and Mark Chapter 1 grid search to optimize penalties for /h/ (3.0-5.0), glottal stop /'/ (0.5-1.2), and min-logprob thresholds.

### Execution Subtask Sequence for Feature Pipeline
- **TASK-318**: Wire phonotactic text preparation and PR #6 parameters into CTCSegmentationAligner, pipeline.py, and realign_bible.py CLI.
- **TASK-308**: Add minimum character confidence thresholding to CTCSegmentationAligner anomaly detection.
- **TASK-319**: Calibrate per-token intrusion penalties and min-logprobs on Mark Chapter 1 and evaluate against benchmark.
- *(Superseded: TASK-307 & TASK-316 are incorporated into TASK-318/TASK-319; Out of Scope: TASK-320 full bible run)*.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 CTCSegmentationAligner and pipeline.py use prepare_cherokee_text with is_intrusive_site and is_syncope_token masks
- [ ] #2 Per-token intrusive penalties and intrusive min logprobs are configurable and passed to CtcSegmentationParameters
- [ ] #3 Word anomaly detection in CTCSegmentationAligner flags single-letter transcript typos (e.g., Mark 1:1 yihstv) using character-level minimum confidence
- [ ] #4 Calibration grid search on Mark Chapter 1 determines optimal default penalties and logprobs without spurious breath insertions
- [ ] #5 Unit and regression tests pass in test_ctc_aligner.py, test_pipeline.py, and test_phonotactics.py
<!-- AC:END -->
