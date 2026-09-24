---
id: TASK-269.2
title: Implement audio perturbation transforms in perturbations.py
status: Done
assignee:
  - '@audio-engineer'
created_date: '2026-09-02 16:46'
updated_date: '2026-09-02 16:51'
labels:
  - evaluation
  - audio
  - perturbations
dependencies: []
parent_task_id: TASK-269
priority: high
type: feature
ordinal: 273000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement AudioTransform base class and composable transforms: AdditiveNoise (white, pink via spectral 1/sqrt(f), ambient with looping), TimeWarp with vocoder pitch preservation, AcousticFilter Butterworth bandpass (300-3400Hz), and ComposeTransforms.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Define AudioTransform abstract base class
- [x] #2 Implement AdditiveNoise supporting white, pink, and ambient noise with dynamic RMS calculation
- [x] #3 Implement TimeWarp with pitch preservation using phase vocoding / STFT time-stretch
- [x] #4 Implement AcousticFilter Butterworth 300-3400Hz using scipy.signal
- [x] #5 Implement ComposeTransforms pipeline
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented AudioTransform, AdditiveNoise, TimeWarp with vocoder pitch preservation, AcousticFilter, and ComposeTransforms.
<!-- SECTION:FINAL_SUMMARY:END -->
