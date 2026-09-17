---
id: TASK-347
title: >-
  Disable flag_min_char_confidence default in CTCAlignerConfig to rely on word
  confidence
status: Done
assignee:
  - '@self'
created_date: '2026-09-17 18:24'
updated_date: '2026-09-17 18:26'
labels: []
dependencies: []
ordinal: 363000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Set flag_min_char_confidence default to 0.0 in CTCAlignerConfig to prevent spurious anomaly flagging from natural glottal smoothing and de-aspiration, relying on word confidence (flag_min_confidence=0.05) for anomaly detection.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CTCAlignerConfig default flag_min_char_confidence is set to 0.0
- [x] #2 Unit tests and documentation in docs/alignment.md reflect the 0.0 default
- [x] #3 All pytest tests and pyright pass cleanly
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update CTCAlignerConfig.flag_min_char_confidence default to 0.0 in transcription/alignment/models.py.\n2. Update tests in test_models_and_metrics.py and documentation in docs/alignment.md.\n3. Run pytest and pyright to ensure full test suite passes.\n4. Realign Mark 1 and verify updated manifest and statistics.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Disabled flag_min_char_confidence default in CTCAlignerConfig (0.0005 -> 0.0) to eliminate false-positive anomaly flags on natural continuous speech reductions (e.g. glottal transitions and de-aspiration). Updated tests, documentation in docs/alignment.md, and verified Mark chapter 1 realignment: flagged verses dropped from 23 (51.1%) to 8 (17.8%), and flagged words dropped from 139 (26.1%) to 27 (5.1%), all corresponding to genuine low-confidence word anomalies (< 0.05).
<!-- SECTION:FINAL_SUMMARY:END -->
