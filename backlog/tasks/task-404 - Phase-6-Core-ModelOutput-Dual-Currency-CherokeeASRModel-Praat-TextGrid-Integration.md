---
id: TASK-404
title: >-
  Phase 6: Core ModelOutput Dual Currency, CherokeeASRModel & Praat TextGrid
  Integration
status: Done
assignee:
  - '@self'
created_date: '2026-11-07 21:25'
updated_date: '2026-11-07 22:42'
labels: []
dependencies:
  - TASK-403
modified_files:
  - digohwelisgi/core/models/output.py
  - digohwelisgi/core/models/inference.py
  - digohwelisgi/utils/model_utils.py
  - digohwelisgi/core/exporters/textgrid.py
  - digohwelisgi/cherokee/tests/test_dual_inference_e2e.py
  - dual_head_tone_asr_plan.md
ordinal: 437300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend ModelOutput currency with optional lpz_tone and vocab_tone projections (.decode_tone_tokens(), .decode_reconciled()), wire dual-head inference into CherokeeASRModel, and export Moraic Tone tiers in Praat TextGrids.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ModelOutput supports optional lpz_tone and vocab_tone with .decode_tone_tokens(), .decode_reconciled(mode), and .npz serialization
- [x] #2 CherokeeASRModel detects and loads Wav2Vec2ForDualCTC checkpoints and populates dual emissions in ModelOutput
- [x] #3 TextGrid exporter and enrichment/dialogue pipelines support exporting a dedicated Moraic Tone tier alongside Syllabary and TTH tiers
- [x] #4 End-to-end dual inference unit tests pass with 0 Pyright errors
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update ModelOutput in digohwelisgi/core/models/output.py with lpz_tone, vocab_tone, decode_tone_tokens, decode_reconciled, and .npz persistence.
2. Update digohwelisgi/core/models/inference.py and digohwelisgi/cherokee/models/loader.py to populate lpz_tone and vocab_tone for Wav2Vec2ForDualCTC models.
3. Update digohwelisgi/core/exporters/textgrid.py to support optional Moraic Tone tier.
4. Add digohwelisgi/cherokee/tests/test_dual_inference_e2e.py and verify with pytest and pyright.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Extended ModelOutput with lpz_tone, vocab_tone, decode_tone_tokens(), decode_reconciled(), and .npz persistence; wired Wav2Vec2ForDualCTC detection into get_model/CherokeeASRModel; added Moraic Tones tier to export_textgrid; verified with test_dual_inference_e2e.py and pyright.
<!-- SECTION:FINAL_SUMMARY:END -->
