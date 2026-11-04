---
id: TASK-392
title: Update best_model.json to checkpoint-8000 and create best_model_pre_bible.json
status: Done
assignee:
  - '@self'
created_date: '2026-11-04 16:06'
updated_date: '2026-11-04 16:08'
labels: []
dependencies: []
modified_files:
  - best_model.json
  - best_model_pre_bible.json
  - digohwelisgi/utils/model_utils.py
  - digohwelisgi/cherokee/models/loader.py
ordinal: 425300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update best_model.json to point to the winning checkpoint-8000 (charliemcvicker/bible-augmentation-checkpoint-8000-wav2vec2-large-xlsr) and preserve the canonical pre-Bible baseline model configuration in best_model_pre_bible.json.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Update best_model.json with repo charliemcvicker/bible-augmentation-checkpoint-8000-wav2vec2-large-xlsr and revision 1dd927526fd11671ee41890ae393fa6a7c527414
- [x] #2 Create best_model_pre_bible.json containing charliemcvicker/length-only-20260704-155307-asr-cherokee-colon @ 76e62140955f4738abdab345ea34068b02d8d2a2
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated best_model.json to point to checkpoint-8000 (charliemcvicker/bible-augmentation-checkpoint-8000-wav2vec2-large-xlsr @ 1dd927526fd11671ee41890ae393fa6a7c527414). Preserved the baseline pre-Bible model configuration in best_model_pre_bible.json (charliemcvicker/length-only-20260704-155307-asr-cherokee-colon @ 76e62140955f4738abdab345ea34068b02d8d2a2). Added optional config_filename parameter to get_best_model_config and CherokeeASRModel.get_best_model.
<!-- SECTION:FINAL_SUMMARY:END -->
