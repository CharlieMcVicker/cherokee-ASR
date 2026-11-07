---
id: TASK-401
title: 'Phase 3: Dual-Stream Dataset Preparation & Vocabulary Serialization'
status: Done
assignee:
  - '@self'
created_date: '2026-11-07 21:25'
updated_date: '2026-11-07 22:08'
labels: []
dependencies:
  - TASK-400
modified_files:
  - digohwelisgi/cherokee/orthography/tones.py
  - digohwelisgi/cherokee/orthography/__init__.py
  - digohwelisgi/cherokee/training/prepare_csv.py
  - scripts/split_bible_dataset.py
  - scripts/fetch_training_data.py
  - scripts/upload_labels_to_hf.py
  - digohwelisgi/cherokee/training/tests/test_dual_dataset_pipeline.py
ordinal: 434300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Regenerate Cherokee dictionary and Bible Wav2Vec2 CSV splits with Standard 2 cv_text, tone_tokens, and canonical_toned columns, serialize 19-token vocab.json and 10-token vocab_tone.json, and add safe remote fetching flags.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 chr-wav2vec2-{train,valid,test}.csv are regenerated with path,cv_text,tone_tokens,canonical_toned and 0 colons or leading hs-
- [x] #2 bible-wav2vec2-{train,valid,test}.csv are updated with path,cv_text,tone_tokens (empty tone_tokens) and Standard 2 normalization
- [x] #3 vocab.json (19 tokens) and vocab_tone.json (10 attested moraic tokens) encode and decode bidirectionally with 100% dataset coverage
- [x] #4 scripts/fetch_training_data.py supports --preserve-local-csvs and scripts/upload_labels_to_hf.py provides lightweight label syncing
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update prepare_csv.py and scripts/split_bible_dataset.py to emit Standard 2 dual-stream CSVs and regenerate splits
2. Implement bidirectional CV & Moraic Tone vocabulary serialization and tokenization helpers
3. Update scripts/fetch_training_data.py (--preserve-local-csvs) and create scripts/upload_labels_to_hf.py
4. Add unit tests in digohwelisgi/cherokee/training/tests/test_dual_dataset_pipeline.py and pass Integration Gate 3 audit
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed Phase 3 Dual-Stream Dataset Preparation & Vocabulary Serialization.
- Regenerated chr-wav2vec2-{train,valid,test}.csv (1,482 / 186 / 185 rows) with path,cv_text,tone_tokens,canonical_toned and split_audio_syl_target.csv.
- Regenerated bible-wav2vec2-{train,valid,test}.csv (939 / 105 / 100 rows) with Standard 2 cv_text and empty tone_tokens.
- Serialized vocab.json (19 tokens) and vocab_tone.json (10 attested tokens) and added bidirectional encode/decode helpers.
- Added --preserve-local-csvs to scripts/fetch_training_data.py and created scripts/upload_labels_to_hf.py.
- Passed Integration Gate 3 audit (11/11 tests pass).
<!-- SECTION:FINAL_SUMMARY:END -->
