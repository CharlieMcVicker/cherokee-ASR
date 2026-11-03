---
id: TASK-382
title: >-
  Migrate language code from cim to chr and add configurable colon stripping to
  training dataset preparation
status: Done
assignee:
  - '@self'
created_date: '2026-11-03 17:29'
updated_date: '2026-11-03 17:34'
labels: []
dependencies: []
modified_files:
  - digohwelisgi/training/prepare_csv.py
  - digohwelisgi/training/train.py
  - digohwelisgi/training/evaluate_checkpoint.py
  - digohwelisgi/training/evaluate_local_checkpoints.py
  - digohwelisgi/training/evaluate_revisions.py
  - digohwelisgi/training/prepare_conrad_csv.py
  - scripts/run_noisy_eval.py
  - scripts/generate_confusion_matrix.py
  - AGENTS.md
ordinal: 415300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Cherokee ISO-639-3 language code is "chr" (not Cook Islands Maori "cim"). Migrate all references, dataset filenames, and defaults from "cim" to "chr". In prepare_csv.py, add a --strip-colons / --no-strip-colons flag (defaulting to True) so that generated train/valid/test CSV splits are clean of colons on disk while preserving all original markings in the master sentence_audio.csv.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add --strip-colons / --no-strip-colons flag to prepare_csv.py (default True) to produce colon-free splits without mutating master sentence_audio.csv
- [x] #2 Regenerate training splits as chr-wav2vec2-{train,valid,test}.csv without colons and delete legacy cim-wav2vec2-*.csv files
- [x] #3 Migrate language code from cim to chr across train.py, evaluation scripts, driver scripts, and AGENTS.md
- [x] #4 Verify all unit tests and pyright pass with zero regressions
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update digohwelisgi/training/prepare_csv.py to support --strip-colons / --no-strip-colons (default True) and set default prefix to data/training/processed/chr-wav2vec2
2. Run prepare_csv.py to regenerate chr-wav2vec2-train.csv, chr-wav2vec2-valid.csv, and chr-wav2vec2-test.csv
3. Verify the generated chr-wav2vec2-*.csv files have zero colons, and delete legacy cim-wav2vec2-*.csv files
4. Update cim -> chr across train.py, evaluate_*.py, scripts, and AGENTS.md
5. Run pytest and pyright to ensure zero regressions
6. Commit changes, push to branch, update PR #10, and mark TASK-382 Done
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR-Style Summary:
- Added `--strip-colons` / `--no-strip-colons` CLI flag to `digohwelisgi/training/prepare_csv.py` (default: True), ensuring generated training splits omit vowel-length colons on disk while leaving master `sentence_audio.csv` untouched.
- Migrated language code from "cim" (Cook Islands Maori) to "chr" (Cherokee) across training configurations, evaluation scripts, drivers, and AGENTS.md.
- Regenerated dataset splits as `chr-wav2vec2-train.csv` (1,482 samples), `chr-wav2vec2-valid.csv` (186 samples), and `chr-wav2vec2-test.csv` (185 samples), verifying zero colons on disk.
- Deleted obsolete `cim-wav2vec2-*.csv` files.
- Confirmed full test suite passes (369/369) and pyright reports 0 errors.
<!-- SECTION:FINAL_SUMMARY:END -->
