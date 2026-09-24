---
id: TASK-373
title: >-
  Migrate project audio and datasets to data/projects and update codebase
  callers
status: Done
assignee:
  - '@agent'
created_date: '2026-09-24 16:28'
updated_date: '2026-09-24 16:34'
labels: []
dependencies: []
ordinal: 406300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Multiple standalone project folders (saving-the-voices, cherokee_new_testament, cvcs-mp3s, audiofiles-to-transcribe), output directories (output_praat, output_w2v2, remote_output_w2v2, runs), and loose data files (charlie_shell_denoised.wav, mark_01_metadata.json, eval_results.json, wav-metadata.csv, match_cnt.py) reside at the root directory. Migrating these into structured subfolders under data/projects/ and data/ simplifies root navigation and groups assets with their project contexts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 saving-the-voices, cherokee_new_testament, cvcs-mp3s, and audiofiles-to-transcribe are relocated under data/projects/
- [x] #2 Loose root audio and metadata files are relocated to their corresponding data/projects or data/ subdirectories
- [x] #3 match_cnt.py is relocated to scripts/
- [x] #4 All Python scripts, pipelines, and test callers referencing relocated paths are updated
- [x] #5 All pipeline and unit tests pass with updated paths
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create data/projects/, data/runs/, data/models/, data/archive/ directories\n2. Move project directories (saving-the-voices, cherokee_new_testament, cvcs-mp3s, audiofiles-to-transcribe, output_praat) into data/projects/\n3. Move runs/ to data/runs/ and pretrained_models/ to data/models/pretrained_models/\n4. Move colab-script-rips, archive, and loose root wav/json files to data/archive/ and data/projects/saving-the-voices/\n5. Move match_cnt.py to scripts/match_cnt.py\n6. Update all referencing Python modules, scripts, tests, and Dockerfile\n7. Run pytest and verify all tests pass
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Relocated project directories (saving-the-voices, cherokee_new_testament, cvcs-mp3s, audiofiles-to-transcribe, output_praat) into data/projects/, moved runs/ to data/runs/, pretrained_models/ to data/models/, and legacy files/archives to data/archive/. Relocated match_cnt.py to scripts/. Updated all referencing scripts, pipelines, tests, and distance configurations to use updated paths. Verified with test suite (358 passed).
<!-- SECTION:FINAL_SUMMARY:END -->
