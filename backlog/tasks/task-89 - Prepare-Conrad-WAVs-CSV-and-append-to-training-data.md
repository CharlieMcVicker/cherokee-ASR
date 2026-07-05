---
id: TASK-89
title: Prepare Conrad WAVs CSV and append to training data
status: Done
assignee:
  - '@agent'
created_date: '2026-07-03 15:57'
updated_date: '2026-07-03 16:06'
labels: []
dependencies: []
ordinal: 85000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Convert transcription column in conrad-wavs-all.csv to length-only (normalizing, removing accents, turning colons to doubled vowels) and append processed rows without headers to cim-wav2vec2-train.csv using prepare_conrad_csv.py.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 prepare_conrad_csv.py exists in transcription/training/
- [x] #2 Existing text normalization function is used to convert transcription column
- [x] #3 Processed conrad rows are appended to data/processed/cim-wav2vec2-train.csv with no header
- [x] #4 Verify appended rows in target CSV
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create transcription/training/prepare_conrad_csv.py next to prepare_csv.py.
2. Implement python code to import remove_tones_and_double_vowels from transcription.utils.tone_normalization, read data/processed/conrad-wavs-all.csv, apply normalization and cleaning to the transcription column, and append the processed rows to data/processed/cim-wav2vec2-train.csv (without headers).
3. Run the script using the python executable in the venv virtual environment.
4. Verify the output has been correctly appended to the target training file.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Regenerating train.csv splits with ʔ mapped to ' and ʼ, ‚ dropped, then rerunning the Conrad script.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created prepare_conrad_csv.py next to prepare_csv.py. The script imports remove_tones_and_double_vowels from transcription.utils.tone_normalization and clean_transcription from transcription.training.prepare_csv. It reads data/processed/conrad-wavs-all.csv, processes the transcription column, and appends the results without headers to data/processed/cim-wav2vec2-train.csv. Successfully ran the script and verified the output was appended.

Updated prepare_conrad_csv.py to drop ? and . from the text before normalization to avoid treating ? as a glottal stop. Restored the target training CSV and re-ran the updated script successfully.

Updated prepare_conrad_csv.py to drop left/right quotes, '‚', 'ʼ', and map 'ʔ' to '''. Verified that the appended Conrad data contains zero non-ASCII characters.

Modified prepare_csv.py to drop 'ʼ' and '‚' and map 'ʔ' to ''' before normalization. Regenerated the training splits and reran the Conrad preparation script. Confirmed zero non-ASCII characters remain in the entire train.csv file.

Identified that capitalized consonants (e.g., J, D) in the source text bypassed the consonant respelling rules in tone_normalization.py. Modified both prepare_csv.py and prepare_conrad_csv.py to convert raw text to lowercase prior to normalization. Successfully verified that all 'j's are now mapped to 'ts' (0 occurrences of j remain in the output) and capitalized D is correctly mapped to t (e.g., 'Daligwa' -> 'talikwa').
<!-- SECTION:FINAL_SUMMARY:END -->
