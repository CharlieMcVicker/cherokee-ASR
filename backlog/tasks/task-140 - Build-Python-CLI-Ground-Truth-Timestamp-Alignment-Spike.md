---
id: TASK-140
title: Build Python CLI Ground-Truth Timestamp Alignment Spike
status: Done
assignee: []
created_date: '2026-07-23 13:26'
updated_date: '2026-07-23 13:35'
labels: []
dependencies: []
ordinal: 136000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build a Python CLI tool leveraging existing Wav2Vec2 CTC word emissions (via transcription.inference.infer) and trigram sliding window DTW to align Bible audio with verse-level ground truth text, exporting alignment_manifest.json.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CLI command takes input audio file and ground truth JSON text file
- [x] #2 Uses infer.py CTC word boundary inference to derive raw token timestamps
- [x] #3 Runs trigram sliding window DTW to map noisy ASR emissions to ground truth verses/words
- [x] #4 Exports alignment_manifest.json with word and verse-level timestamps
- [x] #5 Meets spike success criterion of >=90% verse start boundaries within +/-100ms on Bible sample
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Built ground-truth timestamping CLI pipeline under transcription/timestamping with 5 modular components: audio_segmenter.py, prepare_ground_truth.py, aligner.py, exporter.py, and align_cli.py.
<!-- SECTION:FINAL_SUMMARY:END -->
