---
id: TASK-350.2
title: >-
  Extract 5,000 Phonetically Balanced 16kHz Word Clips and Manifest from
  LibriSpeech
status: Done
assignee:
  - '@supervisor'
created_date: '2026-09-18 14:44'
updated_date: '2026-09-18 15:24'
labels:
  - alignment
  - dataset
  - librispeech
  - arpabet
dependencies:
  - TASK-350.1
parent_task_id: TASK-350
priority: high
type: feature
ordinal: 371000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Download and process LibriSpeech dev-clean to extract 5,000 single-word audio clips. Uses PyTorch torchaudio forced aligner in Python to extract word timestamps, applies +/-25ms acoustic boundary padding, saves files to disk as 16kHz mono PCM .wav files under data/arpabet_alignment/words/, balances word selection to prevent Zipfian overrepresentation of stopwords, and exports a durable words_manifest.json with stress-stripped g2p_en ARPAbet tokens.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Extract 5,000 word clips with +/-25ms boundary padding saved as 16kHz mono PCM .wav files on disk
- [x] #2 Apply phonetic balancing and frequency capping to ensure even distribution across all 39 ARPAbet phonemes
- [x] #3 Extract stress-stripped ARPAbet phoneme sequences via g2p_en
- [x] #4 Write durable words_manifest.json mapping clip_id, audio_path, word, duration, and arpabet tokens
- [x] #5 Provide automated tests validating dataset extraction, audio formatting, and manifest schema
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Build G2P extractor in transcription/alignment/arpabet/g2p.py wrapping g2p_en to produce clean stress-stripped ARPAbet tokens.
2. Build English word forced aligner in transcription/alignment/arpabet/forced_aligner.py using torchaudio MMS_FA / Wav2Vec2 forced aligner.
3. Build dataset extractor and balancer in transcription/alignment/arpabet/dataset.py to sample 5,000 words with phoneme frequency balancing (max 2 occurrences per lemma) and +/-25ms padding, saving 16kHz mono .wav files and writing words_manifest.json.
4. Add comprehensive unit tests in transcription/alignment/tests/test_arpabet_dataset.py testing G2P, aligner, balancer, and manifest persistence.
5. Execute extraction on LibriSpeech dev-clean to generate data/arpabet_alignment/words/ and data/arpabet_alignment/words_manifest.json.
6. Verify tests with pytest and type check with pyright.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented G2pExtractor (g2p_en with stress stripping and punctuation cleaning), MMSForcedAligner (torchaudio MMS_FA pipeline with CPU/MPS execution), and PhoneticWordBalancer (max 2 occurrences per lemma, greedy max-min phoneme fairness). Processed all 2,703 LibriSpeech dev-clean utterances (52,093 word candidates), successfully extracted 5,000 balanced single-word clips (3,592 unique words) with +/-25ms boundary padding saved as 16kHz mono PCM_16 WAV under data/arpabet_alignment/words/, and exported data/arpabet_alignment/words_manifest.json with 100% coverage (39/39 ARPAbet phonemes, minimum 58 occurrences for rarest phoneme ZH). Automated test suite in test_arpabet_dataset.py passed (15/15 tests) and pyright reported 0 errors.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Built G2P extraction, torchaudio MMS_FA forced alignment, and phonetic dataset balancing modules for LibriSpeech. Extracted 5,000 single-word 16kHz mono PCM .wav clips with +/-25ms boundary padding under data/arpabet_alignment/words/ across 3,592 unique words (capped at <=2 occurrences per word) and wrote durable data/arpabet_alignment/words_manifest.json with 100% coverage across all 39 standard ARPAbet phonemes (min 58 occurrences for ZH). Verified audio format with soundfile, schema with WordManifestEntry, and validated with 15 passing unit tests and 0 pyright errors.
<!-- SECTION:FINAL_SUMMARY:END -->
