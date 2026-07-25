---
id: TASK-167
title: Build batch_inference_aligner.py with Caching & Disk Persistence
status: Done
assignee:
  - '@agent'
created_date: '2026-07-25 19:55'
updated_date: '2026-07-25 20:02'
labels: []
dependencies: []
ordinal: 163000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build batch inference and alignment pipeline by combining batched GPU inference (from transcription/inference/batch.py) with the decoupled CPU alignment functions from transcription.timestamping. Save all transcription and alignment manifest data to disk by default, with a --force-recompute flag (off by default) to allow fast iteration over downstream reconciliation logic without re-running ASR inference.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Build batch_inference_aligner.py ingesting data manifest JSON/JSONL
- [x] #2 Use batch GPU inference from transcription.inference.batch for emitted_text
- [x] #3 Call character alignment engine to attach aligned_pairs tuples [(syllabary_char, asr_text)]
- [x] #4 Persist emitted_text and aligned_pairs to disk cache manifest
- [x] #5 Add --force-recompute flag (default False) to reload cached alignment manifest when present
- [x] #6 Use batched GPU inference from transcription.inference.batch to compute ASR logits/emissions
- [x] #7 Pass batched emissions directly into decoupled CPU alignment functions from transcription.timestamping
- [x] #8 Call character alignment engine to attach aligned_pairs tuples [(syllabary_char, asr_text)]
- [x] #9 Persist emitted_text and aligned_pairs to disk cache manifest
- [x] #10 Add --force-recompute flag (default False) to reload cached alignment manifest when present
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect transcription/inference/batch.py and transcription/syllabary_enrichment/alignment_engine.py.\n2. Create batch_inference_aligner.py in transcription/syllabary_enrichment.\n3. Implement batch GPU inference for emitted_text using batch.py helper functions.\n4. Call align_character_syllable() to produce aligned_pairs tuples [(syllabary_char, asr_text)].\n5. Implement disk manifest caching with --force-recompute flag (default False).\n6. Add unit/integration tests for batch_inference_aligner.py.\n7. Check ACs and set TASK-167 to Done.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented batch_inference_aligner.py module in transcription/syllabary_enrichment with support for data manifest (JSON/JSONL) ingestion, batched GPU ASR inference (via transcription.inference.batch), fine-grained character alignment (align_character_syllable), disk manifest caching, and --force-recompute flag. Added unit and integration tests in test_batch_inference_aligner.py which pass cleanly.
<!-- SECTION:FINAL_SUMMARY:END -->
