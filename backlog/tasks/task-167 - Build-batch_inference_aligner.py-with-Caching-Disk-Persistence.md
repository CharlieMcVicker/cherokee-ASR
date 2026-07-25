---
id: TASK-167
title: Build batch_inference_aligner.py with Caching & Disk Persistence
status: To Do
assignee: []
created_date: '2026-07-25 19:55'
updated_date: '2026-07-25 19:57'
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
- [ ] #1 Build batch_inference_aligner.py ingesting data manifest JSON/JSONL
- [ ] #2 Use batch GPU inference from transcription.inference.batch for emitted_text
- [ ] #3 Call character alignment engine to attach aligned_pairs tuples [(syllabary_char, asr_text)]
- [ ] #4 Persist emitted_text and aligned_pairs to disk cache manifest
- [ ] #5 Add --force-recompute flag (default False) to reload cached alignment manifest when present
- [ ] #6 Use batched GPU inference from transcription.inference.batch to compute ASR logits/emissions
- [ ] #7 Pass batched emissions directly into decoupled CPU alignment functions from transcription.timestamping
- [ ] #8 Call character alignment engine to attach aligned_pairs tuples [(syllabary_char, asr_text)]
- [ ] #9 Persist emitted_text and aligned_pairs to disk cache manifest
- [ ] #10 Add --force-recompute flag (default False) to reload cached alignment manifest when present
<!-- AC:END -->
