---
id: TASK-269.3
title: Implement NoisyEvaluator with CTC peak top-K extraction in evaluator.py
status: Done
assignee:
  - '@pipeline-engineer'
created_date: '2026-09-02 16:47'
updated_date: '2026-09-02 16:55'
labels:
  - evaluation
  - evaluator
  - asr
dependencies: []
parent_task_id: TASK-269
priority: high
type: feature
ordinal: 274000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement EvaluationRecord Pydantic model and NoisyEvaluator multi-SNR batch runner with non-blank emission peak tracking, top-K posterior collection, streaming JSONL sink, and checkpoint resume support.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Define EvaluationRecord schema using Pydantic
- [x] #2 Implement NoisyEvaluator interfacing with CherokeeASRModel via public API
- [x] #3 Extract non-blank CTC emission peaks and top-K posterior probabilities per position
- [x] #4 Stream evaluation records directly to JSONL with checkpoint resume by audio_id
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented EvaluationRecord schema and NoisyEvaluator with non-blank CTC peak extraction and streaming JSONL checkpoint resume.
<!-- SECTION:FINAL_SUMMARY:END -->
