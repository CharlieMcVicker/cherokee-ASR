---
id: TASK-263
title: 'Address outstanding direct review feedback on PR #3'
status: Done
assignee:
  - '@agent-pr3-fixes'
created_date: '2026-09-02 15:33'
updated_date: '2026-09-02 15:35'
labels:
  - code-review
  - refactor
dependencies: []
priority: high
ordinal: 265000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address direct code review comments on PR #3:
1. Remove exporter import aliases in transcription/alignment/cli.py (export_* as write_*).
2. Remove top-level magic os.environ flags in cli.py and batch.py.
3. Dynamically compute WER/CER metric matrix over (name, transform_fn) tuples in transcription/utils/evaluation.py.
4. Correct documentation terminology in docs/alignment.md regarding Cherokee ASR model vs language-agnostic output data structures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Remove exporter import aliases in transcription/alignment/cli.py
- [x] #2 Remove module-level os.environ mutations in cli.py and batch.py
- [x] #3 Dynamically generate evaluation metric dictionary in transcription/utils/evaluation.py over transform tuples
- [x] #4 Correct documentation in docs/alignment.md
- [x] #5 Verify all unit tests pass and pyright reports 0 errors
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed direct PR #3 review comments: removed exporter import aliases and top-level os.environ mutations in cli.py and batch.py, refactored evaluation.py to dynamically iterate over transform tuples for WER/CER calculations, and corrected language-agnostic extraction wording in docs/alignment.md.
<!-- SECTION:FINAL_SUMMARY:END -->
