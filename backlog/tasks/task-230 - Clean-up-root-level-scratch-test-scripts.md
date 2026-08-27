---
id: TASK-230
title: Clean up root-level scratch test scripts
status: Done
assignee:
  - '@antigravity'
created_date: '2026-08-27 14:18'
updated_date: '2026-08-27 14:19'
labels: []
dependencies: []
ordinal: 221000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove junk root-level test scripts (test_decode.py, test_vocab.py, and related scratch files) from the repository.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Delete root-level scratch scripts (test_decode.py, test_vocab.py, test_topk.py, test_vocab_out.json)
- [x] #2 Verify test suite still runs cleanly
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Remove root-level scratch files: test_decode.py, test_vocab.py, test_topk.py, test_vocab_out.json.\n2. Run pytest to verify all tests pass.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed obsolete root-level scratch scripts (test_decode.py, test_vocab.py, test_topk.py, test_vocab_out.json) and verified test suite passes cleanly.
<!-- SECTION:FINAL_SUMMARY:END -->
