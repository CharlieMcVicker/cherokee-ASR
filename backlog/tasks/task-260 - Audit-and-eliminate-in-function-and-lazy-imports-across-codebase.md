---
id: TASK-260
title: Audit and eliminate in-function and lazy imports across codebase
status: To Do
assignee: []
created_date: '2026-09-02 15:01'
updated_date: '2026-09-02 15:05'
labels:
  - code-smell
  - refactor
  - code-quality
dependencies: []
priority: medium
ordinal: 262000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Code smell identified in PR #3 review: imports defined inside functions instead of at module top level.

Target files in `transcription/`:
1. `transcription/inference/batch.py` (init_worker, batch_transcribe_parallel)
2. `transcription/inference/run.py` (get_vad_model)
3. `transcription/alignment/cli.py` (_load_model_extractor)
4. `transcription/training/evaluate_checkpoint.py` (main)
5. `transcription/training/evaluate_local_checkpoints.py` (main)
6. `transcription/training/evaluate_revisions.py` (main)
7. `transcription/training/train.py` (parse_args, run_checkpoint_evaluations, etc.)
8. `transcription/alignment/tests/test_cli.py` (test methods)

Deliverables:
- Move all imports to top-level module scope across all `transcription/` Python files.
- Resolve any circular imports cleanly if encountered.
- Ensure all 100 unit tests pass via `pytest` and `pyright` reports 0 errors.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Identify all in-function / lazy imports in transcription and tests
- [ ] #2 Move imports to top-level module scope
- [ ] #3 Ensure no circular import regressions or startup delays
- [ ] #4 Verify all tests pass and pyright typechecking succeeds
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect all identified files for in-function imports.
2. Hoist standard library, third-party, and internal module imports to top of file.
3. Test for circular dependency or initialization side-effects.
4. Run `pytest` and `pyright transcription`.
5. Format with pre-commit / black and verify clean working tree.
<!-- SECTION:PLAN:END -->
