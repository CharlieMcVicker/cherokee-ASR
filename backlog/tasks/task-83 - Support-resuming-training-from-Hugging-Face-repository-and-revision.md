---
id: TASK-83
title: Support resuming training from Hugging Face repository and revision
status: Done
assignee:
  - '@antigravity'
created_date: '2026-07-02 21:00'
updated_date: '2026-07-02 21:01'
labels: []
dependencies: []
ordinal: 79000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add support in transcription/training/train.py to resume training from a Hugging Face Hub repository and revision using snapshot_download.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add --resume-from-repo and --resume-from-revision CLI arguments to train.py
- [x] #2 Use snapshot_download to fetch the checkpoint from the Hub if --resume-from-repo is set
- [x] #3 Support standard --resume-from-checkpoint argument for local resuming
- [x] #4 Pass downloaded snapshot path to trainer.train()
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add --resume-from-repo, --resume-from-revision, and --resume-from-checkpoint CLI arguments in main() of train.py.
2. In main() of train.py, check if --resume-from-repo is set. If so, call snapshot_download using the provided repo ID, revision, and HF token to download the checkpoint.
3. If --resume-from-checkpoint is set, resolve its path (e.g. if 'latest', find latest local checkpoint folder, or use the path directly).
4. Pass the downloaded checkpoint folder or the local resume checkpoint folder to trainer.train(resume_from_checkpoint=...).
5. Test training command/script execution to make sure code compiles and handles arguments correctly.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added support for resuming training from a Hugging Face Hub repository and revision, or a local checkpoint.
- Introduced CLI parameters: --resume-from-repo, --resume-from-revision, and --resume-from-checkpoint.
- Implemented download functionality with snapshot_download from the huggingface_hub package when --resume-from-repo is specified.
- Enabled automatic local checkpoint resumption via latest or by specifying a direct folder path.
- Passed the resolved checkpoint path directory to trainer.train(resume_from_checkpoint=...).
<!-- SECTION:FINAL_SUMMARY:END -->
