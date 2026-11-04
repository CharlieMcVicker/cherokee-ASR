---
id: TASK-391
title: Add script to push arbitrary local checkpoint to Hugging Face Hub
status: Done
assignee:
  - '@self'
created_date: '2026-11-04 00:59'
updated_date: '2026-11-04 01:00'
labels: []
dependencies: []
modified_files:
  - scripts/push_checkpoint_to_hub.py
ordinal: 424300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add standalone utility script scripts/push_checkpoint_to_hub.py to bundle checkpoint weights, processor configs, and eval logs and upload to Hugging Face Hub.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Support pushing specified local checkpoint with processor and eval logs
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added scripts/push_checkpoint_to_hub.py to stage and push any local checkpoint, processor config, and eval logs to Hugging Face Hub.
<!-- SECTION:FINAL_SUMMARY:END -->
