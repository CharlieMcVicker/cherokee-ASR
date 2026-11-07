---
id: TASK-398
title: Update Vast.ai guide and dual-head plan with custom Docker image workflow
status: Done
assignee:
  - '@self'
created_date: '2026-11-07 21:06'
updated_date: '2026-11-07 21:06'
labels: []
dependencies: []
modified_files:
  - backlog/docs/guides/vast_ai_training/doc-2 - Vast.ai-Training-Guide.md
  - dual_head_tone_asr_plan.md
ordinal: 431300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Audit doc-2 (Vast.ai Training Guide) and dual_head_tone_asr_plan.md to replace obsolete third-party image references (rolandocoto) and placeholder strings with our dedicated Docker image workflow (charliemcvicker/asr:latest built from project Dockerfile). Document image build, push, and remote run steps.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Update doc-2 to specify our project Dockerfile, charliemcvicker/asr:latest image, and digohwelisgi module paths
- [x] #2 Update dual_head_tone_asr_plan.md Section 6 with our Docker image build/push step and Vast launch template
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Audited and updated doc-2 and dual_head_tone_asr_plan.md to replace obsolete third-party image references (rolandocoto) and placeholder strings with our dedicated Docker image workflow: charliemcvicker/asr:latest built directly from the project root Dockerfile. Documented the prebaked foundation model cache, local image build/push commands, and updated module entrypoint digohwelisgi.cherokee.training.train.
<!-- SECTION:FINAL_SUMMARY:END -->
