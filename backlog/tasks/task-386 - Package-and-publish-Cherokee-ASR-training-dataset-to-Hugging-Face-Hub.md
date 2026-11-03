---
id: TASK-386
title: Package and publish Cherokee ASR training dataset to Hugging Face Hub
status: To Do
assignee: []
created_date: '2026-11-03 19:45'
labels: []
dependencies:
  - TASK-385
ordinal: 419300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the Cherokee-specific dataset packaging driver under Tier 2 that loads the 6 active Cherokee splits (chr and bible), packages the 2,442 referenced audio files (462MB), and publishes the dataset to charliemcvicker/cherokee-asr-data on Hugging Face Hub.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Define Cherokee dataset packaging specification targeting the 6 active CSV splits
- [ ] #2 Resolve and verify the 2,442 referenced audio files across sentence_audio and split_audio
- [ ] #3 Publish the dataset to charliemcvicker/cherokee-asr-data with documentation and manifest on Hugging Face Hub
- [ ] #4 Verify that all 6 splits can be successfully parsed against staged audio
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Dataset is staged, validated, and accessible on Hugging Face
<!-- DOD:END -->
