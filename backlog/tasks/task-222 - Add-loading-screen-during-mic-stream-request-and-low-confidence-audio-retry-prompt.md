---
id: TASK-222
title: >-
  Add loading screen during mic stream request and low-confidence audio retry
  prompt
status: Done
assignee:
  - '@agent-k'
created_date: '2026-08-12 22:27'
updated_date: '2026-08-12 22:28'
labels: []
dependencies: []
ordinal: 213000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
1. Show loading screen ('Connecting to your microphone...') while requesting mic stream/media stream, only transitioning to mic selection screen after media stream is acquired. 2. Filter/check ASR transcription confidence score; if confidence is under 90, display/prompt 'Sorry, please say that again'.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Display loading state ('Connecting to your microphone...') while obtaining media stream before showing mic options
- [x] #2 Acquire media stream before rendering microphone selection options
- [x] #3 Check transcription confidence values and trigger 'sorry please say that again' if under 90
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Locate mic permission / stream acquisition & device selector logic and transcription response / confidence handling in syllabary_transcriber frontend.
2. Implement 'Connecting to your microphone...' loading state while getUserMedia is resolving media stream before allowing mic selection page display.
3. Update transcription result handler to evaluate confidence score; if under 90, display/feedback 'Sorry please say that again'.
4. Test and verify changes.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added microphone stream loading state screen ('Connecting to your microphone...') to present while getUserMedia acquires audio stream prior to showing microphone options. Integrated confidence score evaluation threshold (<90 threshold triggers 'Sorry, please say that again' feedback prompt).
<!-- SECTION:FINAL_SUMMARY:END -->
