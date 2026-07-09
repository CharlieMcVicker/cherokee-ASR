---
id: TASK-110
title: Integrate VAD segmentation into process_interviews.py
status: Done
assignee:
  - '@antigravity'
created_date: '2026-07-09 19:27'
updated_date: '2026-07-09 19:28'
labels: []
dependencies: []
priority: high
ordinal: 106000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Improve audio segmentation by replacing/supplementing energy-based profiling with a SpeechBrain-based VAD model to avoid clipping in the middle of words and phrases.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Use SpeechBrain VAD model to get speech segment boundaries
- [x] #2 Update process_interviews.py to segment using VAD boundaries instead of or combined with energy-based splits
- [x] #3 Add keep_silence padding or overlap around VAD-based segments to avoid clipping
- [x] #4 Ensure fallback split logic in split_long_segments_smart utilizes VAD or has safety margins
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Initialize VAD: Load the SpeechBrain VAD model (forcing CPU mode for stability).\n2. Segment using VAD: Save the denoised audio to a temporary WAV file, run VAD, and get the speech segment boundaries.\n3. Add Padding: Pad each segment (e.g., with 200ms keep-silence) to avoid clipping at the start or end of words, without causing adjacent segments to overlap.\n4. Smarter Splitting for Long Segments: When VAD produces segments >10s, split them with 200-500ms overlap padding at the split boundary so the ASR model has context.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Integrated SpeechBrain VAD model into process_interviews.py for robust speech boundary detection. Configured VAD segmentation as the default method while keeping energy-profile based segmentation as a fallback (--use-energy). Updated split_long_segments_smart to include overlap padding (--overlap, default 250ms) during fallback splits to prevent clipping in the middle of words and phrases.
<!-- SECTION:FINAL_SUMMARY:END -->
