---
id: TASK-342
title: >-
  Add 3-way comparative view (Syllabary vs Greedy vs Guided) to Bible
  realignment and inspector UX
status: Done
assignee:
  - '@myself'
created_date: '2026-09-16 17:36'
updated_date: '2026-09-16 17:39'
labels: []
dependencies: []
ordinal: 358000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update Bible realignment pipeline to extract and persist Greedy ASR hypotheses alongside Syllabary-Guided CTC trellis emissions for every verse. Update the interactive alignment viewer (view_ctc_comparison.py) to display Syllabary, Greedy inference, and Guided inference side-by-side with visual diffing to rapidly spot syncope and intrusion regressions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Update realign_bible.py and pipeline to record both Greedy ASR hypothesis and Syllabary-Guided trellis emission in alignment records
- [x] #2 Update view_ctc_comparison.py to support Bible alignment records (mark_alignment_records.json, bible_alignment_records.json)
- [x] #3 Render 3-way view (1. Native Syllabary, 2. Greedy ASR Inference, 3. Syllabary-Guided Trellis Emission) with character diff highlighting in the UI
- [x] #4 Verify Mark Chapter 1 records contain all 3 representations and display correctly in the browser viewer
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update realign_bible.py to extract greedy ASR inference for each verse slice and store both greedy_hypothesis and guided_hypothesis in alignment JSON records.
2. Update view_ctc_comparison.py to support loading Bible alignment records (mark_alignment_records.json, bible_alignment_records.json) as well as comparison JSONs.
3. Enhance UI in view_ctc_comparison.py to display 3 distinct comparison tiers for every verse: (1) Native Cherokee Syllabary, (2) Greedy ASR Inference, (3) Syllabary-Guided Trellis Emission, along with inline visual diff highlights and audio playback.
4. Realign Mark Chapter 1 and test viewer.
5. Verify pytest and pyright, then finalize task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated realign_bible.py to extract and store greedy_hypothesis alongside guided_hypothesis. Enhanced scripts/view_ctc_comparison.py with a 3-tier comparative layout (Native Syllabary, Greedy ASR Inference, Syllabary-Guided CTC Segmentation), word-level diff highlight, and audio playback. Verified with pytest and pyright.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented 3-way comparative alignment inspection in Bible realignment and webview. Updated scripts/realign_bible.py to decode verse audio slices with greedy ASR and record both greedy_hypothesis and guided_hypothesis in alignment JSONs. Updated scripts/view_ctc_comparison.py to support loading any alignment dataset, displaying 3 distinct tiers (Native Syllabary, Greedy Inference, Guided Inference), inline diff comparison, and audio snippet playback. Verified across Mark Chapter 1 records with 275 passing tests and 0 pyright errors.
<!-- SECTION:FINAL_SUMMARY:END -->
