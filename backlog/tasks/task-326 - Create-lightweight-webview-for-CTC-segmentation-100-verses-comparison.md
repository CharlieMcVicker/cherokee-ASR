---
id: TASK-326
title: Create lightweight webview for CTC segmentation 100 verses comparison
status: Done
assignee:
  - '@antigravity'
created_date: '2026-09-14 14:45'
updated_date: '2026-09-14 14:48'
labels: []
dependencies: []
ordinal: 342000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build a lightweight webview interface to inspect CTC segmentation benchmark comparison data (runs/evaluation/ctc_segmentation_100_verses_comparison.json), highlighting anomalous verses and flagged words with embedded audio playback.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Webview displays overall summary stats and lists verses from comparison JSON
- [x] #2 Provides filter/toggle to isolate anomalous verses and view flagged words
- [x] #3 Highlights flagged vs aligned words with confidence scores, timestamps, and emitted spellings
- [x] #4 Includes audio player to listen to full verse audio and click-to-play individual word segments
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Design a clean, modern, lightweight standalone webview (HTML/CSS/JS) or Python server/CLI tool to view the CTC segmentation comparison.
2. Provide interactive filtering (All vs Anomalies only, search by verse ID or text).
3. Display summary metrics (total verses, anomalies count, flagged words count).
4. For each verse, render reference syllabary, phonetic citation, ASR hypotheses, word-by-word alignment chips with confidence scores, timestamps, and flagged indicators.
5. Provide native audio playback for each verse and word-level snippet playback on click.
6. Provide a standalone Python webview server runner (e.g.  or standalone HTML file generator) so it's trivial to launch.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented scripts/view_ctc_comparison.py with RangeRequestHandler for audio streaming and rich responsive UI. Verified with static HTML export, curl HTTP range headers, pyright typecheck, and full pytest suite (263 passed).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Built a lightweight, responsive webview application in scripts/view_ctc_comparison.py for inspecting runs/evaluation/ctc_segmentation_100_verses_comparison.json. Features interactive filtering for anomalous verses, real-time search, sorting, Cherokee Syllabary / phonetics display, flagged word callouts with confidence scores and timestamps, full verse audio player, and click-to-play word segment playback.
<!-- SECTION:FINAL_SUMMARY:END -->
