---
id: TASK-313
title: Rerun realignment pipeline and benchmark comparison on Bible verses
status: Done
assignee:
  - '@myself'
created_date: '2026-09-12 20:59'
updated_date: '2026-09-12 21:03'
labels: []
dependencies: []
ordinal: 329000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute realign_bible.py and benchmark_ctc_segmentation_100_verses.py to evaluate alignment records, training CSVs, and phonetic differences after switching to direct trellis emissions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Run realign_bible.py across target Bible chapter/book
- [x] #2 Run benchmark_ctc_segmentation_100_verses.py and inspect diffs
- [x] #3 Document and report output changes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Run realign_bible.py on Mark chapter 1 (or all chapters if needed).
2. Run benchmark_ctc_segmentation_100_verses.py to evaluate phonetic differences.
3. Compare output changes (training CSVs, alignment records, diff breakdown).
4. Summarize results and mark task as Done.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reran Mark Chapter 1 realignment and benchmark comparison with direct trellis emissions (no syllabary enrichment, no fallback to phonetic text). Verified that spurious aspirations were significantly reduced (aspiration_removed: 36/45 verses) and anomalous verses (Mark 1:1, 1:16, 1:24, 1:27) were correctly excluded from training CSV mark.csv.
<!-- SECTION:FINAL_SUMMARY:END -->
