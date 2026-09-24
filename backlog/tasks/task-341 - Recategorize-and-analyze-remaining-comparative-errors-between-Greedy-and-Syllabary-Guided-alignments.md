---
id: TASK-341
title: >-
  Recategorize and analyze remaining comparative errors between Greedy and
  Syllabary-Guided alignments
status: Done
assignee:
  - '@myself'
created_date: '2026-09-16 17:10'
updated_date: '2026-09-16 17:15'
labels: []
dependencies: []
documentation:
  - >-
    backlog/docs/evaluation/doc-8 -
    Syllabary-Guided-vs-Greedy-CTC-Segmentation-Progress-Tracker.md
priority: high
type: spike
ordinal: 357000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Perform a comprehensive 3-way comparative error analysis across the 1,389 Cherokee Syllabary dataset samples (Ground Truth vs. Greedy ASR vs. Syllabary-Guided CTC Segmentation) following the resolution of lateral deaffrication and laryngeal masking. Classify all remaining degradation events into structured buckets (e.g. voiceless sonorant aspiration, spelling inconsistencies in ground truth, hiatus glottal stops, minor medial syncope) to guide the next phase of aligner enhancements.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Generate an exhaustive 3-way comparative error breakdown script and report across clean and noisy evaluation conditions
- [x] #2 Classify remaining mismatch events into discrete phonological and transcript categories with exact counts and examples
- [x] #3 Update doc-8 with the revised error taxonomy and next targeted interventions
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Build comprehensive 3-way comparative error analysis script (scripts/analyze_comparative_errors.py) comparing Ground Truth, Greedy ASR, and Syllabary-Guided CTC Segmentation across clean and noisy evaluation conditions.
2. Run script across all 1,389 samples (clean & noisy) to categorize error patterns into granular phonological and transcript taxonomies (sonorant pre-aspiration, glottal/hiatus stop variation, vowel syncope, ground truth typos/inconsistencies, lateral remainder, etc.) with exact counts, percentages, and sample IDs.
3. Generate detailed comparative breakdown report artifact and output JSON.
4. Update doc-8 (Syllabary-Guided vs Greedy Progress Tracker) with updated benchmark taxonomy and targeted future interventions.
5. Verify acceptance criteria and finalize task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Generated 3-way comparative error analysis script (scripts/analyze_comparative_errors.py) and classified all residual mismatches across clean & noisy conditions into structured categories. Verified with pytest (275 tests passed) and pyright (0 errors). Updated doc-8 with revised taxonomy and 4 priority interventions.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented scripts/analyze_comparative_errors.py to perform exhaustive 3-way comparative error analysis between Ground Truth, Greedy ASR, and Syllabary-Guided CTC Segmentation across 1,389 Cherokee Syllabary samples. Categorized error events into discrete phonological/transcript buckets (laryngeal stride intrusions, medial syncope, initial sibilant normalizer, sonorant pre-aspiration), generated comprehensive JSON and Markdown reports, and updated doc-8 progress tracker with targeted next interventions. Verified with pytest and pyright.
<!-- SECTION:FINAL_SUMMARY:END -->
