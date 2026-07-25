---
id: TASK-171
title: >-
  Support model-driven phonological toggle for tla/tli -> lha/lhi in syllabary
  enrichment
status: Done
assignee:
  - '@agent'
created_date: '2026-07-25 20:28'
updated_date: '2026-07-25 20:29'
labels: []
dependencies: []
ordinal: 167000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allow the ASR model predictions to trigger phonological shift from tla/tle/tli/tlo/tlu/tlv to lha/lhe/lhi/lho/lhu/lhv when the aligned ASR acoustic window indicates lateral aspiration/fricative shift.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement tla -> lha phonological shift rule in enrich_syllabary.py based on aligned ASR prediction
- [x] #2 Add unit test cases verifying tla -> lha transformation
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect enrich_syllabary.py.\n2. Implement lateral phonological shift rule: when base syllable is in tla/tle/tli/tlo/tlu/tlv series and aligned ASR window indicates lateral aspiration (lh/lha/lhah/lhi), transform tla -> lha (or lhah if post-aspirated).\n3. Add unit test coverage in test_enrich_syllabary.py.\n4. Re-run evaluation and verify CER drop.\n5. Mark ACs checked and set TASK-171 to Done.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented model-driven lateral phonological shift (tl-series -> lh-series: tla/tle/tli/tlo/tlu/tlv -> lha/lhe/lhi/lho/lhu/lhv) in enrich_syllabary.py when ASR emitted slice contains lateral aspiration/fricatives. Added unit test cases in test_enrich_syllabary.py and validated evaluation metrics.
<!-- SECTION:FINAL_SUMMARY:END -->
