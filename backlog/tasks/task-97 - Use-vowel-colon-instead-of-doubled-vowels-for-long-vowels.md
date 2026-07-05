---
id: TASK-97
title: Use vowel-colon instead of doubled vowels for long vowels
status: Done
assignee:
  - '@agent'
created_date: '2026-07-04 15:26'
updated_date: '2026-07-04 15:27'
labels: []
dependencies: []
ordinal: 93000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update CSV prep scripts to represent long vowels using V: instead of doubled vowels VV. Implement this by replacing VV with V: as a final step after colon stripping.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Modify prepare_csv.py to convert doubled vowels to vowel-colon format as a final step
- [x] #2 Modify prepare_conrad_csv.py to convert doubled vowels to vowel-colon format as a final step
- [x] #3 Re-run the preparation pipelines and verify that colons exist only after vowels
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Modified prepare_csv.py and prepare_conrad_csv.py to represent long vowels with vowel-colon format (V:) rather than doubled vowels (VV) as a final step. The scripts first strip non-vowel colons, then convert doubled vowels (aa, ee, ii, oo, uu, vv) back to single vowel + colon (a:, e:, i:, o:, u:, v:). Re-ran both prep pipelines and verified via Python that all 17,543 colons in the generated dataset splits are preceded by a vowel.
<!-- SECTION:FINAL_SUMMARY:END -->
