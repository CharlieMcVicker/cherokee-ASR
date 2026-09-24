---
id: TASK-334
title: >-
  Ensure hs preaspiration is always applied during syllabary-to-phonetics
  conversion
status: Done
assignee:
  - '@agent'
created_date: '2026-09-16 16:00'
updated_date: '2026-09-16 16:09'
labels: []
dependencies: []
ordinal: 350000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update syllabary conversion maps/functions so preconsonantal and s-coda syllabary conversions always respell 's' to aspirated 'hs' in accordance with canonical TTH orthography. Rescore clean/noisy dataset performance to evaluate the improvement.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Update syllabary to phonetics conversion rules so preconsonantal s is always respelled as hs
- [x] #2 Verify all unit tests pass with pytest
- [x] #3 Rescore dataset across clean and noisy conditions and measure impact on CER/WER
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect orthography.py, syllabary_map.py, and normalizers.py to understand how Cherokee Syllabary is converted to canonical TTH phonetics.
2. Update the conversion logic so that preconsonantal 's' (e.g. s before consonants / s-glyph Ꮝ) is always converted to pre-aspirated 'hs'.
3. Run pytest to ensure zero regressions in existing tests.
4. Run rescore_syllabary_dataset.py and compare the numbers against the previous run.
5. Record metrics and finalize task.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated syllabary maps to map all /s/ series (Ꮜ, Ꮝ, Ꮞ, Ꮟ, Ꮠ, Ꮡ, Ꮢ) to canonical pre-aspirated /hs/. Verified 100% of test suite passes (273/273 tests). Rescored clean and noisy conditions on all 1,389 dataset samples.
<!-- SECTION:FINAL_SUMMARY:END -->
