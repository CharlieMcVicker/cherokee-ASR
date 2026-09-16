---
id: TASK-340
title: Optimize intrusion masking and max stride for Cherokee acoustic pre-aspiration
status: Done
assignee:
  - '@agent'
created_date: '2026-09-16 16:49'
updated_date: '2026-09-16 16:52'
labels: []
dependencies: []
ordinal: 356000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update CTCAlignerConfig intrusive_max_stride to 4 and refine phonotactic intrusion site masking in phonotactics.py to comprehensively cover pre-consonantal stops, post-syncope sonorants, and coda laryngeals. Verify with unit tests and rescore dataset.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Update CTCAlignerConfig intrusive_max_stride to 4
- [x] #2 Refine phonotactics intrusion site masking to license all attested Cherokee laryngeal environments
- [x] #3 Verify all unit tests pass with pytest
- [x] #4 Rescore dataset on clean and noisy conditions and measure reduction in missing h errors
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update CTCAlignerConfig.intrusive_max_stride default from 1 to 4 in transcription/alignment/models.py.\n2. Review and refine get_intrusion_mask in transcription/alignment/phonotactics.py.\n3. Run pytest to verify all unit tests pass.\n4. Run scripts/rescore_syllabary_dataset.py on clean and noisy conditions.\n5. Analyze reduction in missing_h errors and overall CER/WER.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated CTCAlignerConfig.intrusive_max_stride default to 4 and refined get_intrusion_site_mask in phonotactics.py to license pre-consonantal intrusions before aspirated stops and voiceless sonorants as well as word-initial vowel onsets while blocking word-initial consonant pre-aspiration. Verified with 274 passing pytest tests and rescored the full dataset, reducing unassociated missing H events from 112 to 53, achieving 2.91% Clean Valid CER and beating Greedy by -2.20% CER / -9.56% WER on Noisy Valid and -0.83% CER / -3.36% WER on Noisy Test.
<!-- SECTION:FINAL_SUMMARY:END -->
