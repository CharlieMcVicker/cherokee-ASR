---
id: TASK-350.7
title: Realign saving-the-voices/gs_mm Interview with Code-Switched Greedy Aligner
status: Done
assignee:
  - '@supervisor'
created_date: '2026-09-18 14:49'
updated_date: '2026-09-18 16:17'
labels:
  - alignment
  - interview
  - benchmark
  - evaluation
dependencies:
  - TASK-350.6
parent_task_id: TASK-350
priority: high
type: feature
ordinal: 376000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Realign the sample interview saving-the-voices/gs_mm.wav and saving-the-voices/gs_mm.txt using the code-switched alignment pipeline. Verify that English words, names (Guy Soldier, Jay, Charley McCoy, Dry Creek), and conversational markers (yeah, ok) receive accurate word-level alignment timestamps without collapsing or being dropped. Export Praat TextGrid and alignment JSON manifest, and report alignment quality metrics and improvements over baseline greedy alignment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Execute code-switched alignment on saving-the-voices/gs_mm.wav with ground truth from saving-the-voices/gs_mm.txt
- [x] #2 Verify English names and words (Guy Soldier, Jay, Charley McCoy, Dry Creek, yeah, ok) receive valid aligned boundaries
- [x] #3 Export multi-tier Praat TextGrid and alignment manifest to saving-the-voices/output_codeswitched/
- [x] #4 Evaluate and document alignment scores, matched word ratio, and CER comparison against baseline output_greedy
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect baseline alignment in saving-the-voices/output_greedy/ to establish baseline word match counts and English word alignment status.
2. Run align_syllabary_greedy on saving-the-voices/gs_mm.wav with saving-the-voices/gs_mm.txt using toneless pre-Bible model and code_switched=True with our calibrated confusion matrix projector.
3. Export multi-tier Praat TextGrid and alignment JSON manifest to saving-the-voices/output_codeswitched/.
4. Audit aligned word intervals for English names and code-switched phrases (Guy Soldier, Jay, Charley McCoy, Dry Creek, yeah, ok).
5. Compare alignment quality, matched word ratio, and CER metrics against baseline output_greedy.
6. Add integration test in transcription/alignment/tests/test_interview_realignment.py verifying alignment artifacts and validity.
7. Update Backlog task TASK-350.7 and parent epic TASK-350.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Executed code-switched alignment on saving-the-voices/gs_mm.wav and gs_mm.txt using toneless pre-Bible model (charliemcvicker/length-only-20260704-155307-asr-cherokee-colon @ 76e62140955f4738abdab345ea34068b02d8d2a2) on mps. Added _build_english_word_tier and updated _build_syllabary_word_tier in pipeline.py. Output generated at saving-the-voices/output_codeswitched/gs_mm_codeswitched.TextGrid (7 tiers) and gs_mm_codeswitched_manifest.json (290 lines). Realigned interview achieved 100% matched chunk ratio (290/290) vs 87.93% (255/290) baseline, aligning +2,891 additional ground-truth characters. Audited all key code-switched words ('Guy', 'Soldier', 'Jay', 'Charley', 'McCoy', 'Dry', 'Creek', 'yeah', 'ok') verifying non-zero durations and valid boundary alignments. Added unit and integration tests in test_interview_realignment.py; full 240-test suite passing with 0 pyright errors.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Realigned saving-the-voices/gs_mm interview audio and syllabary transcript using the toneless pre-Bible ASR model (charliemcvicker/length-only-20260704-155307-asr-cherokee-colon @ 76e62140955f4738abdab345ea34068b02d8d2a2) and calibrated confusion matrix projector. Extended pipeline.py with _build_english_word_tier and fused-word-aware _build_syllabary_word_tier, exporting 7-tier Praat TextGrid and alignment JSON manifest to saving-the-voices/output_codeswitched/. Increased matched chunk ratio from 87.93% (255/290) to 100.0% (290/290) while expanding total aligned ground-truth characters from 24,659 to 27,550 (+2,891 chars). Verified all key code-switched names and loanwords (Guy Soldier, Jay, Charley McCoy, Dry Creek, yeah, ok) have valid non-zero durations and accurate aligned boundaries. Added regression and artifact validation tests in test_interview_realignment.py; verified with 240 passing pytest tests and 0 pyright errors.
<!-- SECTION:FINAL_SUMMARY:END -->
