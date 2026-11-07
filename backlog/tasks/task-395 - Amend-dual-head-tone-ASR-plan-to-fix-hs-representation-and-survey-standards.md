---
id: TASK-395
title: Amend dual head tone ASR plan to fix hs representation and survey standards
status: Done
assignee:
  - '@self'
created_date: '2026-11-07 20:54'
updated_date: '2026-11-07 20:55'
labels: []
dependencies: []
modified_files:
  - dual_head_tone_asr_plan.md
ordinal: 428300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Review dual_head_tone_asr_plan.md, investigate standards for /hs/ and /s/ in Cherokee phonology and existing orthographies, and add an amendment section specifying the standardized /hs/ representation, dataset regeneration, and orthography engine fixes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Research and document existing /hs/ standards across Cherokee linguistics, dictionaries, and acoustic ASR
- [x] #2 Review dual_head_tone_asr_plan.md and formulate the /hs/ remediation strategy
- [x] #3 Amend dual_head_tone_asr_plan.md with the /hs/ representation fix and amendment phase
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reviewed dual_head_tone_asr_plan.md and surveyed Cherokee /hs/ standards across Feeling (1975) DG, modern phonology (Montgomery-Anderson, Uchihara), contextual TTH acoustic, unconditional syllabary mapping, and plain sibilant conventions. Appended Section 5 to dual_head_tone_asr_plan.md detailing the root-cause regex fix in tones.py, the adopted Standard 2 (Contextual Phonotactic TTH), its impact on Dual-Head CTC alignment, and actionable Phase 0 prerequisite tasks.
<!-- SECTION:FINAL_SUMMARY:END -->
