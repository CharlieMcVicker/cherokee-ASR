---
id: TASK-175
title: Generalize aspiration transfer from ASR across all consonant series
status: In Progress
assignee:
  - '@agent'
created_date: '2026-07-25 20:47'
updated_date: '2026-07-25 20:47'
labels: []
dependencies: []
ordinal: 171000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Refactor enrich_syllabary.py so that aspiration from ASR is generally preserved across all consonant series and clusters according to the core principles: 1) Vowel deletion/syncopation follows ASR, and 2) Aspiration (pre-aspiration h, post-aspiration/laryngeal toggles th/kh/lh/nh/wh/yh/rh/sh/ch, and trailing h) follows ASR across all syllabary units uniformly, without ad-hoc per-consonant rules.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reorganize _enrich_single_syllable around general Rule 1 (ASR Vowel Deletion) and Rule 2 (ASR Aspiration Transfer)
- [ ] #2 Ensure pre-consonantal, post-consonantal, and post-vocalic aspiration in emitted ASR are systematically transferred onto base syllabary
- [ ] #3 Ensure syncopation drops final vowel without stripping emitted aspiration (e.g. li + ASR 'lh' -> 'lh', ni + ASR 'nh' -> 'nh', si + ASR 'hs' -> 'hs')
- [ ] #4 All unit tests pass cleanly
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Refactor _enrich_single_syllable in enrich_syllabary.py to decouple aspiration transfer into generic pre/post/laryngeal aspiration rules for all consonants (l, n, w, y, r, t, k, s, m, etc.).\n2. First apply ASR laryngeal/aspiration modifications (pre-h, internal 'h' in digraphs like th/kh/lh/nh/wh/yh/tsh/sh/ch, and trailing post-vocalic h) onto the base consonant/syllable.\n3. Then evaluate Rule 1 (vowel deletion/syncopation): if ASR emitted contains no vowel and base ends in a vowel, drop only the final vowel while preserving all pre- and post-consonantal aspiration ('h').\n4. Update unit tests in test_enrich_syllabary.py to cover general aspiration transfer across multiple consonant series.\n5. Execute unittest discover and verify.
<!-- SECTION:PLAN:END -->
