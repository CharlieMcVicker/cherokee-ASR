---
id: TASK-309
title: Fix intra-verse slice cumulative word emission bug in align_verse_slice
status: To Do
assignee: []
created_date: '2026-09-12 20:01'
labels: []
dependencies: []
ordinal: 325000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix character state window extraction in align_verse_slice to avoid accumulating previous word characters when tokens are unaligned, ensuring benchmark reports cleanly separated emitted words.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Constrain character state extraction in align_verse_slice strictly to the word interval
- [ ] #2 Prevent unaligned words from accumulating state characters from previous words
- [ ] #3 Verify benchmark comparison output shows proper word-level emitted tokens
<!-- AC:END -->
