---
id: TASK-360
title: Deconstruct and modularize repository into 4-tier architecture
status: Done
assignee:
  - '@supervisor'
created_date: '2026-09-21 20:24'
updated_date: '2026-09-23 15:33'
labels: []
dependencies: []
ordinal: 386000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deconstruct and refactor transcription into a clean 4-tier modular architecture separating language-agnostic core algorithms (models, audio, generic alignment, exporters), Cherokee phonetics (orthography, phonotactics, codeswitching, enrichment), domain use-case pipelines (Scripture, Dialogue, Enrichment), and downstream applications/scripts, adhering to the Types & Maps principle and Clean Break Protocol.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 transcription.core contains purely language-agnostic ASR, audio, CTC/DP alignment, and TextGrid/manifest exporters
- [x] #2 transcription.cherokee encapsulates all orthography, phonotactics, Arpabet projection, and syllabary enrichment logic
- [x] #3 transcription.pipelines encapsulates Scripture, Dialogue, and Enrichment domain workflows
- [x] #4 scripts/realign_gs_mm_ctc.py and scripts/realign_bible.py are thin declarative pipeline drivers (<30 lines)
- [x] #5 pyproject.toml entrypoints, AGENTS.md, and docs/ are fully synchronized
- [x] #6 All 390+ unit and integration tests pass and pyright transcription reports 0 errors
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Successfully deconstructed and modularized repository into a clean 4-tier architecture (Tier 1: transcription.core, Tier 2: transcription.cherokee, Tier 3: transcription.pipelines, Tier 4: transcription.apps). Streamlined driver scripts to thin declarative wrappers, pruned obsolete legacy files per Clean Break Protocol, synchronized AGENTS.md and technical documentation in docs/, and verified full test suite (460 passed, 0 failures) and static typing (0 pyright errors).
<!-- SECTION:FINAL_SUMMARY:END -->
