---
id: TASK-360
title: Deconstruct and modularize repository into 4-tier architecture
status: In Progress
assignee:
  - '@supervisor'
created_date: '2026-09-21 20:24'
updated_date: '2026-09-22 14:48'
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
- [ ] #1 transcription.core contains purely language-agnostic ASR, audio, CTC/DP alignment, and TextGrid/manifest exporters
- [ ] #2 transcription.cherokee encapsulates all orthography, phonotactics, Arpabet projection, and syllabary enrichment logic
- [ ] #3 transcription.pipelines encapsulates Scripture, Dialogue, and Enrichment domain workflows
- [ ] #4 scripts/realign_gs_mm_ctc.py and scripts/realign_bible.py are thin declarative pipeline drivers (<30 lines)
- [ ] #5 pyproject.toml entrypoints, AGENTS.md, and docs/ are fully synchronized
- [ ] #6 All 390+ unit and integration tests pass and pyright transcription reports 0 errors
<!-- AC:END -->
