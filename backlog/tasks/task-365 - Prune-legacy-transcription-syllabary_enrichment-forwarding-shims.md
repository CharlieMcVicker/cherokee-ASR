---
id: TASK-365
title: Prune legacy transcription/syllabary_enrichment forwarding shims
status: To Do
assignee: []
created_date: '2026-09-23 15:55'
labels: []
dependencies: []
ordinal: 395300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
transcription/syllabary_enrichment/enrich_syllabary.py and alignment_engine.py forward to transcription.cherokee.enrichment.syllable_alignment. Update all callers in evaluation and tools to import directly from Tier 2 transcription.cherokee.enrichment or Tier 3 transcription.pipelines.enrichment, and prune the deprecated shim files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 enrich_syllabary.py and legacy shims in transcription/syllabary_enrichment/ pruned
- [ ] #2 Call sites updated to transcription.cherokee.enrichment
- [ ] #3 All tests pass and pyright reports 0 errors
<!-- AC:END -->
