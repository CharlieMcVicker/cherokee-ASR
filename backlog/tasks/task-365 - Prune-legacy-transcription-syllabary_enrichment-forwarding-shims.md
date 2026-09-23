---
id: TASK-365
title: Prune legacy transcription/syllabary_enrichment forwarding shims
status: In Progress
assignee:
  - '@subagent'
created_date: '2026-09-23 15:55'
updated_date: '2026-09-23 16:30'
labels: []
dependencies: []
modified_files:
  - transcription/pipelines/scripture/pipeline.py
  - docs/alignment.md
  - docs/syllabary_enrichment.md
  - syllabary_enrichment_failure_analysis.md
  - transcription/syllabary_enrichment/__init__.py
  - transcription/syllabary_enrichment/alignment_engine.py
  - transcription/syllabary_enrichment/batch_inference_aligner.py
  - transcription/syllabary_enrichment/enrich_syllabary.py
  - transcription/syllabary_enrichment/evaluate_reconciliation.py
  - transcription/syllabary_enrichment/generate_delta_cer_analysis.py
  - transcription/syllabary_enrichment/inspect_pipeline.py
  - transcription/syllabary_enrichment/prepare_validation_sheet.py
  - transcription/syllabary_enrichment/test_alignment_engine.py
  - transcription/syllabary_enrichment/test_batch_inference_aligner.py
  - transcription/syllabary_enrichment/test_enrich_syllabary.py
  - transcription/syllabary_enrichment/test_evaluate_reconciliation.py
ordinal: 395300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
transcription/syllabary_enrichment/enrich_syllabary.py and alignment_engine.py forward to transcription.cherokee.enrichment.syllable_alignment. Update all callers in evaluation and tools to import directly from Tier 2 transcription.cherokee.enrichment or Tier 3 transcription.pipelines.enrichment, and prune the deprecated shim files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 enrich_syllabary.py and legacy shims in transcription/syllabary_enrichment/ pruned
- [x] #2 Call sites updated to transcription.cherokee.enrichment
- [x] #3 All tests pass and pyright reports 0 errors
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update caller in transcription/pipelines/scripture/pipeline.py to import directly from transcription.cherokee.enrichment.\n2. Update documentation and analysis markdown files referencing transcription.syllabary_enrichment to reference transcription.cherokee.enrichment or transcription.pipelines.enrichment.\n3. Delete legacy transcription/syllabary_enrichment/ package and all its contents.\n4. Run pytest and pyright transcription to verify zero regressions.\n5. Commit changes and track modified/deleted files.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Pruned legacy transcription/syllabary_enrichment forwarding shims and obsolete files. Updated call sites across transcription/pipelines/scripture/pipeline.py, docs/alignment.md, docs/syllabary_enrichment.md, and syllabary_enrichment_failure_analysis.md to import directly from Tier 2 transcription.cherokee.enrichment and Tier 3 transcription.pipelines.enrichment. Verified all 399 unit and integration tests pass with pytest and 0 errors reported by pyright.
<!-- SECTION:FINAL_SUMMARY:END -->
