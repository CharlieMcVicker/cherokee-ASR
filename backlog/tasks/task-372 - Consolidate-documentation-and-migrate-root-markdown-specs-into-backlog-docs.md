---
id: TASK-372
title: Consolidate documentation and migrate root markdown specs into backlog/docs
status: Done
assignee:
  - '@agent'
created_date: '2026-09-24 16:28'
updated_date: '2026-09-24 16:32'
labels: []
dependencies: []
ordinal: 405300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The root directory contains loose markdown analysis and specification files (segment_sweep.md, syllabary ASR enrichment.md, syllabary_enrichment_failure_analysis.md, training test valid csv plan.md, wav2vec2_format.md, write_syllabary_spec.md) alongside the docs/ directory. Consolidating all project documentation and specs into structured folders under backlog/docs establishes a single source of truth and cleans up root clutter.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Root markdown files are categorized and migrated into backlog/docs/ (e.g. specs, analysis, archive)
- [x] #2 docs/ directory contents are consolidated with backlog/docs/
- [x] #3 References in AGENTS.md and README.md are updated to the consolidated documentation paths
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create docs/specs/ and docs/archive/ subdirectories\n2. Move loose root specs (training test valid csv plan.md, wav2vec2_format.md, write_syllabary_spec.md, syllabary_enrichment_failure_analysis.md) and docs/spec_*.md to docs/specs/\n3. Move historical notes (segment_sweep.md, syllabary ASR enrichment.md) to docs/archive/\n4. Update documentation references in AGENTS.md and README.md\n5. Verify that no loose markdown files remain in repository root (except README.md and AGENTS.md)
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Consolidated technical documentation by creating docs/specs/ and docs/archive/ subdirectories. Migrated all loose root markdown files (training test valid csv plan.md, wav2vec2_format.md, write_syllabary_spec.md, syllabary_enrichment_failure_analysis.md) and docs/spec_*.md to docs/specs/, and historical experiment notes (segment_sweep.md, syllabary ASR enrichment.md) to docs/archive/. Verified README.md and AGENTS.md links and verified no loose markdown files remain at repository root.
<!-- SECTION:FINAL_SUMMARY:END -->
