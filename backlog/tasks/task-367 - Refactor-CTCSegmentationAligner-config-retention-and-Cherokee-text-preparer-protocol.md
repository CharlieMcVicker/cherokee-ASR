---
id: TASK-367
title: >-
  Refactor CTCSegmentationAligner config retention and Cherokee text preparer
  protocol
status: Done
assignee:
  - '@agent'
created_date: '2026-09-24 14:17'
updated_date: '2026-09-24 14:24'
labels: []
dependencies: []
ordinal: 397300
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
CTCSegmentationAligner copies all config properties onto itself instead of holding the config object directly. In addition, Cherokee-specific phonotactic flags and tokens leak into Tier 1 core alignment instead of being configured in Tier 2 via dedicated preparer functions and a Cherokee config helper.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CTCSegmentationAligner holds self.config directly without decomposing individual properties
- [x] #2 CTCAlignerConfig in Tier 1 has language-agnostic defaults and no Cherokee-specific hardcoded tokens
- [x] #3 Tier 2 Cherokee provides create_cherokee_ctc_config helper with phonotactic defaults
- [x] #4 Tier 2 Cherokee provides prepare_cherokee_with_intrusion and prepare_cherokee_direct conforming to TextPreparerProtocol
- [x] #5 Pipelines and tests updated cleanly without deprecated shims and all tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Refactor CTCAlignerConfig in transcription.core.alignment.models to remove Cherokee hardcoded tokens and rename enforce_phonotactics.
2. Refactor CTCSegmentationAligner in transcription.core.alignment.ctc to retain self.config directly and clean up TextPreparerProtocol signature to (config, text, char_list=None, token_masks=None).
3. Create create_cherokee_ctc_config, prepare_cherokee_with_intrusion, and prepare_cherokee_direct in transcription.cherokee.phonotactics.
4. Update ScripturePipeline, DialogueAlignmentPipeline, and all test suites to use the new config factory and preparer functions.
5. Verify with pyright transcription and pytest.
6. Commit directly to dev.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Refactored CTCSegmentationAligner to hold self.config directly, cleaned CTCAlignerConfig to have language-agnostic defaults, added create_cherokee_ctc_config, prepare_cherokee_with_intrusion, and prepare_cherokee_direct in transcription.cherokee.phonotactics conforming to the simplified TextPreparerProtocol, and updated all pipelines and tests. Verified with pyright (0 errors) and pytest (394/394 passing).
<!-- SECTION:FINAL_SUMMARY:END -->
