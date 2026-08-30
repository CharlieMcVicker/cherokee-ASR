---
id: TASK-235.5
title: 'Connect Compatibility Facade, Update align_cli, and Verify Tests'
status: Done
assignee:
  - '@subagent'
created_date: '2026-08-30 22:45'
updated_date: '2026-08-30 22:50'
labels: []
dependencies: []
parent_task_id: TASK-235
ordinal: 234000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rewire transcription.timestamping.aligner and align_cli to use new ports & adapters backend while maintaining full backward compatibility
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Ensure align_audio_segment, align_tokens_to_verses, and align_emissions_to_text delegate cleanly to new engine
- [x] #2 Ensure all timestamping and new_testament pipeline tests pass in cherokee-asr environment
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Re-routed legacy aligner façade (align_audio_segment, align_tokens_to_verses, align_emissions_to_text, compute_alignment_metrics, compute_trigram_edit_cost, _align_words_char_range) to the new transcription.alignment core engine and adapters. Verified all 93 tests in the transcription test suite pass cleanly.
<!-- SECTION:FINAL_SUMMARY:END -->
