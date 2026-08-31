---
id: TASK-244
title: >-
  Refactor AlignmentPipeline to require all arguments at construction and
  decouple exporters from run()
status: Done
assignee:
  - '@antigravity'
created_date: '2026-08-31 21:09'
updated_date: '2026-08-31 21:09'
labels: []
dependencies: []
ordinal: 246000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove make_default from AlignmentPipeline since all arguments (chunk_adapter, extractor, engine, exporters) are critical. Refactor pipeline.run() to take only (audio_input, output_dir=None) and run the injected self.exporters. Update cli.py and all test callers to explicitly construct and pass exporters to AlignmentPipeline.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Remove make_default from AlignmentPipeline
- [x] #2 Simplify AlignmentPipeline.run() signature and implementation to execute injected self.exporters without boolean flags
- [x] #3 Update cli.py and tests to configure exporters at pipeline construction
- [x] #4 Verify pyright and pytest pass with 0 errors
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. In transcription/alignment/pipeline.py, remove AlignmentPipeline.make_default.
2. In AlignmentPipeline.run(self, audio_input: Any, output_dir: Optional[str] = None) -> AlignmentOutput, remove export_praat, export_manifest, debug_export flags and hardcoded adapter instantiations. Simply iterate through self.exporters and call exporter.export(alignment, os.path.join(output_dir, filename)).
3. In transcription/alignment/cli.py, configure the list of (OutboundAlignmentAdapter, filename) tuples based on CLI flags (--export-praat, --export-manifest, --debug-export if custom exporter or exporter protocol) and pass them explicitly into AlignmentPipeline.
4. Update test_pipeline.py and test_cli.py.
5. Verify pyright and pytest.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Refactored AlignmentPipeline to require all arguments (chunk_adapter, extractor, engine, exporters) explicitly at construction and removed make_default. Cleaned pipeline.run() to only accept (audio_input, output_dir) and execute injected exporters. Updated cli.py and test_pipeline.py to pass configured exporters at pipeline creation. Verified 0 pyright errors and 86/86 pytest tests pass.
<!-- SECTION:FINAL_SUMMARY:END -->
