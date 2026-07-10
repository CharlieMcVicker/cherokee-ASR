---
id: TASK-131
title: Add noisy segment entries to segmentation manifest
status: Done
assignee:
  - '@antigravity'
created_date: '2026-07-10 14:23'
updated_date: '2026-07-10 14:23'
labels: []
dependencies: []
ordinal: 127000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Append rows to segmentation_manifest.csv mapping the newly extracted raw (noisy) segments with their correct relative paths, matching the existing schema.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Parse the existing manifest to identify the processed MP3 files
- [x] #2 Generate new rows for the noisy segments pointing to data/processed/noisy_segments
- [x] #3 Append the generated rows to segmentation_manifest.csv
- [x] #4 Verify manifest size and row counts
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read existing rows in segmentation_manifest.csv.\n2. Filter the rows to identify the ones corresponding to MP3 files.\n3. Generate a matching row for each, replacing 'denoised_segments' with 'noisy_segments' in the segmented_filepath, but preserving original_filename, start_ms, and end_ms.\n4. Append the new rows to segmentation_manifest.csv.\n5. Verify that the new entries are correctly added and row counts match.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Successfully parsed segmentation_manifest.csv and identified 52,644 denoised interview MP3 segment rows. Generated corresponding entries pointing to the newly created undenoised (noisy) segments by replacing 'denoised_segments' with 'noisy_segments' in the segmented_filepath, while keeping other attributes. Appended the 52,644 new rows back to the manifest, bringing the total entries to 107,351 rows.
<!-- SECTION:FINAL_SUMMARY:END -->
