# -*- coding: utf-8 -*-
"""
exporters.py

Pure export functions for saving AlignmentOutput into Praat .TextGrid, alignment_manifest.json,
and alignment_debug.json formats.

Re-exports core language-agnostic exporters from digohwelisgi.core.exporters.
"""

from digohwelisgi.core.exporters.manifest import (
    export_debug_json,
    export_manifest,
    serialize_word_interval,
)
from digohwelisgi.core.exporters.textgrid import (
    IntervalTier,
    TextGridBuilder,
    build_contiguous_intervals,
    build_padded_word_intervals,
    export_textgrid,
)

__all__ = [
    "IntervalTier",
    "TextGridBuilder",
    "build_contiguous_intervals",
    "build_padded_word_intervals",
    "serialize_word_interval",
    "export_textgrid",
    "export_manifest",
    "export_debug_json",
]
