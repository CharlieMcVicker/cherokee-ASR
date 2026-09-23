# -*- coding: utf-8 -*-
"""
Core domain models for transcription and timestamping alignment.

Re-exported from transcription.core.alignment.models for backward compatibility.
"""

from transcription.core.alignment.models import (
    AlignedChunk,
    AlignmentMetrics,
    AlignmentOutput,
    CTCAlignerConfig,
    TextChunk,
    TokenEmission,
    WordInterval,
)

__all__ = [
    "AlignedChunk",
    "AlignmentMetrics",
    "AlignmentOutput",
    "CTCAlignerConfig",
    "TextChunk",
    "TokenEmission",
    "WordInterval",
]
