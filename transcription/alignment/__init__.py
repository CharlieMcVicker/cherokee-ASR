"""
transcription.alignment package.

Provides modular ports-and-adapters architecture for audio and text chunk alignment.
"""

from transcription.alignment.domain.models import (
    AlignedChunk,
    AlignmentMetrics,
    AlignmentOutput,
    TextChunk,
    TokenEmission,
    WordInterval,
)
from transcription.alignment.pipeline import AlignmentPipeline
from transcription.alignment.strategies.extractors import (
    CallbackEmissionsExtractor,
    CherokeeASRExtractor,
    PrecomputedEmissionsExtractor,
)

__all__ = [
    "AlignedChunk",
    "AlignmentMetrics",
    "AlignmentOutput",
    "AlignmentPipeline",
    "CallbackEmissionsExtractor",
    "CherokeeASRExtractor",
    "PrecomputedEmissionsExtractor",
    "TextChunk",
    "TokenEmission",
    "WordInterval",
]
