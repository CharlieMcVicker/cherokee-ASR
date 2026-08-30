"""
Port definitions (Protocols) for alignment engine components.
"""

from transcription.alignment.ports.protocols import (
    ASREmissionsExtractor,
    ChunkAlignmentEngine,
    DistanceMetric,
    PhoneticPreprocessor,
    ReconciliationStrategy,
)

__all__ = [
    "ASREmissionsExtractor",
    "ChunkAlignmentEngine",
    "DistanceMetric",
    "PhoneticPreprocessor",
    "ReconciliationStrategy",
]
