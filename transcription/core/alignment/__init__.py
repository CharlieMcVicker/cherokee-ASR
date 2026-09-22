# -*- coding: utf-8 -*-
"""
transcription.core.alignment

Pure core alignment domain models, dynamic programming aligners, distance metrics,
and CTC trellis segmentation engines.
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
from transcription.core.alignment.distance import (
    CharacterErrorRateMetric,
    ConfusionMatrixCostMetric,
    CustomCallableDistanceMetric,
    DefaultCERDistanceMetric,
    DistanceMetric,
    LevenshteinDistanceMetric,
    PhonologicalDistanceMetric,
    calculate_cer,
)
from transcription.core.alignment.dp import (
    NeedlemanWunschWordAligner,
    SlidingWindowDTWAligner,
)
from transcription.core.alignment.ctc import (
    CTCSegmentationAligner,
    TextPreparerProtocol,
    default_text_preparer,
)

__all__ = [
    # Models
    "CTCAlignerConfig",
    "TokenEmission",
    "TextChunk",
    "WordInterval",
    "AlignedChunk",
    "AlignmentMetrics",
    "AlignmentOutput",
    # Distance
    "DistanceMetric",
    "DefaultCERDistanceMetric",
    "CharacterErrorRateMetric",
    "PhonologicalDistanceMetric",
    "LevenshteinDistanceMetric",
    "CustomCallableDistanceMetric",
    "ConfusionMatrixCostMetric",
    "calculate_cer",
    # DP
    "NeedlemanWunschWordAligner",
    "SlidingWindowDTWAligner",
    # CTC
    "CTCSegmentationAligner",
    "TextPreparerProtocol",
    "default_text_preparer",
]
