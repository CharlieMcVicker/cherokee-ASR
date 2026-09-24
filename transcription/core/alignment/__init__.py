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
    WagnerFischerAligner,
)
from transcription.core.alignment.ctc import (
    CTCSegmentationAligner,
    TextPreparerProtocol,
    default_text_preparer,
)
from transcription.core.alignment.forced_aligner import (
    AlignedWordSpan,
    ForcedAlignerProtocol,
    MMSForcedAligner,
    get_default_forced_aligner,
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
    "WagnerFischerAligner",
    # CTC
    "CTCSegmentationAligner",
    "TextPreparerProtocol",
    "default_text_preparer",
    # Forced Aligner
    "AlignedWordSpan",
    "ForcedAlignerProtocol",
    "MMSForcedAligner",
    "get_default_forced_aligner",
]
