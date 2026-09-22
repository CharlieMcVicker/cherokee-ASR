# -*- coding: utf-8 -*-
"""
Distance metric implementations for alignment.

Re-exported from transcription.core.alignment.distance for backward compatibility.
"""

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

__all__ = [
    "DistanceMetric",
    "calculate_cer",
    "DefaultCERDistanceMetric",
    "CharacterErrorRateMetric",
    "PhonologicalDistanceMetric",
    "LevenshteinDistanceMetric",
    "CustomCallableDistanceMetric",
    "ConfusionMatrixCostMetric",
]
