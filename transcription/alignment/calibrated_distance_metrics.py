# -*- coding: utf-8 -*-
"""
calibrated_distance_metrics.py

Phonologically-calibrated distance metric wrapper for Cherokee ASR alignment.
Re-exported from transcription.cherokee.distance for backwards compatibility.
"""

from transcription.cherokee.distance import (
    CHEROKEE_VOWEL_DROP_COUNTS,
    CHEROKEE_VOWEL_DROP_PROBABILITIES,
    DEFAULT_CALIBRATED_DELETION_COSTS,
    DEFAULT_CALIBRATED_INSERTION_COSTS,
    DEFAULT_CONFUSION_COST_MATRIX_PATH,
    ConfusionMatrixCostMetric,
    PhonologicalConfusionCostMetric,
)
from transcription.core.alignment.distance import DistanceMetric

__all__ = [
    "CHEROKEE_VOWEL_DROP_COUNTS",
    "CHEROKEE_VOWEL_DROP_PROBABILITIES",
    "DEFAULT_CALIBRATED_DELETION_COSTS",
    "DEFAULT_CALIBRATED_INSERTION_COSTS",
    "DEFAULT_CONFUSION_COST_MATRIX_PATH",
    "DistanceMetric",
    "PhonologicalConfusionCostMetric",
    "ConfusionMatrixCostMetric",
]
