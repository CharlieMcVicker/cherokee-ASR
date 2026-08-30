"""
Concrete strategy implementations for preprocessors, distance metrics, and reconciliation.
"""

from transcription.alignment.strategies.distance_metrics import (
    DefaultCERDistanceMetric,
    PhonologicalDistanceMetric,
)
from transcription.alignment.strategies.extractors import (
    CallbackEmissionsExtractor,
    CherokeeASRExtractor,
    PrecomputedEmissionsExtractor,
)
from transcription.alignment.strategies.preprocessors import (
    CherokeePhoneticPreprocessor,
    SyllabaryToPhoneticPreprocessor,
)
from transcription.alignment.strategies.reconciliation import (
    CherokeeSyllabaryReconciliationStrategy,
)

__all__ = [
    "CallbackEmissionsExtractor",
    "CherokeeASRExtractor",
    "CherokeePhoneticPreprocessor",
    "CherokeeSyllabaryReconciliationStrategy",
    "DefaultCERDistanceMetric",
    "PhonologicalDistanceMetric",
    "PrecomputedEmissionsExtractor",
    "SyllabaryToPhoneticPreprocessor",
]
