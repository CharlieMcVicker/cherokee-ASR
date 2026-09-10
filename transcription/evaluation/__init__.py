# -*- coding: utf-8 -*-
"""
transcription.evaluation

Perturbation-driven evaluation and phonetic confusion analysis engine.
"""

from transcription.evaluation.confusion import (
    ConfusionAccumulator,
    character_levenshtein_align,
)
from transcription.evaluation.cost_engine import (
    ConfusionCostEngine,
    probability_to_normalized_cost,
)
from transcription.evaluation.evaluator import (
    EvaluationRecord,
    NoisyEvaluator,
    calculate_cer,
)
from transcription.evaluation.manifold import PhoneticManifoldAnalyzer
from transcription.evaluation.perturbations import (
    AcousticFilter,
    AdditiveNoise,
    AudioTransform,
    ComposeTransforms,
    TimeWarp,
)
from transcription.evaluation.visualizer import ManifoldVisualizer

__all__ = [
    "EvaluationRecord",
    "NoisyEvaluator",
    "calculate_cer",
    "ConfusionAccumulator",
    "character_levenshtein_align",
    "ConfusionCostEngine",
    "probability_to_normalized_cost",
    "PhoneticManifoldAnalyzer",
    "ManifoldVisualizer",
    "AudioTransform",
    "AdditiveNoise",
    "TimeWarp",
    "AcousticFilter",
    "ComposeTransforms",
]
