# -*- coding: utf-8 -*-
"""
digohwelisgi.evaluation

Perturbation-driven evaluation and phonetic confusion analysis engine.
"""

from digohwelisgi.evaluation.confusion import (
    ConfusionAccumulator,
    character_levenshtein_align,
)
from digohwelisgi.evaluation.cost_engine import (
    ConfusionCostEngine,
    probability_to_normalized_cost,
)
from digohwelisgi.evaluation.evaluator import (
    EvaluationRecord,
    NoisyEvaluator,
    calculate_cer,
)
from digohwelisgi.evaluation.manifold import PhoneticManifoldAnalyzer
from digohwelisgi.evaluation.perturbations import (
    AcousticFilter,
    AdditiveNoise,
    AudioTransform,
    ComposeTransforms,
    TimeWarp,
)
from digohwelisgi.evaluation.visualizer import ManifoldVisualizer

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
