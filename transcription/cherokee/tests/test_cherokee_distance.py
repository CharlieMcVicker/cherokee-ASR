# -*- coding: utf-8 -*-
"""
test_cherokee_distance.py

Unit tests for Cherokee distance metrics: PhonologicalConfusionCostMetric and
ConfusionMatrixCostMetric in transcription.cherokee.distance.
"""

import json
from pathlib import Path
import pytest

from transcription.cherokee.distance import (
    CHEROKEE_VOWEL_DROP_COUNTS,
    CHEROKEE_VOWEL_DROP_PROBABILITIES,
    DEFAULT_CALIBRATED_DELETION_COSTS,
    DEFAULT_CALIBRATED_INSERTION_COSTS,
    ConfusionMatrixCostMetric,
    PhonologicalConfusionCostMetric,
)
from transcription.core.alignment.distance import DistanceMetric


def test_distance_metric_protocol_conformance():
    """Verify PhonologicalConfusionCostMetric and ConfusionMatrixCostMetric conform to DistanceMetric."""
    m1 = PhonologicalConfusionCostMetric()
    assert isinstance(m1, DistanceMetric)

    m2 = ConfusionMatrixCostMetric(unigram_costs={})
    assert isinstance(m2, DistanceMetric)


def test_identical_and_empty_strings():
    metric = PhonologicalConfusionCostMetric(
        base_metric=ConfusionMatrixCostMetric(unigram_costs={})
    )
    assert metric.compute_cost("", "") == 0.0
    assert metric.compute_cost("osiyo", "osiyo") == 0.0
    assert metric.compute_cost("a", "a") == 0.0


def test_vowel_drop_counts_and_hierarchy():
    assert "i" in CHEROKEE_VOWEL_DROP_COUNTS
    assert CHEROKEE_VOWEL_DROP_COUNTS["i"] == (580, 4804)
    assert CHEROKEE_VOWEL_DROP_PROBABILITIES["i"] == 580 / 4804

    # High drop frequency -> lower deletion penalty
    assert (
        DEFAULT_CALIBRATED_DELETION_COSTS["i"] < DEFAULT_CALIBRATED_DELETION_COSTS["e"]
    )
    assert (
        DEFAULT_CALIBRATED_DELETION_COSTS["i"] < DEFAULT_CALIBRATED_DELETION_COSTS["u"]
    )
    assert (
        DEFAULT_CALIBRATED_DELETION_COSTS["v"] < DEFAULT_CALIBRATED_DELETION_COSTS["e"]
    )


def test_aspiration_insertion_discount():
    assert DEFAULT_CALIBRATED_INSERTION_COSTS["h"] == 0.10

    metric = PhonologicalConfusionCostMetric(
        base_metric=ConfusionMatrixCostMetric(unigram_costs={})
    )
    # Inserting 'h' should have cost 0.10 / len(reference)
    cost = metric.compute_cost("hosiyo", "osiyo")
    assert cost < 0.25


def test_confusion_matrix_cost_metric_from_dict():
    costs = {
        "t": {"k": 0.2},
        "k": {"t": 0.8},
    }
    metric = ConfusionMatrixCostMetric(unigram_costs=costs)
    assert metric.get_sub_cost("t", "k") == 0.2
    assert metric.get_sub_cost("t", "t") == 0.0
    assert metric.get_sub_cost("t", "unknown") == 1.0
    assert metric.compute_cost("k", "t") == 0.2
