# -*- coding: utf-8 -*-
"""
Unit tests for ConfusionMatrixCostMetric in digohwelisgi.alignment.distance_metrics.
"""

import json
import pytest
from pathlib import Path

from digohwelisgi.alignment import (
    ConfusionMatrixCostMetric,
    DistanceMetric,
    NeedlemanWunschWordAligner,
    TokenEmission,
)


def test_protocol_verification():
    metric = ConfusionMatrixCostMetric(unigram_costs={})
    assert isinstance(metric, DistanceMetric)


def test_identical_strings_and_empty_cases():
    metric = ConfusionMatrixCostMetric(
        unigram_costs={
            "a": {"b": 0.3},
            "k": {"g": 0.2},
        },
        insertion_cost=1.0,
        deletion_cost=1.0,
        default_substitution_cost=1.0,
    )

    # Identical strings cost 0.0
    assert metric.compute_cost("osiyo", "osiyo") == 0.0
    assert metric.compute_cost("a", "a") == 0.0
    assert metric.compute_cost("", "") == 0.0

    # Empty vs non-empty
    assert metric.compute_cost("abc", "") == 3.0
    assert metric.compute_cost("", "abc") == 1.0  # 3 deletions normalized by 3 -> 1.0


def test_asymmetric_substitution_costs():
    # Asymmetric costs: ref 'k' -> hyp 'g' costs 0.2, but ref 'g' -> hyp 'k' costs 0.8
    unigram_costs = {
        "k": {"g": 0.2},
        "g": {"k": 0.8},
    }
    metric = ConfusionMatrixCostMetric(
        unigram_costs=unigram_costs,
        insertion_cost=1.0,
        deletion_cost=1.0,
        default_substitution_cost=1.0,
    )

    # Check direct get_sub_cost
    assert metric.get_sub_cost("k", "g") == 0.2
    assert metric.get_sub_cost("g", "k") == 0.8
    assert metric.get_sub_cost("k", "k") == 0.0
    assert metric.get_sub_cost("k", "z") == 1.0

    # Asymmetric edit distance
    # ref="ka", hyp="ga" -> ref[0]='k', hyp[0]='g' -> cost = 0.2 / 2 = 0.1
    cost_ka_to_ga = metric.compute_cost(hypothesis="ga", reference="ka")
    # ref="ga", hyp="ka" -> ref[0]='g', hyp[0]='k' -> cost = 0.8 / 2 = 0.4
    cost_ga_to_ka = metric.compute_cost(hypothesis="ka", reference="ga")

    assert pytest.approx(cost_ka_to_ga, 0.001) == 0.1
    assert pytest.approx(cost_ga_to_ka, 0.001) == 0.4
    assert cost_ka_to_ga != cost_ga_to_ka


def test_from_json_artifact(tmp_path: Path):
    # Test loading structured artifact with metadata and parameters
    artifact_data = {
        "unigram_costs": {
            "a": {"e": 0.25},
            "o": {"u": 0.35},
        },
        "insertion_cost": 1.2,
        "deletion_cost": 0.9,
        "default_substitution_cost": 0.95,
        "metadata": {
            "source": "perturbation_sweep",
            "iterations": 100,
        },
    }
    artifact_file = tmp_path / "confusion_costs.json"
    artifact_file.write_text(json.dumps(artifact_data), encoding="utf-8")

    metric = ConfusionMatrixCostMetric.from_json(artifact_file)
    assert isinstance(metric, DistanceMetric)
    assert metric.insertion_cost == 1.2
    assert metric.deletion_cost == 0.9
    assert metric.default_substitution_cost == 0.95
    assert metric.get_sub_cost("a", "e") == 0.25
    assert metric.get_sub_cost("o", "u") == 0.35
    assert metric.get_sub_cost("a", "z") == 0.95

    # Test loading raw unigram dictionary JSON
    raw_dict_data = {
        "a": {"e": 0.3},
    }
    raw_file = tmp_path / "raw_costs.json"
    raw_file.write_text(json.dumps(raw_dict_data), encoding="utf-8")

    raw_metric = ConfusionMatrixCostMetric.from_json(
        raw_file,
        insertion_cost=1.5,
        deletion_cost=1.5,
        default_substitution_cost=1.0,
    )
    assert raw_metric.insertion_cost == 1.5
    assert raw_metric.deletion_cost == 1.5
    assert raw_metric.get_sub_cost("a", "e") == 0.3
    assert raw_metric.get_sub_cost("a", "x") == 1.0


def test_integration_with_needleman_wunsch_word_aligner():
    unigram_costs = {
        "o": {"u": 0.15},
        "s": {"sh": 0.2},
    }
    metric = ConfusionMatrixCostMetric(unigram_costs=unigram_costs)

    aligner = NeedlemanWunschWordAligner(
        distance_metric=metric,
        gap_cost=0.8,
        fuse_penalty=0.15,
    )

    words = ["osiyo", "tohiju"]
    tokens = [
        TokenEmission(word="osiyo", start_sec=0.1, end_sec=0.6, confidence=0.95),
        TokenEmission(word="tohiju", start_sec=0.7, end_sec=1.2, confidence=0.92),
    ]

    intervals = aligner.align_words(raw_words=words, matched_tokens=tokens)
    assert len(intervals) == 2
    assert intervals[0].word == "osiyo"
    assert intervals[0].start_sec == 0.1
    assert intervals[0].end_sec == 0.6
    assert intervals[1].word == "tohiju"
    assert intervals[1].start_sec == 0.7
    assert intervals[1].end_sec == 1.2
