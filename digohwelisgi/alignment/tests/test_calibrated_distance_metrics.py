# -*- coding: utf-8 -*-
"""
Unit tests for PhonologicalConfusionCostMetric in digohwelisgi.alignment.calibrated_distance_metrics.
"""

import json
from pathlib import Path
import pytest

from digohwelisgi.alignment import (
    NeedlemanWunschWordAligner,
    SlidingWindowDTWAligner,
    TextChunk,
    TokenEmission,
)
from digohwelisgi.core.alignment.distance import (
    ConfusionMatrixCostMetric,
    DistanceMetric,
)
from digohwelisgi.cherokee.distance import (
    CHEROKEE_VOWEL_DROP_COUNTS,
    CHEROKEE_VOWEL_DROP_PROBABILITIES,
    DEFAULT_CALIBRATED_DELETION_COSTS,
    DEFAULT_CALIBRATED_INSERTION_COSTS,
    PhonologicalConfusionCostMetric,
)


def test_protocol_verification():
    metric = PhonologicalConfusionCostMetric()
    assert isinstance(metric, DistanceMetric)


def test_identical_and_empty_strings():
    metric = PhonologicalConfusionCostMetric(
        base_metric=ConfusionMatrixCostMetric(unigram_costs={})
    )
    assert metric.compute_cost("", "") == 0.0
    assert metric.compute_cost("osiyo", "osiyo") == 0.0
    assert metric.compute_cost("a", "a") == 0.0


def test_vowel_drop_counts_and_hierarchy():
    # Verify vowel drop probabilities match gathered empirical counts
    assert "i" in CHEROKEE_VOWEL_DROP_COUNTS
    assert CHEROKEE_VOWEL_DROP_COUNTS["i"] == (580, 4804)
    assert CHEROKEE_VOWEL_DROP_PROBABILITIES["i"] == 580 / 4804

    # Verify hierarchy of calibrated default deletion costs:
    # 'i' has highest drop rate -> lowest deletion cost penalty
    # Hierarchy: i < v < o < a < u < e < consonant (1.0)
    del_costs = DEFAULT_CALIBRATED_DELETION_COSTS
    assert del_costs["i"] < del_costs["v"]
    assert del_costs["v"] < del_costs["o"]
    assert del_costs["o"] < del_costs["a"]
    assert del_costs["a"] < del_costs["u"]
    assert del_costs["u"] < del_costs["e"]
    assert del_costs["e"] < 1.0


def test_calibrated_vowel_deletion_penalties():
    metric = PhonologicalConfusionCostMetric(
        base_metric=ConfusionMatrixCostMetric(unigram_costs={}),
        default_deletion_cost=1.0,
        default_insertion_cost=1.0,
    )

    # Deleting vowel vs consonant from reference
    i_del_cost = DEFAULT_CALIBRATED_DELETION_COSTS["i"]
    assert (
        pytest.approx(metric.compute_cost(hypothesis="", reference="i"), 0.001)
        == i_del_cost
    )

    # Consonant deletion uses default deletion cost 1.0
    assert (
        pytest.approx(metric.compute_cost(hypothesis="", reference="k"), 0.001) == 1.0
    )
    assert (
        pytest.approx(metric.compute_cost(hypothesis="", reference="t"), 0.001) == 1.0
    )
    assert (
        pytest.approx(metric.compute_cost(hypothesis="", reference="s"), 0.001) == 1.0
    )

    # In-word vowel drop vs consonant drop:
    # "osiyo" -> "osyo" (dropped 'i', len=5): cost is i_del_cost / 5
    vowel_drop_cost = metric.compute_cost(hypothesis="osyo", reference="osiyo")
    assert pytest.approx(vowel_drop_cost, 0.001) == i_del_cost / 5.0

    # "osiyo" -> "oiyo" (dropped 's', len=5): cost is 1.0 / 5 = 0.20
    consonant_drop_cost = metric.compute_cost(hypothesis="oiyo", reference="osiyo")
    assert pytest.approx(consonant_drop_cost, 0.001) == 1.0 / 5.0
    assert vowel_drop_cost < consonant_drop_cost


def test_custom_vowel_deletion_penalties():
    custom_deletions = {
        "i": 0.46,
        "v": 0.60,
        "o": 0.63,
        "a": 0.68,
        "u": 0.87,
        "e": 0.89,
    }
    metric = PhonologicalConfusionCostMetric(
        base_metric=ConfusionMatrixCostMetric(unigram_costs={}),
        custom_deletion_costs=custom_deletions,
        default_deletion_cost=1.0,
        default_insertion_cost=1.0,
    )

    assert (
        pytest.approx(metric.compute_cost(hypothesis="", reference="i"), 0.001) == 0.46
    )
    assert (
        pytest.approx(metric.compute_cost(hypothesis="", reference="v"), 0.001) == 0.60
    )
    assert (
        pytest.approx(metric.compute_cost(hypothesis="", reference="o"), 0.001) == 0.63
    )
    assert (
        pytest.approx(metric.compute_cost(hypothesis="", reference="a"), 0.001) == 0.68
    )
    assert (
        pytest.approx(metric.compute_cost(hypothesis="", reference="u"), 0.001) == 0.87
    )
    assert (
        pytest.approx(metric.compute_cost(hypothesis="", reference="e"), 0.001) == 0.89
    )

    # In-word vowel drop
    vowel_drop_cost = metric.compute_cost(hypothesis="osyo", reference="osiyo")
    assert pytest.approx(vowel_drop_cost, 0.001) == 0.46 / 5.0


def test_calibrated_aspiration_insertion_discount():
    metric = PhonologicalConfusionCostMetric(
        base_metric=ConfusionMatrixCostMetric(unigram_costs={}),
        default_deletion_cost=1.0,
        default_insertion_cost=1.0,
    )

    # Inserting 'h' when reference is empty
    assert (
        pytest.approx(metric.compute_cost(hypothesis="h", reference=""), 0.001) == 0.10
    )

    # Inserting consonant 'k' when reference is empty
    assert (
        pytest.approx(metric.compute_cost(hypothesis="k", reference=""), 0.001) == 1.0
    )

    # In-word aspiration insertion vs consonant insertion:
    # "osiyo" -> "ohsiyo" (inserted 'h', len=5): cost is 0.10 / 5 = 0.02
    h_ins_cost = metric.compute_cost(hypothesis="ohsiyo", reference="osiyo")
    assert pytest.approx(h_ins_cost, 0.001) == 0.10 / 5.0

    # "osiyo" -> "oksiyo" (inserted 'k', len=5): cost is 1.0 / 5 = 0.20
    k_ins_cost = metric.compute_cost(hypothesis="oksiyo", reference="osiyo")
    assert pytest.approx(k_ins_cost, 0.001) == 1.0 / 5.0
    assert h_ins_cost < k_ins_cost


def test_full_string_alignment_with_substitutions_vowels_and_aspiration():
    # Base metric with substitution 'k' -> 'g' costing 0.2
    base = ConfusionMatrixCostMetric(
        unigram_costs={"k": {"g": 0.2}},
        default_substitution_cost=1.0,
    )
    metric = PhonologicalConfusionCostMetric(
        base_metric=base,
        custom_deletion_costs={"i": 0.46},
        custom_insertion_costs={"h": 0.10},
    )

    # Reference: "kani" (len=4)
    # Hypothesis: "gahni"
    # Operations:
    # - 'k' -> 'g': substitution cost = 0.2
    # - 'a': match = 0.0
    # - 'h': insertion cost = 0.10
    # - 'n': match = 0.0
    # - 'i': match = 0.0
    # Total DP raw cost = 0.2 + 0.10 = 0.30 -> normalized = 0.30 / 4 = 0.075
    cost = metric.compute_cost(hypothesis="gahni", reference="kani")
    assert pytest.approx(cost, 0.001) == 0.30 / 4.0

    # Reference: "kanii" (len=5) -> hypothesis "gahni" (sub k->g: 0.2, ins h: 0.1, drop i: 0.46)
    # Total DP raw cost = 0.2 + 0.10 + 0.46 = 0.76 -> normalized = 0.76 / 5 = 0.152
    cost_drop = metric.compute_cost(hypothesis="gahni", reference="kanii")
    assert pytest.approx(cost_drop, 0.001) == 0.76 / 5.0


def test_from_json_and_from_base_metric(tmp_path: Path):
    artifact_data = {
        "unigram_costs": {
            "a": {"e": 0.25},
        },
        "insertion_cost": 1.0,
        "deletion_cost": 1.0,
        "default_substitution_cost": 0.9,
    }
    artifact_file = tmp_path / "test_costs.json"
    artifact_file.write_text(json.dumps(artifact_data), encoding="utf-8")

    # Instantiate from_json
    metric_from_json = PhonologicalConfusionCostMetric.from_json(artifact_file)
    assert isinstance(metric_from_json, DistanceMetric)
    assert (
        metric_from_json.custom_deletion_costs["i"]
        == DEFAULT_CALIBRATED_DELETION_COSTS["i"]
    )
    assert metric_from_json.custom_insertion_costs["h"] == 0.10

    # Verify substitution from base artifact
    # ref="a", hyp="e" -> sub cost = 0.25 / 1 = 0.25
    assert pytest.approx(metric_from_json.compute_cost("e", "a"), 0.001) == 0.25

    # Verify disk artifact was not modified
    re_read = json.loads(artifact_file.read_text(encoding="utf-8"))
    assert re_read == artifact_data

    # Instantiate from_base_metric
    base_metric = ConfusionMatrixCostMetric(unigram_costs={"o": {"u": 0.15}})
    metric_from_base = PhonologicalConfusionCostMetric.from_base_metric(
        base_metric=base_metric,
        custom_deletion_costs={"i": 0.30},
        custom_insertion_costs={"h": 0.05},
    )
    assert metric_from_base.custom_deletion_costs["i"] == 0.30
    assert metric_from_base.custom_insertion_costs["h"] == 0.05
    assert pytest.approx(metric_from_base.compute_cost("u", "o"), 0.001) == 0.15
    assert pytest.approx(metric_from_base.compute_cost("", "i"), 0.001) == 0.30
    assert pytest.approx(metric_from_base.compute_cost("h", ""), 0.001) == 0.05


def test_integration_with_needleman_wunsch_word_aligner():
    metric = PhonologicalConfusionCostMetric()
    aligner = NeedlemanWunschWordAligner(
        distance_metric=metric,
        gap_cost=0.8,
        fuse_penalty=0.15,
    )

    words = ["osiyo", "tohiju"]
    tokens = [
        # Hypothesis with 'i' dropped in "osyo" and aspiration added in "thohiju"
        TokenEmission(word="osyo", start_sec=0.1, end_sec=0.6, confidence=0.95),
        TokenEmission(word="thohiju", start_sec=0.7, end_sec=1.2, confidence=0.92),
    ]

    intervals = aligner.align_words(raw_words=words, matched_tokens=tokens)
    assert len(intervals) == 2
    assert intervals[0].word == "osiyo"
    assert intervals[0].start_sec == 0.1
    assert intervals[0].end_sec == 0.6
    assert intervals[1].word == "tohiju"
    assert intervals[1].start_sec == 0.7
    assert intervals[1].end_sec == 1.2


def test_integration_with_sliding_window_dtw_aligner():
    metric = PhonologicalConfusionCostMetric()
    word_aligner = NeedlemanWunschWordAligner(distance_metric=metric)
    dtw_aligner = SlidingWindowDTWAligner(word_aligner=word_aligner)

    chunks = [
        TextChunk(chunk_id="chk_01", text="osiyo tohiju"),
    ]
    tokens = [
        TokenEmission(word="osyo", start_sec=0.1, end_sec=0.6, confidence=0.95),
        TokenEmission(word="tohiju", start_sec=0.7, end_sec=1.2, confidence=0.92),
    ]

    output = dtw_aligner.align(chunks=chunks, emissions=tokens)
    assert len(output.aligned_chunks) == 1
    assert output.aligned_chunks[0].chunk_id == "chk_01"
    assert len(output.aligned_chunks[0].words) == 2
    assert output.aligned_chunks[0].words[0].word == "osiyo"
    assert output.aligned_chunks[0].words[1].word == "tohiju"
