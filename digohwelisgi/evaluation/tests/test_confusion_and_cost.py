# -*- coding: utf-8 -*-
"""
Tests for Wagner-Fischer sequence alignment, ConfusionAccumulator, and ConfusionCostEngine.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from digohwelisgi.evaluation.confusion import (
    ConfusionAccumulator,
    character_levenshtein_align,
)
from digohwelisgi.evaluation.cost_engine import ConfusionCostEngine

# ============================================================================
# 1. Tests for character_levenshtein_align
# ============================================================================


def test_character_levenshtein_align_empty() -> None:
    assert character_levenshtein_align("", "") == []


def test_character_levenshtein_align_exact_match() -> None:
    pairs = character_levenshtein_align("cat", "cat")
    assert pairs == [
        ("c", "c", "match"),
        ("a", "a", "match"),
        ("t", "t", "match"),
    ]


def test_character_levenshtein_align_single_substitution() -> None:
    pairs = character_levenshtein_align("cat", "cut")
    assert pairs == [
        ("c", "c", "match"),
        ("a", "u", "substitution"),
        ("t", "t", "match"),
    ]


def test_character_levenshtein_align_deletion() -> None:
    pairs = character_levenshtein_align("abc", "ac")
    assert pairs == [
        ("a", "a", "match"),
        ("b", None, "deletion"),
        ("c", "c", "match"),
    ]


def test_character_levenshtein_align_insertion() -> None:
    pairs = character_levenshtein_align("ac", "abc")
    assert pairs == [
        ("a", "a", "match"),
        (None, "b", "insertion"),
        ("c", "c", "match"),
    ]


def test_character_levenshtein_align_all_deletion() -> None:
    pairs = character_levenshtein_align("abc", "")
    assert pairs == [
        ("a", None, "deletion"),
        ("b", None, "deletion"),
        ("c", None, "deletion"),
    ]


def test_character_levenshtein_align_all_insertion() -> None:
    pairs = character_levenshtein_align("", "xyz")
    assert pairs == [
        (None, "x", "insertion"),
        (None, "y", "insertion"),
        (None, "z", "insertion"),
    ]


def test_character_levenshtein_align_cherokee() -> None:
    # Cherokee: ᎣᏏᏲ (osiyo) vs ᎣᏏᏳ (osiyu)
    ref = "ᎣᏏᏲ"
    hyp = "ᎣᏏᏳ"
    pairs = character_levenshtein_align(ref, hyp)
    assert pairs == [
        ("Ꭳ", "Ꭳ", "match"),
        ("Ꮟ", "Ꮟ", "match"),
        ("Ᏺ", "Ᏻ", "substitution"),
    ]


def test_character_levenshtein_align_complex() -> None:
    ref = "kitten"
    hyp = "sitting"
    pairs = character_levenshtein_align(ref, hyp)
    # kitten -> sitting: sub k->s, match i, match t, match t, sub e->i, match n, ins g
    ops = [op for _, _, op in pairs]
    assert ops.count("match") == 4
    assert ops.count("substitution") == 2
    assert ops.count("insertion") == 1
    assert ops.count("deletion") == 0


# ============================================================================
# 2. Tests for ConfusionAccumulator
# ============================================================================


def test_confusion_accumulator_basic() -> None:
    vocab = ["a", "b", "c"]
    accumulator = ConfusionAccumulator(vocab=vocab)

    alignment = [
        ("a", "a", "match"),
        ("b", "c", "substitution"),
        ("c", None, "deletion"),
        (None, "a", "insertion"),
    ]
    accumulator.update_from_alignment(alignment)

    counts, labels = accumulator.get_raw_counts()
    assert labels == vocab
    # C[a, a] = 1, C[b, c] = 1, C[c, <del>] = 1, C[<ins>, a] = 1
    idx_a = labels.index("a")
    idx_b = labels.index("b")
    idx_c = labels.index("c")

    assert counts[idx_a, idx_a] == 1.0
    assert counts[idx_b, idx_c] == 1.0
    assert counts[idx_c, idx_c] == 0.0

    del_counts = accumulator.get_deletion_counts()
    assert del_counts == {"c": 1.0}

    ins_counts = accumulator.get_insertion_counts()
    assert ins_counts == {"a": 1.0}


def test_confusion_accumulator_top_k_soft_counts() -> None:
    vocab = ["c", "a", "u", "t", "o", "k"]
    accumulator = ConfusionAccumulator(vocab=vocab)

    # Reference: "cat", Hypothesis: "cut"
    alignment = [
        ("c", "c", "match"),
        ("a", "u", "substitution"),
        ("t", "t", "match"),
    ]
    # Top-K for hypothesis chars: 'c', 'u', 't'
    top_k = [
        [("c", 0.9), ("k", 0.1)],
        [("u", 0.7), ("o", 0.3)],
        [("t", 1.0)],
    ]

    accumulator.update_from_alignment(alignment, top_k_per_hyp_char=top_k)

    counts_dict = accumulator.counts
    assert counts_dict[("c", "c")] == pytest.approx(0.9)
    assert counts_dict[("c", "k")] == pytest.approx(0.1)
    assert counts_dict[("a", "u")] == pytest.approx(0.7)
    assert counts_dict[("a", "o")] == pytest.approx(0.3)
    assert counts_dict[("t", "t")] == pytest.approx(1.0)


def test_confusion_accumulator_dirichlet_smoothing() -> None:
    vocab = ["a", "b"]
    accumulator = ConfusionAccumulator(vocab=vocab)

    accumulator.add_count("a", "a", 9.0)
    accumulator.add_count("a", "b", 1.0)
    # Row for 'b' has 0 counts

    alpha = 0.5
    probs, labels = accumulator.get_conditional_probabilities(dirichlet_alpha=alpha)

    assert labels == ["a", "b"]
    # Row 0 ('a'): C = [9, 1], sum = 10. With alpha=0.5, |V|=2:
    # P[a, a] = (9 + 0.5) / (10 + 2 * 0.5) = 9.5 / 11.0
    # P[a, b] = (1 + 0.5) / (10 + 2 * 0.5) = 1.5 / 11.0
    assert probs[0, 0] == pytest.approx(9.5 / 11.0)
    assert probs[0, 1] == pytest.approx(1.5 / 11.0)
    assert np.sum(probs[0, :]) == pytest.approx(1.0)

    # Row 1 ('b'): C = [0, 0], sum = 0.
    # P[b, a] = (0 + 0.5) / (0 + 2 * 0.5) = 0.5 / 1.0 = 0.5
    # P[b, b] = (0 + 0.5) / (0 + 2 * 0.5) = 0.5 / 1.0 = 0.5
    assert probs[1, 0] == pytest.approx(0.5)
    assert probs[1, 1] == pytest.approx(0.5)
    assert np.sum(probs[1, :]) == pytest.approx(1.0)


def test_confusion_accumulator_dynamic_vocab() -> None:
    accumulator = ConfusionAccumulator()
    alignment = [
        ("x", "y", "substitution"),
        ("z", "z", "match"),
    ]
    accumulator.update_from_alignment(alignment)

    probs, labels = accumulator.get_conditional_probabilities(dirichlet_alpha=0.1)
    assert sorted(labels) == ["x", "y", "z"]
    assert probs.shape == (3, 3)
    # All row sums should be 1.0
    for row in probs:
        assert np.sum(row) == pytest.approx(1.0)


def test_confusion_accumulator_invalid_alpha() -> None:
    accumulator = ConfusionAccumulator(vocab=["a", "b"])
    with pytest.raises(ValueError, match="dirichlet_alpha must be non-negative"):
        accumulator.get_conditional_probabilities(dirichlet_alpha=-0.1)


# ============================================================================
# 3. Tests for ConfusionCostEngine
# ============================================================================


def test_confusion_cost_engine_diagonal_zero() -> None:
    engine = ConfusionCostEngine()
    labels = ["a", "b", "c"]
    cond_probs = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.05, 0.05, 0.9],
        ]
    )

    cost_dict = engine.compute_costs(cond_probs, labels, min_support=0)
    unigram_costs = cost_dict["unigram_costs"]

    for label in labels:
        assert unigram_costs[label][label] == 0.0


def test_confusion_cost_engine_clamped_log_scaling() -> None:
    engine = ConfusionCostEngine()
    labels = ["a", "b"]
    eps = 1e-5
    # Let P(b|a) = 0.5 and P(b|a) = 0.0001
    cond_probs = np.array(
        [
            [0.5, 0.5],
            [0.9999, 0.0001],
        ]
    )

    cost_dict = engine.compute_costs(cond_probs, labels, epsilon=eps)
    costs = cost_dict["unigram_costs"]

    # For P=0.5: ln(0.5 + 1e-5) / ln(1e-5)
    expected_cost_ab = float(np.clip(np.log(0.5 + eps) / np.log(eps), 0.0, 1.0))
    assert costs["a"]["b"] == pytest.approx(expected_cost_ab)

    # For P=0.9999: ln(0.9999 + 1e-5) / ln(1e-5) is very small (near 0)
    expected_cost_ba = float(np.clip(np.log(0.9999 + eps) / np.log(eps), 0.0, 1.0))
    assert costs["b"]["a"] == pytest.approx(expected_cost_ba)
    assert costs["b"]["a"] < costs["a"]["b"]


def test_confusion_cost_engine_min_support_fallback() -> None:
    engine = ConfusionCostEngine()
    labels = ["a", "b"]
    cond_probs = np.array(
        [
            [0.9, 0.1],
            [0.4, 0.6],
        ]
    )
    # Raw counts: 'a' has support 10, 'b' has support 2
    raw_counts = np.array(
        [
            [9.0, 1.0],
            [0.8, 1.2],
        ]
    )

    cost_dict = engine.compute_costs(
        cond_probs,
        labels,
        raw_counts=raw_counts,
        min_support=5,
        default_substitution_cost=1.0,
    )
    costs = cost_dict["unigram_costs"]

    # 'a' support = 10 >= 5 -> uses log cost
    assert costs["a"]["a"] == 0.0
    assert costs["a"]["b"] < 1.0

    # 'b' support = 2 < 5 -> fallback to default cost 1.0 for substitutions, 0.0 for identity
    assert costs["b"]["b"] == 0.0
    assert costs["b"]["a"] == 1.0


def test_confusion_cost_engine_validation_errors() -> None:
    engine = ConfusionCostEngine()
    labels = ["a", "b"]
    probs = np.eye(2)

    with pytest.raises(ValueError, match="cond_probs shape"):
        engine.compute_costs(np.eye(3), labels)

    with pytest.raises(ValueError, match="epsilon must be strictly between 0 and 1"):
        engine.compute_costs(probs, labels, epsilon=0.0)

    with pytest.raises(ValueError, match="epsilon must be strictly between 0 and 1"):
        engine.compute_costs(probs, labels, epsilon=1.5)

    with pytest.raises(ValueError, match="min_support must be non-negative"):
        engine.compute_costs(probs, labels, min_support=-1)

    with pytest.raises(ValueError, match="raw_counts row dimension"):
        engine.compute_costs(probs, labels, raw_counts=np.zeros((3, 3)))


def test_confusion_cost_engine_artifact_roundtrip(tmp_path: Path) -> None:
    engine = ConfusionCostEngine()
    labels = ["Ꭰ", "Ꭱ", "Ꭲ"]
    cond_probs = np.array(
        [
            [0.9, 0.08, 0.02],
            [0.05, 0.9, 0.05],
            [0.01, 0.09, 0.9],
        ]
    )

    cost_dict = engine.compute_costs(
        cond_probs,
        labels,
        min_support=5,
        epsilon=1e-5,
        default_substitution_cost=1.0,
        insertion_cost=1.0,
        deletion_cost=1.0,
    )

    artifact_path = tmp_path / "artifacts" / "confusion_costs.json"
    ConfusionCostEngine.save_cost_artifact(cost_dict, artifact_path)

    assert artifact_path.exists()
    loaded_dict = ConfusionCostEngine.load_cost_artifact(artifact_path)

    assert loaded_dict["vocabulary"] == labels
    assert loaded_dict["epsilon"] == 1e-5
    assert loaded_dict["min_support"] == 5
    assert loaded_dict["default_substitution_cost"] == 1.0
    assert loaded_dict["insertion_cost"] == 1.0
    assert loaded_dict["deletion_cost"] == 1.0
    assert loaded_dict["unigram_costs"]["Ꭰ"]["Ꭰ"] == 0.0
    assert loaded_dict["unigram_costs"]["Ꭰ"]["Ꭱ"] == pytest.approx(
        cost_dict["unigram_costs"]["Ꭰ"]["Ꭱ"]
    )


def test_load_cost_artifact_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        ConfusionCostEngine.load_cost_artifact(missing)


# ============================================================================
# 4. End-to-End Pipeline Integration Test
# ============================================================================


def test_end_to_end_confusion_and_cost_pipeline(tmp_path: Path) -> None:
    vocab = ["a", "b", "c", "d"]
    accumulator = ConfusionAccumulator(vocab=vocab)

    # Simulate alignment of multiple sequences
    corpus = [
        ("abcd", "abcc"),  # sub d -> c
        ("ab", "ac"),  # sub b -> c
        ("abc", "abc"),  # perfect match
        ("abcd", "abd"),  # del c
    ]

    for ref, hyp in corpus:
        aligned = character_levenshtein_align(ref, hyp)
        accumulator.update_from_alignment(aligned)

    cond_probs, labels = accumulator.get_conditional_probabilities(dirichlet_alpha=0.1)
    raw_counts, _ = accumulator.get_raw_counts()

    assert labels == vocab
    assert cond_probs.shape == (4, 4)

    engine = ConfusionCostEngine()
    cost_dict = engine.compute_costs(
        cond_probs=cond_probs,
        labels=labels,
        raw_counts=raw_counts,
        min_support=2,
    )

    artifact_file = tmp_path / "test_costs.json"
    engine.save_cost_artifact(cost_dict, artifact_file)

    reloaded = engine.load_cost_artifact(artifact_file)
    assert reloaded["vocabulary"] == vocab
    for ch in vocab:
        assert reloaded["unigram_costs"][ch][ch] == 0.0


def test_probability_to_normalized_cost():
    from digohwelisgi.evaluation.cost_engine import probability_to_normalized_cost

    # Boundary conditions
    assert pytest.approx(probability_to_normalized_cost(1.0), 0.001) == 0.0
    assert pytest.approx(probability_to_normalized_cost(0.0), 0.001) == 1.0

    # Intermediate values with standard epsilon=1e-5
    cost_half = probability_to_normalized_cost(0.5, epsilon=1e-5)
    assert 0.0 < cost_half < 1.0
    assert pytest.approx(cost_half, 0.01) == 0.06

    # Invalid epsilon raises ValueError
    with pytest.raises(ValueError):
        probability_to_normalized_cost(0.5, epsilon=0.0)
    with pytest.raises(ValueError):
        probability_to_normalized_cost(0.5, epsilon=1.0)
