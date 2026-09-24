# -*- coding: utf-8 -*-
"""
distance.py

Distance metric strategy interface and concrete implementations for alignment evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Optional, Protocol, Tuple, Union, runtime_checkable
from jiwer import cer


@runtime_checkable
class DistanceMetric(Protocol):
    """Protocol for distance cost evaluation between hypothesis and reference strings."""

    def compute_cost(self, hypothesis: str, reference: str) -> float:
        """
        Computes distance / error cost between hypothesis and reference strings.

        Args:
            hypothesis: The predicted or emitted string.
            reference: The ground-truth reference string.

        Returns:
            Non-negative float representing the distance score.
        """
        ...


def calculate_cer(hypothesis: str, reference: str) -> float:
    """Computes Character Error Rate (CER) between hypothesis and reference strings."""
    if not reference and not hypothesis:
        return 0.0
    if not reference or not hypothesis:
        return 1.0
    try:
        return float(cer(reference, hypothesis))
    except Exception:
        return 1.0


class DefaultCERDistanceMetric:
    """Default DistanceMetric using jiwer CER (Character Error Rate)."""

    def compute_cost(self, hypothesis: str, reference: str) -> float:
        """
        Computes Character Error Rate between reference and hypothesis strings.
        Returns 0.0 if both are empty, or 1.0 if one is empty and the other is not.
        """
        return calculate_cer(hypothesis, reference)


# CharacterErrorRateMetric is an alias for DefaultCERDistanceMetric
CharacterErrorRateMetric = DefaultCERDistanceMetric


class PhonologicalDistanceMetric:
    """
    Configurable weighted edit distance metric.

    Allows specifying custom substitution costs for phonetic/phonological pairs.
    """

    def __init__(
        self,
        substitution_weights: Optional[Dict[Tuple[str, str], float]] = None,
        insertion_cost: float = 1.0,
        deletion_cost: float = 1.0,
        default_substitution_cost: float = 1.0,
    ):
        self.substitution_weights = substitution_weights or {}
        self.insertion_cost = insertion_cost
        self.deletion_cost = deletion_cost
        self.default_substitution_cost = default_substitution_cost

    def _get_sub_cost(self, c1: str, c2: str) -> float:
        if c1 == c2:
            return 0.0
        if (c1, c2) in self.substitution_weights:
            return self.substitution_weights[(c1, c2)]
        if (c2, c1) in self.substitution_weights:
            return self.substitution_weights[(c2, c1)]
        return self.default_substitution_cost

    def compute_cost(self, hypothesis: str, reference: str) -> float:
        """
        Computes normalized weighted edit distance cost between hypothesis and reference.
        Cost is normalized by reference length (or max length if reference is empty).
        """
        if not reference and not hypothesis:
            return 0.0
        if not reference:
            return float(len(hypothesis) * self.insertion_cost)

        m = len(reference)
        n = len(hypothesis)

        # Dynamic programming matrix for weighted edit distance
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] + self.deletion_cost
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] + self.insertion_cost

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                del_cost = dp[i - 1][j] + self.deletion_cost
                ins_cost = dp[i][j - 1] + self.insertion_cost
                sub_cost = dp[i - 1][j - 1] + self._get_sub_cost(
                    reference[i - 1], hypothesis[j - 1]
                )
                dp[i][j] = min(del_cost, ins_cost, sub_cost)

        raw_distance = dp[m][n]
        return float(raw_distance / m)


# LevenshteinDistanceMetric is an alias for PhonologicalDistanceMetric
LevenshteinDistanceMetric = PhonologicalDistanceMetric


class CustomCallableDistanceMetric:
    """DistanceMetric wrapping a custom callable (hyp, ref) -> float."""

    def __init__(self, fn: Callable[[str, str], float]):
        self.fn = fn

    def compute_cost(self, hypothesis: str, reference: str) -> float:
        return float(self.fn(hypothesis, reference))


class ConfusionMatrixCostMetric:
    """
    Empirically learned asymmetric unigram edit distance metric.

    Computes normalized edit distance using character-level substitution costs
    derived from ASR confusion matrices. Directly conforms to DistanceMetric protocol.
    """

    def __init__(
        self,
        unigram_costs: Dict[str, Dict[str, float]],
        insertion_cost: float = 1.0,
        deletion_cost: float = 1.0,
        default_substitution_cost: float = 1.0,
    ):
        self.unigram_costs = unigram_costs or {}
        self.insertion_cost = float(insertion_cost)
        self.deletion_cost = float(deletion_cost)
        self.default_substitution_cost = float(default_substitution_cost)

    @classmethod
    def from_json(
        cls,
        path: Union[str, Path],
        insertion_cost: float = 1.0,
        deletion_cost: float = 1.0,
        default_substitution_cost: float = 1.0,
    ) -> ConfusionMatrixCostMetric:
        """Loads JSON cost artifact generated by cost engine."""
        path_obj = Path(path)
        with path_obj.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "unigram_costs" in data:
            unigram_costs = data["unigram_costs"]
            ins_cost = float(data.get("insertion_cost", insertion_cost))
            del_cost = float(data.get("deletion_cost", deletion_cost))
            def_sub_cost = float(
                data.get("default_substitution_cost", default_substitution_cost)
            )
        elif isinstance(data, dict):
            unigram_costs = data
            ins_cost = insertion_cost
            del_cost = deletion_cost
            def_sub_cost = default_substitution_cost
        else:
            raise ValueError(
                f"Invalid JSON format for ConfusionMatrixCostMetric in {path}"
            )

        return cls(
            unigram_costs=unigram_costs,
            insertion_cost=ins_cost,
            deletion_cost=del_cost,
            default_substitution_cost=def_sub_cost,
        )

    def get_sub_cost(self, ref_char: str, hyp_char: str) -> float:
        """Returns asymmetric substitution cost from ref_char to hyp_char."""
        if ref_char == hyp_char:
            return 0.0
        return float(
            self.unigram_costs.get(ref_char, {}).get(
                hyp_char, self.default_substitution_cost
            )
        )

    def compute_cost(self, hypothesis: str, reference: str) -> float:
        """
        Computes normalized edit distance cost between hypothesis and reference.

        Uses Wagner-Fischer DP table of size (m+1) x (n+1) where m=len(reference), n=len(hypothesis).
        Normalized by m if m > 0 else returns n * insertion_cost.
        """
        if not reference and not hypothesis:
            return 0.0

        m = len(reference)
        n = len(hypothesis)

        if m == 0:
            return float(n * self.insertion_cost)

        dp = [[0.0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] + self.deletion_cost
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] + self.insertion_cost

        for i in range(1, m + 1):
            ref_c = reference[i - 1]
            for j in range(1, n + 1):
                hyp_c = hypothesis[j - 1]
                del_cost = dp[i - 1][j] + self.deletion_cost
                ins_cost = dp[i][j - 1] + self.insertion_cost
                sub_cost = dp[i - 1][j - 1] + self.get_sub_cost(ref_c, hyp_c)
                dp[i][j] = min(del_cost, ins_cost, sub_cost)

        return float(dp[m][n] / m)


__all__ = [
    "DistanceMetric",
    "DefaultCERDistanceMetric",
    "CharacterErrorRateMetric",
    "PhonologicalDistanceMetric",
    "LevenshteinDistanceMetric",
    "CustomCallableDistanceMetric",
    "ConfusionMatrixCostMetric",
    "calculate_cer",
]
