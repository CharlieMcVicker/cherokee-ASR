"""
Distance metric strategy implementations.
"""

from typing import Dict, Optional, Tuple
from jiwer import cer


class DefaultCERDistanceMetric:
    """Default DistanceMetric using jiwer CER (Character Error Rate)."""

    def compute_cost(self, hypothesis: str, reference: str) -> float:
        """
        Computes Character Error Rate between reference and hypothesis strings.
        Returns 0.0 if both are empty, or 1.0 if one is empty and the other is not.
        """
        if not reference and not hypothesis:
            return 0.0
        if not reference or not hypothesis:
            return 1.0
        try:
            return float(cer(reference, hypothesis))
        except Exception:
            return 1.0


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
