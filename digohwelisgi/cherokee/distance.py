# -*- coding: utf-8 -*-
"""
digohwelisgi.cherokee.distance

Cherokee phonologically-calibrated and confusion-matrix distance metrics for alignment.
Includes empirical Cherokee vowel deletion penalties and aspiration insertion discounts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from digohwelisgi.core.alignment.distance import (
    ConfusionMatrixCostMetric,
    DistanceMetric,
)
from digohwelisgi.evaluation.cost_engine import probability_to_normalized_cost

# Empirical Cherokee word-internal non-glottal vowel deletion counts & frequencies
# Gathered from 1,864 dictionary sentences in data/training/processed/sentence_audio.csv
# Excluding all word-final vowels and <V>'<V> glottal stop hiatus environments.
CHEROKEE_VOWEL_DROP_COUNTS: Dict[str, Tuple[int, int]] = {
    "i": (580, 4804),  # 12.07% drop rate (primary epenthetic vowel)
    "v": (158, 2252),  # 7.02% drop rate (nasalized vowel)
    "o": (136, 2170),  # 6.27% drop rate
    "a": (297, 5906),  # 5.03% drop rate
    "u": (63, 2751),  # 2.29% drop rate
    "e": (42, 1997),  # 2.10% drop rate
}

CHEROKEE_VOWEL_DROP_PROBABILITIES: Dict[str, float] = {
    v: num / denom for v, (num, denom) in CHEROKEE_VOWEL_DROP_COUNTS.items()
}

DEFAULT_CALIBRATED_DELETION_COSTS: Dict[str, float] = {
    v: probability_to_normalized_cost(prob)
    for v, prob in CHEROKEE_VOWEL_DROP_PROBABILITIES.items()
}

DEFAULT_CALIBRATED_INSERTION_COSTS: Dict[str, float] = {
    "h": 0.10,
}

DEFAULT_CONFUSION_COST_MATRIX_PATH = Path(
    "data/runs/evaluation/confusion_cost_matrix_prebible.json"
)

try:
    from numba import njit  # type: ignore

    @njit(fastmath=True)
    def _numba_wagner_fischer(
        ref_arr: np.ndarray,
        hyp_arr: np.ndarray,
        del_table: np.ndarray,
        ins_table: np.ndarray,
        sub_table: np.ndarray,
        default_ins: float,
    ) -> float:
        m = len(ref_arr)
        n = len(hyp_arr)
        if m == 0:
            total = 0.0
            for j in range(n):
                total += float(ins_table[hyp_arr[j]])
            return float(total)

        dp_prev = np.zeros(n + 1, dtype=np.float32)
        dp_curr = np.zeros(n + 1, dtype=np.float32)

        for j in range(1, n + 1):
            dp_prev[j] = dp_prev[j - 1] + ins_table[hyp_arr[j - 1]]

        for i in range(1, m + 1):
            r = ref_arr[i - 1]
            del_c = del_table[r]
            dp_curr[0] = dp_prev[0] + del_c
            for j in range(1, n + 1):
                h = hyp_arr[j - 1]
                ins_c = ins_table[h]
                sub_c = sub_table[r, h]

                d = dp_prev[j] + del_c
                ins = dp_curr[j - 1] + ins_c
                sub = dp_prev[j - 1] + sub_c
                dp_curr[j] = min(d, min(ins, sub))

            dp_prev[:] = dp_curr[:]

        return float(dp_curr[n] / float(m))

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


class PhonologicalConfusionCostMetric:
    """
    Phonologically-calibrated distance metric wrapping an underlying DistanceMetric.

    Applies empirical character-specific deletion costs (e.g., Cherokee vowel deletion penalties)
    and insertion costs (e.g., aspiration 'h' insertion discounts) during Wagner-Fischer DP.
    Conforms directly to the DistanceMetric protocol.
    """

    def __init__(
        self,
        base_metric: Optional[DistanceMetric] = None,
        custom_deletion_costs: Optional[Dict[str, float]] = None,
        custom_insertion_costs: Optional[Dict[str, float]] = None,
        default_deletion_cost: float = 1.0,
        default_insertion_cost: float = 1.0,
    ):
        if base_metric is not None:
            self.base_metric: DistanceMetric = base_metric
        elif DEFAULT_CONFUSION_COST_MATRIX_PATH.is_file():
            self.base_metric = ConfusionMatrixCostMetric.from_json(
                DEFAULT_CONFUSION_COST_MATRIX_PATH
            )
        else:
            self.base_metric = ConfusionMatrixCostMetric(unigram_costs={})

        if custom_deletion_costs is not None:
            self.custom_deletion_costs = {
                k: float(v) for k, v in custom_deletion_costs.items()
            }
        else:
            self.custom_deletion_costs = dict(DEFAULT_CALIBRATED_DELETION_COSTS)

        if custom_insertion_costs is not None:
            self.custom_insertion_costs = {
                k: float(v) for k, v in custom_insertion_costs.items()
            }
        else:
            self.custom_insertion_costs = dict(DEFAULT_CALIBRATED_INSERTION_COSTS)

        self.default_deletion_cost = float(default_deletion_cost)
        self.default_insertion_cost = float(default_insertion_cost)

        self._init_fast_tables()

    def _init_fast_tables(self) -> None:
        """Precomputes 256-element direct lookup arrays for accelerated DP."""
        self._del_table = np.full(256, self.default_deletion_cost, dtype=np.float32)
        for char, cost in self.custom_deletion_costs.items():
            if len(char) == 1 and ord(char) < 256:
                self._del_table[ord(char)] = float(cost)

        self._ins_table = np.full(256, self.default_insertion_cost, dtype=np.float32)
        for char, cost in self.custom_insertion_costs.items():
            if len(char) == 1 and ord(char) < 256:
                self._ins_table[ord(char)] = float(cost)

        self._sub_table = np.full((256, 256), 1.0, dtype=np.float32)
        for i in range(256):
            self._sub_table[i, i] = 0.0
            c1 = chr(i)
            for j in range(256):
                if i != j:
                    c2 = chr(j)
                    self._sub_table[i, j] = self._get_substitution_cost(c1, c2)

    @classmethod
    def from_json(
        cls,
        path: Union[str, Path],
        custom_deletion_costs: Optional[Dict[str, float]] = None,
        custom_insertion_costs: Optional[Dict[str, float]] = None,
        default_deletion_cost: float = 1.0,
        default_insertion_cost: float = 1.0,
        default_substitution_cost: float = 1.0,
    ) -> PhonologicalConfusionCostMetric:
        """
        Instantiates metric by loading base ConfusionMatrixCostMetric from JSON artifact.
        """
        base = ConfusionMatrixCostMetric.from_json(
            path=path,
            default_substitution_cost=default_substitution_cost,
        )
        return cls(
            base_metric=base,
            custom_deletion_costs=custom_deletion_costs,
            custom_insertion_costs=custom_insertion_costs,
            default_deletion_cost=default_deletion_cost,
            default_insertion_cost=default_insertion_cost,
        )

    @classmethod
    def from_base_metric(
        cls,
        base_metric: DistanceMetric,
        custom_deletion_costs: Optional[Dict[str, float]] = None,
        custom_insertion_costs: Optional[Dict[str, float]] = None,
        default_deletion_cost: float = 1.0,
        default_insertion_cost: float = 1.0,
    ) -> PhonologicalConfusionCostMetric:
        """
        Instantiates wrapper around an existing DistanceMetric.
        """
        return cls(
            base_metric=base_metric,
            custom_deletion_costs=custom_deletion_costs,
            custom_insertion_costs=custom_insertion_costs,
            default_deletion_cost=default_deletion_cost,
            default_insertion_cost=default_insertion_cost,
        )

    def _get_substitution_cost(self, ref_char: str, hyp_char: str) -> float:
        if ref_char == hyp_char:
            return 0.0
        get_sub_fn = getattr(self.base_metric, "get_sub_cost", None)
        if callable(get_sub_fn):
            val: Any = get_sub_fn(ref_char, hyp_char)
            return float(val)
        if self.base_metric is not None:
            return float(
                self.base_metric.compute_cost(hypothesis=hyp_char, reference=ref_char)
            )
        return 1.0

    def compute_cost(self, hypothesis: str, reference: str) -> float:
        """
        Computes normalized edit distance cost between hypothesis and reference.

        Executes accelerated Wagner-Fischer DP with:
        - Reference deletion cost from custom_deletion_costs (fallback default_deletion_cost)
        - Hypothesis insertion cost from custom_insertion_costs (fallback default_insertion_cost)
        - Substitution cost delegated to base_metric

        Normalized by len(reference) (or sum of insertion costs if reference is empty).
        """
        if not reference and not hypothesis:
            return 0.0

        m = len(reference)
        n = len(hypothesis)

        if m == 0:
            return float(
                sum(
                    self.custom_insertion_costs.get(c, self.default_insertion_cost)
                    for c in hypothesis
                )
            )

        if HAS_NUMBA:
            try:
                ref_bytes = np.frombuffer(reference.encode("latin-1"), dtype=np.uint8)
                hyp_bytes = np.frombuffer(hypothesis.encode("latin-1"), dtype=np.uint8)
                return float(
                    _numba_wagner_fischer(
                        ref_bytes,
                        hyp_bytes,
                        self._del_table,
                        self._ins_table,
                        self._sub_table,
                        self.default_insertion_cost,
                    )
                )
            except (UnicodeEncodeError, Exception):
                pass

        # Python fallback for non-Latin-1 characters
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            ref_c = reference[i - 1]
            del_c = self.custom_deletion_costs.get(ref_c, self.default_deletion_cost)
            dp[i][0] = dp[i - 1][0] + del_c

        for j in range(1, n + 1):
            hyp_c = hypothesis[j - 1]
            ins_c = self.custom_insertion_costs.get(hyp_c, self.default_insertion_cost)
            dp[0][j] = dp[0][j - 1] + ins_c

        for i in range(1, m + 1):
            ref_c = reference[i - 1]
            del_c = self.custom_deletion_costs.get(ref_c, self.default_deletion_cost)
            for j in range(1, n + 1):
                hyp_c = hypothesis[j - 1]
                ins_c = self.custom_insertion_costs.get(
                    hyp_c, self.default_insertion_cost
                )
                sub_c = self._get_substitution_cost(ref_c, hyp_c)

                del_cost = dp[i - 1][j] + del_c
                ins_cost = dp[i][j - 1] + ins_c
                sub_cost = dp[i - 1][j - 1] + sub_c
                dp[i][j] = min(del_cost, ins_cost, sub_cost)

        return float(dp[m][n] / m)


__all__ = [
    "CHEROKEE_VOWEL_DROP_COUNTS",
    "CHEROKEE_VOWEL_DROP_PROBABILITIES",
    "DEFAULT_CALIBRATED_DELETION_COSTS",
    "DEFAULT_CALIBRATED_INSERTION_COSTS",
    "DEFAULT_CONFUSION_COST_MATRIX_PATH",
    "PhonologicalConfusionCostMetric",
    "ConfusionMatrixCostMetric",
]
