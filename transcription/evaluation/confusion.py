# -*- coding: utf-8 -*-
"""
transcription.evaluation.confusion

Wagner-Fischer character-level sequence alignment and unigram ConfusionAccumulator
with soft-probability top-K candidate tracking and Dirichlet smoothing.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, Sequence, Tuple

import numpy as np


def character_levenshtein_align(
    reference: str, hypothesis: str
) -> list[tuple[Optional[str], Optional[str], str]]:
    """
    Perform Wagner-Fischer sequence alignment with backtrace at the character level.

    Args:
        reference: Ground truth reference string.
        hypothesis: ASR hypothesis string.

    Returns:
        List of tuples (ref_char, hyp_char, op) where:
            - op is one of "match", "substitution", "deletion", "insertion"
            - deletion: (ref_char, None, "deletion")
            - insertion: (None, hyp_char, "insertion")
            - substitution: (ref_char, hyp_char, "substitution")
            - match: (ref_char, hyp_char, "match")
    """
    m = len(reference)
    n = len(hypothesis)

    if m == 0 and n == 0:
        return []

    # DP table for edit distance
    dp = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(1, m + 1):
        dp[i, 0] = i
    for j in range(1, n + 1):
        dp[0, j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,  # deletion
                dp[i, j - 1] + 1,  # insertion
                dp[i - 1, j - 1] + cost,  # substitution / match
            )

    # Backtrace from (m, n) to (0, 0)
    i, j = m, n
    aligned: list[tuple[Optional[str], Optional[str], str]] = []

    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and reference[i - 1] == hypothesis[j - 1]
            and dp[i, j] == dp[i - 1, j - 1]
        ):
            aligned.append((reference[i - 1], hypothesis[j - 1], "match"))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + 1:
            aligned.append((reference[i - 1], hypothesis[j - 1], "substitution"))
            i -= 1
            j -= 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            aligned.append((reference[i - 1], None, "deletion"))
            i -= 1
        elif j > 0 and dp[i, j] == dp[i, j - 1] + 1:
            aligned.append((None, hypothesis[j - 1], "insertion"))
            j -= 1
        else:
            # Fallback path if tied
            if i > 0 and j > 0:
                aligned.append((reference[i - 1], hypothesis[j - 1], "substitution"))
                i -= 1
                j -= 1
            elif i > 0:
                aligned.append((reference[i - 1], None, "deletion"))
                i -= 1
            else:
                aligned.append((None, hypothesis[j - 1], "insertion"))
                j -= 1

    aligned.reverse()
    return aligned


class ConfusionAccumulator:
    """
    Accumulator for unigram soft-probability confusion counts across alignments.

    Tracks transitions C[ref, hyp] from character alignments, including deletions
    (hyp="<del>") and insertions (ref="<ins>"), supporting top-K alternative
    hypotheses with posterior probabilities.
    """

    DEL_TOKEN: str = "<del>"
    INS_TOKEN: str = "<ins>"

    def __init__(self, vocab: Optional[list[str]] = None) -> None:
        """
        Initialize the accumulator.

        Args:
            vocab: Optional list of character vocabulary tokens. If None, vocabulary
                   is dynamically collected from observed characters.
        """
        self.vocab: list[str] = list(vocab) if vocab is not None else []
        self._counts: dict[tuple[str, str], float] = defaultdict(float)
        self._seen_tokens: set[str] = set(self.vocab)

    @property
    def counts(self) -> dict[tuple[str, str], float]:
        """Return shallow copy of raw count dictionary."""
        return dict(self._counts)

    def add_count(self, ref: str, hyp: str, count: float = 1.0) -> None:
        """
        Directly increment count for transition (ref, hyp).

        Args:
            ref: Reference character (or special token).
            hyp: Hypothesis character (or special token).
            count: Float count increment.
        """
        self._counts[(ref, hyp)] += float(count)
        if ref not in (self.DEL_TOKEN, self.INS_TOKEN):
            self._seen_tokens.add(ref)
        if hyp not in (self.DEL_TOKEN, self.INS_TOKEN):
            self._seen_tokens.add(hyp)

    def update_from_alignment(
        self,
        aligned_pairs: Sequence[tuple[Optional[str], Optional[str], str]],
        top_k_per_hyp_char: Optional[list[list[tuple[str, float]]]] = None,
    ) -> None:
        """
        Update confusion counts from an alignment sequence and optional top-k candidate probabilities.

        Args:
            aligned_pairs: Output of character_levenshtein_align, a list or sequence of (ref_char, hyp_char, op).
            top_k_per_hyp_char: Optional list of top-K (token, prob) tuples for each hypothesis character
                                consumed in the alignment (i.e. where hyp_char is not None).
        """
        hyp_idx = 0
        allowed = set(self.vocab) if self.vocab else None

        for ref_char, hyp_char, op in aligned_pairs:
            if ref_char is not None and (op == "deletion" or hyp_char is None):
                ref = ref_char
                if allowed is None or ref in allowed:
                    self._counts[(ref, self.DEL_TOKEN)] += 1.0
                    self._seen_tokens.add(ref)
            elif ref_char is not None and hyp_char is not None:
                ref = ref_char
                if allowed is None or ref in allowed:
                    self._seen_tokens.add(ref)
                    if (
                        top_k_per_hyp_char is not None
                        and hyp_idx < len(top_k_per_hyp_char)
                        and top_k_per_hyp_char[hyp_idx]
                    ):
                        top_k = top_k_per_hyp_char[hyp_idx]
                        for alt_k, prob_k in top_k:
                            if not alt_k:
                                continue
                            if allowed is not None and alt_k not in allowed:
                                continue
                            self._counts[(ref, alt_k)] += float(prob_k)
                            self._seen_tokens.add(alt_k)
                    else:
                        if hyp_char and (allowed is None or hyp_char in allowed):
                            self._counts[(ref, hyp_char)] += 1.0
                            self._seen_tokens.add(hyp_char)
                hyp_idx += 1
            elif hyp_char is not None and (op == "insertion" or ref_char is None):
                hyp = hyp_char
                if hyp and (allowed is None or hyp in allowed):
                    self._counts[(self.INS_TOKEN, hyp)] += 1.0
                    self._seen_tokens.add(hyp)
                hyp_idx += 1

    def _resolve_labels(self, labels: Optional[list[str]] = None) -> list[str]:
        """Resolve active label list in deterministic order."""
        if labels is not None:
            return list(labels)
        if self.vocab:
            return list(self.vocab)
        tokens = sorted(
            [
                t
                for t in self._seen_tokens
                if t not in (self.DEL_TOKEN, self.INS_TOKEN) and t
            ]
        )
        return tokens

    def get_raw_counts(
        self, labels: Optional[list[str]] = None
    ) -> tuple[np.ndarray, list[str]]:
        """
        Get the unnormalized unigram count matrix.

        Args:
            labels: Optional explicit label ordering. If None, uses self.vocab or sorted seen tokens.

        Returns:
            (counts_matrix, labels) where counts_matrix[i, j] is C[labels[i], labels[j]].
        """
        active_labels = self._resolve_labels(labels)
        v = len(active_labels)
        counts = np.zeros((v, v), dtype=np.float64)
        for i, ref in enumerate(active_labels):
            for j, hyp in enumerate(active_labels):
                counts[i, j] = self._counts.get((ref, hyp), 0.0)
        return counts, active_labels

    def get_conditional_probabilities(
        self,
        dirichlet_alpha: float = 0.1,
        labels: Optional[list[str]] = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Compute row-normalized conditional confusion probabilities P(hyp=j | ref=i)
        using Dirichlet prior smoothing.

        Formula:
            P[i, j] = (C[i, j] + alpha) / (sum_k C[i, k] + alpha * |V|)

        Args:
            dirichlet_alpha: Smoothing hyperparameter alpha >= 0.
            labels: Optional explicit label ordering. Defaults to self.vocab or seen tokens.

        Returns:
            (P, labels) where P is a (V, V) matrix such that rows sum to 1.0.
        """
        active_labels = self._resolve_labels(labels)
        v = len(active_labels)
        if v == 0:
            return np.empty((0, 0), dtype=np.float64), []

        if dirichlet_alpha < 0:
            raise ValueError(
                f"dirichlet_alpha must be non-negative, got {dirichlet_alpha}"
            )

        counts = np.zeros((v, v), dtype=np.float64)
        for i, ref in enumerate(active_labels):
            for j, hyp in enumerate(active_labels):
                counts[i, j] = self._counts.get((ref, hyp), 0.0)

        numerator = counts + dirichlet_alpha
        denominator = np.sum(counts, axis=1, keepdims=True) + (dirichlet_alpha * v)

        zero_rows = denominator.squeeze(-1) == 0
        if np.any(zero_rows):
            prob = np.zeros_like(numerator)
            nonzero_mask = ~zero_rows
            if np.any(nonzero_mask):
                prob[nonzero_mask] = numerator[nonzero_mask] / denominator[nonzero_mask]
            prob[zero_rows] = 1.0 / v
        else:
            prob = numerator / denominator

        return prob, active_labels

    def get_deletion_counts(self) -> dict[str, float]:
        """Return dictionary mapping reference characters to deletion counts."""
        return {
            ref: count
            for (ref, hyp), count in self._counts.items()
            if hyp == self.DEL_TOKEN
        }

    def get_insertion_counts(self) -> dict[str, float]:
        """Return dictionary mapping hypothesis characters to insertion counts."""
        return {
            hyp: count
            for (ref, hyp), count in self._counts.items()
            if ref == self.INS_TOKEN
        }
