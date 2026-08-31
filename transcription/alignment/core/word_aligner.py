"""
Word-level alignment engine using Needleman-Wunsch DP string edit distance with M-to-N token fusion.
"""

from typing import List, Optional, Sequence, Tuple
import numpy as np

from transcription.alignment.domain.models import TokenEmission, WordInterval
from transcription.alignment.ports.protocols import DistanceMetric, PhoneticPreprocessor
from transcription.alignment.strategies.distance_metrics import DefaultCERDistanceMetric
from transcription.alignment.strategies.preprocessors import (
    CherokeePhoneticPreprocessor,
)


class NeedlemanWunschWordAligner:
    """
    Aligns ground-truth words to matched emission tokens using Needleman-Wunsch DP string edit distance.
    Evaluates 1-to-N and M-to-1 token-to-word DP fusion with configurable penalties.
    Preserves exact ASR emission token start_sec and end_sec boundaries for matched tokens.
    """

    def __init__(
        self,
        distance_metric: Optional[DistanceMetric] = None,
        preprocessor: Optional[PhoneticPreprocessor] = None,
        gap_cost: float = 0.8,
        max_fuse_gt: int = 4,
        max_fuse_asr: int = 3,
        fuse_penalty: float = 0.15,
    ):
        self.distance_metric = distance_metric or DefaultCERDistanceMetric()
        self.preprocessor = preprocessor or CherokeePhoneticPreprocessor()
        self.gap_cost = gap_cost
        self.max_fuse_gt = max_fuse_gt
        self.max_fuse_asr = max_fuse_asr
        self.fuse_penalty = fuse_penalty

    def normalize(self, text: str) -> str:
        """Normalizes text using the injected phonetic preprocessor."""
        return self.preprocessor.normalize(text)

    def compute_cost(
        self, hypothesis: str, reference: str, pre_normalized: bool = False
    ) -> float:
        """
        Computes phonetic edit distance cost between hypothesis and reference strings.
        If pre_normalized is False, both strings are normalized via self.preprocessor first.
        """
        hyp = hypothesis if pre_normalized else self.preprocessor.normalize(hypothesis)
        ref = reference if pre_normalized else self.preprocessor.normalize(reference)
        return self.distance_metric.compute_cost(hypothesis=hyp, reference=ref)

    def align_words(
        self,
        raw_words: Sequence[str],
        matched_tokens: Sequence[TokenEmission],
        raw_syllabary_words: Optional[Sequence[str]] = None,
    ) -> List[WordInterval]:
        """
        Aligns raw words to matched emission tokens.

        Args:
            raw_words: List of ground-truth phonetic words.
            matched_tokens: List of TokenEmission objects matched to this chunk.
            raw_syllabary_words: Optional list of corresponding Cherokee syllabary words.

        Returns:
            List of WordInterval objects with computed start_sec, end_sec, and word text.
        """
        if not raw_words or not matched_tokens:
            return []

        words_list: List[str] = list(raw_words)
        tokens_list: List[TokenEmission] = list(matched_tokens)
        syll_list: Optional[List[str]] = (
            list(raw_syllabary_words) if raw_syllabary_words is not None else None
        )

        metric = self.distance_metric
        prep = self.preprocessor

        N = len(words_list)
        M = len(tokens_list)

        # Pre-normalize raw words and tokens
        norm_words = [prep.normalize(w) for w in words_list]
        norm_tokens = [prep.normalize(t.word) for t in tokens_list]

        # DP state: dp[i, j] = cost
        dp = np.full((N + 1, M + 1), fill_value=1e9, dtype=np.float32)
        parent: List[List[Optional[Tuple[int, int, str]]]] = [
            [None for _ in range(M + 1)] for _ in range(N + 1)
        ]

        dp[0, 0] = 0.0

        for i in range(N + 1):
            for j in range(M + 1):
                if dp[i, j] >= 1e8:
                    continue

                current_cost = dp[i, j]

                # Option 1: Unaligned GT word (Deletion gap)
                if i + 1 <= N:
                    cost = current_cost + self.gap_cost
                    if cost < dp[i + 1, j]:
                        dp[i + 1, j] = cost
                        parent[i + 1][j] = (i, j, "gap_gt")

                # Option 2: Unaligned ASR token (Insertion gap)
                if j + 1 <= M:
                    cost = current_cost + self.gap_cost
                    if cost < dp[i, j + 1]:
                        dp[i, j + 1] = cost
                        parent[i][j + 1] = (i, j, "gap_token")

                # Option 3: Match M ASR tokens against K GT words
                for m_len in range(1, self.max_fuse_asr + 1):
                    if j + m_len <= M:
                        asr_concat = "".join(norm_tokens[j : j + m_len])
                        for k in range(1, self.max_fuse_gt + 1):
                            if i + k <= N:
                                gt_concat = " ".join(norm_words[i : i + k])
                                edit_cost = metric.compute_cost(
                                    hypothesis=asr_concat, reference=gt_concat
                                )
                                penalty = self.fuse_penalty * (
                                    k - 1
                                ) + self.fuse_penalty * (m_len - 1)
                                cost = current_cost + edit_cost + penalty

                                if cost < dp[i + k, j + m_len]:
                                    dp[i + k][j + m_len] = cost
                                    parent[i + k][j + m_len] = (
                                        i,
                                        j,
                                        f"match_fuse_{k}_{m_len}",
                                    )

        # Backtrack optimal DP path
        i, j = N, M
        actions = []

        while i > 0 or j > 0:
            p = parent[i][j]
            if p is None:
                break
            pi, pj, action_type = p
            actions.append((pi, pj, i, j, action_type))
            i, j = pi, pj

        actions.reverse()

        # Convert DP actions into WordInterval list
        fused_intervals: List[WordInterval] = []

        for pi, pj, i, j, action_type in actions:
            if action_type == "gap_gt":
                # Unaligned GT word: Interpolate timestamp
                raw_w = words_list[pi]
                syll_w = syll_list[pi] if (syll_list and pi < len(syll_list)) else None
                prev_end = (
                    fused_intervals[-1].end_sec
                    if fused_intervals
                    else tokens_list[0].start_sec
                )
                fused_intervals.append(
                    WordInterval(
                        word=raw_w,
                        start_sec=prev_end,
                        end_sec=prev_end,
                        confidence=0.0,
                        flagged=True,
                        syllabary=syll_w,
                    )
                )
            elif action_type == "gap_token":
                continue
            elif action_type.startswith("match_fuse_"):
                parts = action_type.split("_")
                k = int(parts[2])
                m_len = int(parts[3]) if len(parts) > 3 else 1
                fused_gt_text = " ".join(words_list[pi : pi + k])
                syll_w = (
                    " ".join(syll_list[pi : pi + k])
                    if (syll_list and pi + k <= len(syll_list))
                    else None
                )

                first_tok = tokens_list[pj]
                last_tok = tokens_list[pj + m_len - 1]

                confidences = [t.confidence for t in tokens_list[pj : pj + m_len]]
                avg_conf = float(np.mean(confidences)) if confidences else 1.0

                tok_words = " ".join([t.word for t in tokens_list[pj : pj + m_len]])

                fused_intervals.append(
                    WordInterval(
                        word=fused_gt_text,
                        start_sec=first_tok.start_sec,
                        end_sec=last_tok.end_sec,
                        confidence=avg_conf,
                        flagged=avg_conf < 0.5,
                        syllabary=syll_w,
                        emitted_word=tok_words,
                    )
                )

        return fused_intervals
