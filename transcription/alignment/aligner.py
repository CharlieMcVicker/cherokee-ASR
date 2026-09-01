# -*- coding: utf-8 -*-
"""
aligner.py

Chunk-level Sliding Window DTW and Word-level Needleman-Wunsch DP alignment engine.
"""

from typing import Any, Callable, List, Optional, Sequence, Tuple
import numpy as np

from transcription.alignment.distance_metrics import DefaultCERDistanceMetric
from transcription.alignment.models import (
    AlignedChunk,
    AlignmentMetrics,
    AlignmentOutput,
    TextChunk,
    TokenEmission,
    WordInterval,
)


class NeedlemanWunschWordAligner:
    """
    Aligns ground-truth words to matched emission tokens using Needleman-Wunsch DP string edit distance.
    Evaluates 1-to-N and M-to-1 token-to-word DP fusion with configurable penalties.
    Preserves exact ASR emission token start_sec and end_sec boundaries for matched tokens.
    """

    def __init__(
        self,
        distance_metric: Optional[Any] = None,
        chunk_normalizer: Optional[Callable[[str], str]] = None,
        emission_normalizer: Optional[Callable[[str], str]] = None,
        gap_cost: float = 0.8,
        max_fuse_gt: int = 4,
        max_fuse_asr: int = 3,
        fuse_penalty: float = 0.15,
    ):
        self.distance_metric = distance_metric or DefaultCERDistanceMetric()
        self.chunk_norm = chunk_normalizer or (lambda s: s)
        self.emission_norm = emission_normalizer or (lambda s: s)
        self.gap_cost = gap_cost
        self.max_fuse_gt = max_fuse_gt
        self.max_fuse_asr = max_fuse_asr
        self.fuse_penalty = fuse_penalty

    def compute_cost(self, hypothesis: str, reference: str) -> float:
        """
        Computes phonetic edit distance cost between hypothesis and reference strings
        after applying phonetic normalization.
        """
        hyp = self.emission_norm(hypothesis)
        ref = self.chunk_norm(reference)
        return float(self.distance_metric.compute_cost(hypothesis=hyp, reference=ref))

    def align_words(
        self,
        raw_words: Sequence[str],
        matched_tokens: Sequence[TokenEmission],
    ) -> List[WordInterval]:
        """
        Aligns raw words to matched emission tokens.

        Args:
            raw_words: Sequence of ground-truth phonetic words.
            matched_tokens: Sequence of TokenEmission objects matched to this chunk.

        Returns:
            List of WordInterval objects with computed start_sec, end_sec, and word text.
        """
        if not raw_words or not matched_tokens:
            return []

        words_list: List[str] = list(raw_words)
        tokens_list: List[TokenEmission] = list(matched_tokens)

        metric = self.distance_metric
        N = len(words_list)
        M = len(tokens_list)

        # Pre-normalize raw words and tokens
        norm_words = [self.chunk_norm(w) for w in words_list]
        norm_tokens = [self.emission_norm(t.word) for t in tokens_list]

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
        actions: List[Tuple[int, int, int, int, str]] = []

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

        for pi, pj, _i, _j, action_type in actions:
            if action_type == "gap_gt":
                # Unaligned GT word: Interpolate timestamp
                raw_w = words_list[pi]
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
                    )
                )
            elif action_type == "gap_token":
                continue
            elif action_type.startswith("match_fuse_"):
                parts = action_type.split("_")
                k = int(parts[2])
                m_len = int(parts[3]) if len(parts) > 3 else 1
                fused_gt_text = " ".join(words_list[pi : pi + k])

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
                        emitted_word=tok_words,
                    )
                )

        return fused_intervals


class SlidingWindowDTWAligner:
    """
    Sliding Window Dynamic Time Warping (DTW) chunk-level alignment engine.
    Maps token emissions to TextChunks, calculates alignment metrics, and runs word DP alignment.
    """

    def __init__(
        self,
        word_aligner: Optional[NeedlemanWunschWordAligner] = None,
        max_preamble_skip: int = 30,
        max_normal_skip: int = 5,
    ):
        self.word_aligner = word_aligner or NeedlemanWunschWordAligner()
        self.max_preamble_skip = max_preamble_skip
        self.max_normal_skip = max_normal_skip

    def _compute_metrics(
        self,
        aligned_chunks: List[AlignedChunk],
        chunks: Sequence[TextChunk],
    ) -> AlignmentMetrics:
        total_chunks = len(aligned_chunks)
        if total_chunks == 0:
            return AlignmentMetrics(
                total_chunks=0,
                matched_chunks=0,
                match_ratio=0.0,
                mean_distance_score=1.0,
                total_ground_truth_chars=0,
                total_emitted_chars=0,
            )

        chunk_map = {c.chunk_id: c.text for c in chunks}
        matched_chunk_list = [
            c for c in aligned_chunks if c.words and (c.end_sec > c.start_sec)
        ]
        matched_chunks_count = len(matched_chunk_list)
        match_ratio = round(matched_chunks_count / total_chunks, 4)

        scores = [c.distance_score for c in matched_chunk_list]
        mean_score = round(float(np.mean(scores)), 4) if scores else 1.0

        matched_gt = []
        matched_emitted = []
        for c in matched_chunk_list:
            gt_raw = chunk_map.get(c.chunk_id, "")
            gt_norm = self.word_aligner.chunk_norm(gt_raw)
            em_norm = self.word_aligner.emission_norm(c.emitted_text)
            if gt_norm:
                matched_gt.append(gt_norm)
                matched_emitted.append(em_norm)

        concat_gt = " ".join(matched_gt)
        concat_emitted = " ".join(matched_emitted)

        return AlignmentMetrics(
            total_chunks=total_chunks,
            matched_chunks=matched_chunks_count,
            match_ratio=match_ratio,
            mean_distance_score=mean_score,
            total_ground_truth_chars=len(concat_gt),
            total_emitted_chars=len(concat_emitted),
        )

    def align(
        self,
        emissions: Sequence[TokenEmission],
        chunks: Sequence[TextChunk],
        source_id: str = "",
    ) -> AlignmentOutput:
        """
        Aligns text chunks against sequence of token emissions.
        """
        emissions_list = list(emissions)
        aligned_chunks: List[AlignedChunk] = []

        if not emissions_list or not chunks:
            for c in chunks:
                aligned_chunks.append(
                    AlignedChunk(
                        chunk_id=c.chunk_id,
                        start_sec=0.0,
                        end_sec=0.0,
                        words=[],
                        distance_score=1.0,
                        emitted_text="",
                    )
                )
            metrics = self._compute_metrics(aligned_chunks, chunks)
            return AlignmentOutput(
                aligned_chunks=aligned_chunks,
                source_id=source_id,
                raw_tokens=emissions_list,
                metrics=metrics,
            )

        num_tokens = len(emissions_list)
        token_idx = 0

        for c_idx, chunk in enumerate(chunks):
            norm_txt = self.word_aligner.chunk_norm(chunk.text)
            raw_words = [w for w in chunk.text.split() if w]
            num_words = len(raw_words)

            if not norm_txt or num_words == 0:
                aligned_chunks.append(
                    AlignedChunk(
                        chunk_id=chunk.chunk_id,
                        start_sec=0.0,
                        end_sec=0.0,
                        words=[],
                        distance_score=1.0,
                        emitted_text="",
                    )
                )
                continue

            # 2D Sliding Window search
            best_cost = float("inf")
            best_start_idx = token_idx
            best_end_idx = token_idx

            max_skip = min(
                num_tokens - token_idx,
                self.max_preamble_skip if c_idx == 0 else self.max_normal_skip,
            )
            max_search_len = min(num_tokens - token_idx, max(num_words * 3, 10))

            if max_search_len > 0:
                for skip in range(0, max_skip + 1):
                    curr_start = token_idx + skip
                    if curr_start >= num_tokens:
                        break

                    for k in range(1, max_search_len + 1):
                        if curr_start + k > num_tokens:
                            break
                        candidate_tokens = emissions_list[curr_start : curr_start + k]
                        tok_words = [t.word for t in candidate_tokens]

                        cost_spaced = self.word_aligner.compute_cost(
                            hypothesis=" ".join(tok_words),
                            reference=norm_txt,
                        )
                        cost_concat = self.word_aligner.compute_cost(
                            hypothesis="".join(tok_words),
                            reference=norm_txt,
                        )
                        cost = min(cost_spaced, cost_concat)

                        if cost < best_cost:
                            best_cost = cost
                            best_start_idx = curr_start
                            best_end_idx = curr_start + k

            matched_tokens = emissions_list[best_start_idx:best_end_idx]

            if matched_tokens:
                chunk_start = matched_tokens[0].start_sec
                chunk_end = matched_tokens[-1].end_sec
                emitted_text = " ".join([t.word for t in matched_tokens])
                chunk_score = self.word_aligner.compute_cost(
                    hypothesis=emitted_text,
                    reference=norm_txt,
                )

                word_intervals = self.word_aligner.align_words(
                    raw_words=raw_words,
                    matched_tokens=matched_tokens,
                )
                token_idx = best_end_idx
            else:
                prev_end = aligned_chunks[-1].end_sec if aligned_chunks else 0.0
                chunk_start = prev_end
                chunk_end = prev_end
                word_intervals = []
                emitted_text = ""
                chunk_score = 1.0

            aligned_chunks.append(
                AlignedChunk(
                    chunk_id=chunk.chunk_id,
                    start_sec=chunk_start,
                    end_sec=chunk_end,
                    words=word_intervals,
                    distance_score=round(chunk_score, 4),
                    emitted_text=emitted_text,
                )
            )

        metrics = self._compute_metrics(aligned_chunks, chunks)
        return AlignmentOutput(
            aligned_chunks=aligned_chunks,
            source_id=source_id,
            raw_tokens=emissions_list,
            metrics=metrics,
        )
