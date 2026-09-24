# -*- coding: utf-8 -*-
"""
dp.py

Chunk-level Sliding Window DTW and Word-level Needleman-Wunsch DP alignment engine.
Consumes ModelOutput universal currency or sequences of TokenEmission / decoded tokens,
configured cleanly by pluggable DistanceMetric strategy implementations.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import numpy as np

from digohwelisgi.core.alignment.distance import (
    DefaultCERDistanceMetric,
    DistanceMetric,
)
from digohwelisgi.core.alignment.models import (
    AlignedChunk,
    AlignmentMetrics,
    AlignmentOutput,
    TextChunk,
    TokenEmission,
    WordInterval,
)
from digohwelisgi.core.models.output import ModelOutput


class NeedlemanWunschWordAligner:
    """
    Aligns ground-truth words to matched emission tokens using Needleman-Wunsch DP string edit distance.
    Evaluates 1-to-N and M-to-1 token-to-word DP fusion with configurable penalties.
    Preserves exact ASR emission token start_sec and end_sec boundaries for matched tokens.
    """

    def __init__(
        self,
        distance_metric: Optional[DistanceMetric] = None,
        chunk_normalizer: Optional[Callable[[str], str]] = None,
        emission_normalizer: Optional[Callable[[str], str]] = None,
        gap_cost: float = 0.8,
        max_fuse_gt: int = 4,
        max_fuse_asr: int = 3,
        fuse_penalty: float = 0.15,
    ):
        self.distance_metric: DistanceMetric = (
            distance_metric
            if distance_metric is not None
            else DefaultCERDistanceMetric()
        )
        self.chunk_norm: Callable[[str], str] = chunk_normalizer or (lambda s: s)
        self.emission_norm: Callable[[str], str] = emission_normalizer or (lambda s: s)
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
                if i < N:
                    new_cost = current_cost + self.gap_cost
                    if new_cost < dp[i + 1, j]:
                        dp[i + 1, j] = new_cost
                        parent[i + 1][j] = (i, j, "gap_gt")

                # Option 2: Spurious ASR token (Insertion gap)
                if j < M:
                    new_cost = current_cost + self.gap_cost
                    if new_cost < dp[i, j + 1]:
                        dp[i, j + 1] = new_cost
                        parent[i][j + 1] = (i, j, "gap_asr")

                # Option 3: Fusion matches (n words to m tokens)
                for n_len in range(1, self.max_fuse_gt + 1):
                    if i + n_len > N:
                        break
                    for m_len in range(1, self.max_fuse_asr + 1):
                        if j + m_len > M:
                            break

                        gt_slice = "".join(norm_words[i : i + n_len])
                        asr_slice = "".join(norm_tokens[j : j + m_len])

                        match_cost = float(
                            self.distance_metric.compute_cost(
                                hypothesis=asr_slice, reference=gt_slice
                            )
                        )

                        # Penalize fusion to prefer 1:1 alignments when costs are comparable
                        extra_penalty = 0.0
                        if n_len > 1:
                            extra_penalty += self.fuse_penalty * (n_len - 1)
                        if m_len > 1:
                            extra_penalty += self.fuse_penalty * (m_len - 1)

                        total_step_cost = match_cost + extra_penalty
                        if current_cost + total_step_cost < dp[i + n_len, j + m_len]:
                            dp[i + n_len, j + m_len] = current_cost + total_step_cost
                            parent[i + n_len][j + m_len] = (
                                i,
                                j,
                                f"match_{n_len}_{m_len}",
                            )

        # Backtrack optimal DP path
        curr_i, curr_j = N, M
        path: List[Tuple[int, int, str]] = []

        while curr_i > 0 or curr_j > 0:
            p = parent[curr_i][curr_j]
            if p is None:
                break
            prev_i, prev_j, op = p
            path.append((prev_i, prev_j, op))
            curr_i, curr_j = prev_i, prev_j

        path.reverse()

        # Build WordInterval results
        fused_intervals: List[WordInterval] = []

        for pi, pj, op in path:
            if op == "gap_gt":
                # Ground truth word was completely skipped
                prev_end = (
                    fused_intervals[-1].end_sec
                    if fused_intervals
                    else tokens_list[0].start_sec
                )
                fused_intervals.append(
                    WordInterval(
                        word=words_list[pi],
                        start_sec=prev_end,
                        end_sec=prev_end,
                        confidence=0.0,
                        flagged=True,
                        emitted_word=None,
                    )
                )
            elif op == "gap_asr":
                # Spurious ASR token skipped; ignore
                continue
            elif op.startswith("match_"):
                parts = op.split("_")
                n_len = int(parts[1])
                m_len = int(parts[2])

                fused_gt_text = " ".join(words_list[pi : pi + n_len])
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
    Maps token emissions or ModelOutput universal currency to TextChunks,
    calculates alignment metrics, and runs word DP alignment.
    """

    def __init__(
        self,
        word_aligner: Optional[NeedlemanWunschWordAligner] = None,
        distance_metric: Optional[DistanceMetric] = None,
        max_preamble_skip: int = 30,
        max_normal_skip: int = 5,
    ):
        if word_aligner is not None:
            self.word_aligner = word_aligner
            if distance_metric is not None:
                self.word_aligner.distance_metric = distance_metric
        else:
            self.word_aligner = NeedlemanWunschWordAligner(
                distance_metric=distance_metric
            )
        self.max_preamble_skip = max_preamble_skip
        self.max_normal_skip = max_normal_skip

    @property
    def distance_metric(self) -> DistanceMetric:
        return self.word_aligner.distance_metric

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

    def _convert_model_output(self, model_output: ModelOutput) -> List[TokenEmission]:
        """
        Converts ModelOutput emissions into a sequence of TokenEmission objects.
        Uses model_output.decode_tokens() and acoustic frame durations to synthesize token emissions.
        """
        raw_tokens = model_output.decode_tokens(collapse_repeats=True, remove_pad=True)
        frame_dur = model_output.frame_duration_sec

        # Construct words from decoded tokens split on spaces / word delimiters
        words: List[str] = []
        current_word_chars: List[str] = []
        for tok in raw_tokens:
            if tok == " ":
                if current_word_chars:
                    words.append("".join(current_word_chars))
                    current_word_chars = []
            else:
                current_word_chars.append(tok)
        if current_word_chars:
            words.append("".join(current_word_chars))

        # Time estimation: distribute across total acoustic duration
        total_frames = (
            model_output.lpz.shape[0]
            if model_output.lpz.ndim == 2
            else model_output.lpz.shape[1]
        )
        total_duration = total_frames * frame_dur
        num_words = len(words)

        if num_words == 0:
            return []

        word_dur = total_duration / num_words if num_words > 0 else total_duration
        emissions: List[TokenEmission] = []
        for idx, w in enumerate(words):
            start = idx * word_dur
            end = min(total_duration, (idx + 1) * word_dur)
            emissions.append(
                TokenEmission(
                    word=w,
                    start_sec=round(start, 3),
                    end_sec=round(end, 3),
                    confidence=1.0,
                )
            )
        return emissions

    def align(
        self,
        emissions: Union[ModelOutput, Sequence[TokenEmission]],
        chunks: Sequence[TextChunk],
        source_id: str = "",
    ) -> AlignmentOutput:
        """
        Aligns text chunks against sequence of token emissions or ModelOutput universal currency.

        Args:
            emissions: Either a ModelOutput instance or a sequence of TokenEmission objects.
            chunks: Sequence of TextChunk reference units to align.
            source_id: Optional identifier for the audio/source document.

        Returns:
            AlignmentOutput containing aligned chunks and overall metrics.
        """
        if isinstance(emissions, ModelOutput):
            emissions_list = self._convert_model_output(emissions)
        else:
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


class WagnerFischerAligner:
    """
    Pure dynamic programming string and phoneme alignment service using Wagner-Fischer algorithm.
    Supports 1-to-1, 1-to-2, 2-to-1, and 2-to-2 multi-gram transitions.
    """

    def __init__(self, cost_metric: Optional[DistanceMetric] = None):
        self.cost_metric = cost_metric or DefaultCERDistanceMetric()

    def align(
        self,
        source_tokens: Sequence[Union[str, Any]],
        target_tokens: Sequence[Union[str, Any]],
        matrix: Optional[Any] = None,
        token_confidences: Optional[Sequence[float]] = None,
    ) -> Any:
        """Aligns source and target token sequences via Wagner-Fischer DP."""
        if matrix is not None and hasattr(matrix, "get_substitution_cost"):
            from digohwelisgi.core.codeswitching.trainer import (
                align_word_pair_generic,
            )

            return align_word_pair_generic(
                source_tokens=source_tokens,
                target_tokens=target_tokens,
                matrix=matrix,
                token_confidences=token_confidences,
            )

        # Fallback to basic string edit alignment if no confusion matrix
        src_strs = [str(s) for s in source_tokens]
        tgt_strs = [str(t) for t in target_tokens]
        N, M = len(src_strs), len(tgt_strs)
        dp = np.zeros((N + 1, M + 1), dtype=np.float32)
        for i in range(1, N + 1):
            dp[i, 0] = dp[i - 1, 0] + 1.0
        for j in range(1, M + 1):
            dp[0, j] = dp[0, j - 1] + 1.0
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                cost_sub = dp[i - 1, j - 1] + float(
                    self.cost_metric.compute_cost(src_strs[i - 1], tgt_strs[j - 1])
                )
                cost_del = dp[i - 1, j] + 1.0
                cost_ins = dp[i, j - 1] + 1.0
                dp[i, j] = min(cost_sub, cost_del, cost_ins)
        return float(dp[N, M])


__all__ = [
    "NeedlemanWunschWordAligner",
    "SlidingWindowDTWAligner",
    "WagnerFischerAligner",
]
