"""
Sliding Window Dynamic Time Warping (DTW) chunk-level alignment engine.
"""

from typing import List, Optional, Sequence
import numpy as np

from transcription.alignment.core.word_aligner import NeedlemanWunschWordAligner
from transcription.alignment.domain.models import (
    AlignedChunk,
    AlignmentMetrics,
    AlignmentOutput,
    TextChunk,
    TokenEmission,
)
from transcription.alignment.ports.protocols import (
    ChunkAlignmentEngine,
    DistanceMetric,
    PhoneticPreprocessor,
    ReconciliationStrategy,
)
from transcription.alignment.strategies.distance_metrics import DefaultCERDistanceMetric
from transcription.alignment.strategies.preprocessors import (
    CherokeePhoneticPreprocessor,
)


class SlidingWindowDTWAligner:
    """
    Chunk alignment engine implementing ChunkAlignmentEngine protocol.
    Performs sliding-window dynamic search to map token emissions to generic TextChunks,
    calculates alignment metrics across matched chunks, and triggers word-level DP alignment.
    """

    def __init__(
        self,
        word_aligner: NeedlemanWunschWordAligner,
        distance_metric: DistanceMetric,
        preprocessor: PhoneticPreprocessor,
        reconciliation_strategy: Optional[ReconciliationStrategy],
        max_preamble_skip: int,
        max_normal_skip: int,
    ):
        self.word_aligner = word_aligner
        self.distance_metric = distance_metric
        self.preprocessor = preprocessor
        self.reconciliation_strategy = reconciliation_strategy
        self.max_preamble_skip = max_preamble_skip
        self.max_normal_skip = max_normal_skip

    def _compute_metrics(
        self,
        aligned_chunks: List[AlignedChunk],
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
            gt_norm = c.chunk.normalized_text or self.preprocessor.normalize(
                c.chunk.raw_text
            )
            em_norm = self.preprocessor.normalize(c.emitted_text)
            if gt_norm:
                matched_gt.append(gt_norm)
                matched_emitted.append(em_norm)

        concat_gt = " ".join(matched_gt)
        concat_emitted = " ".join(matched_emitted)

        metrics = AlignmentMetrics(
            total_chunks=total_chunks,
            matched_chunks=matched_chunks_count,
            match_ratio=match_ratio,
            mean_distance_score=mean_score,
            total_ground_truth_chars=len(concat_gt),
            total_emitted_chars=len(concat_emitted),
        )
        return metrics

    def align_chunks(
        self,
        emissions: Sequence[TokenEmission],
        chunks: Sequence[TextChunk],
    ) -> AlignmentOutput:
        """
        Aligns text chunks against sequence of token emissions.
        """
        metric = self.distance_metric
        prep = self.preprocessor
        reconciliation_strategy = self.reconciliation_strategy

        aligned_chunks: List[AlignedChunk] = []

        if not emissions or not chunks:
            for c in chunks:
                aligned_chunks.append(
                    AlignedChunk(
                        chunk_id=c.chunk_id,
                        chunk=c,
                        start_sec=0.0,
                        end_sec=0.0,
                        words=[],
                        distance_score=1.0,
                        emitted_text="",
                    )
                )
            metrics = self._compute_metrics(aligned_chunks)
            return AlignmentOutput(
                aligned_chunks=aligned_chunks,
                raw_tokens=list(emissions),
                metrics=metrics,
            )

        num_tokens = len(emissions)

        # Pre-process chunks
        normalized_chunks_info = []
        for c in chunks:
            norm_txt = (
                c.normalized_text if c.normalized_text else prep.normalize(c.raw_text)
            )
            raw_words = [w for w in c.raw_text.split() if w]
            if not raw_words:
                raw_words = norm_txt.split()
            normalized_chunks_info.append(
                {
                    "chunk": c,
                    "norm_text": norm_txt,
                    "raw_words": raw_words,
                }
            )

        token_idx = 0

        for c_idx, c_info in enumerate(normalized_chunks_info):
            chunk_obj = c_info["chunk"]
            norm_txt = c_info["norm_text"]
            raw_words = c_info["raw_words"]
            num_words = len(raw_words)

            if not norm_txt or num_words == 0:
                aligned_chunks.append(
                    AlignedChunk(
                        chunk_id=chunk_obj.chunk_id,
                        chunk=chunk_obj,
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
                        candidate_tokens = emissions[curr_start : curr_start + k]
                        tok_words = [t.word for t in candidate_tokens]
                        candidate_norm_spaced = prep.normalize(" ".join(tok_words))
                        candidate_norm_concat = prep.normalize("".join(tok_words))

                        cost_spaced = metric.compute_cost(
                            hypothesis=candidate_norm_spaced, reference=norm_txt
                        )
                        cost_concat = metric.compute_cost(
                            hypothesis=candidate_norm_concat, reference=norm_txt
                        )
                        cost = min(cost_spaced, cost_concat)

                        if cost < best_cost:
                            best_cost = cost
                            best_start_idx = curr_start
                            best_end_idx = curr_start + k

            matched_tokens = emissions[best_start_idx:best_end_idx]

            if matched_tokens:
                chunk_start = matched_tokens[0].start_sec
                chunk_end = matched_tokens[-1].end_sec
                emitted_text = " ".join([t.word for t in matched_tokens])
                emitted_norm = prep.normalize(emitted_text)
                chunk_score = metric.compute_cost(
                    hypothesis=emitted_norm, reference=norm_txt
                )

                raw_syllabary_words = (
                    [w for w in chunk_obj.syllabary_text.split() if w]
                    if chunk_obj.syllabary_text
                    else None
                )

                word_intervals = self.word_aligner.align_words(
                    raw_words=raw_words,
                    matched_tokens=matched_tokens,
                    raw_syllabary_words=raw_syllabary_words,
                )

                if reconciliation_strategy and word_intervals:
                    for w_int in word_intervals:
                        target_emitted = w_int.emitted_word or emitted_text
                        target_syll = w_int.syllabary or chunk_obj.syllabary_text or ""
                        if target_syll and target_emitted:
                            try:
                                rec_word, _ = reconciliation_strategy.reconcile(
                                    syllabary_text=target_syll,
                                    emitted_text=target_emitted,
                                )
                                w_int.reconciled_word = rec_word
                            except Exception:
                                w_int.reconciled_word = w_int.word
                        else:
                            w_int.reconciled_word = w_int.word

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
                    chunk_id=chunk_obj.chunk_id,
                    chunk=chunk_obj,
                    start_sec=chunk_start,
                    end_sec=chunk_end,
                    words=word_intervals,
                    distance_score=round(chunk_score, 4),
                    emitted_text=emitted_text,
                )
            )

        metrics = self._compute_metrics(aligned_chunks)
        return AlignmentOutput(
            aligned_chunks=aligned_chunks,
            raw_tokens=list(emissions),
            metrics=metrics,
        )

    @classmethod
    def make_default(
        cls,
        word_aligner: Optional[NeedlemanWunschWordAligner] = None,
        distance_metric: Optional[DistanceMetric] = None,
        preprocessor: Optional[PhoneticPreprocessor] = None,
        reconciliation_strategy: Optional[ReconciliationStrategy] = None,
        max_preamble_skip: int = 30,
        max_normal_skip: int = 5,
    ) -> "SlidingWindowDTWAligner":
        """Factory method creating a SlidingWindowDTWAligner with default strategies and parameters."""
        metric = distance_metric or DefaultCERDistanceMetric()
        prep = preprocessor or CherokeePhoneticPreprocessor()
        aligner = word_aligner or NeedlemanWunschWordAligner.make_default(
            distance_metric=metric,
            preprocessor=prep,
        )
        return cls(
            word_aligner=aligner,
            distance_metric=metric,
            preprocessor=prep,
            reconciliation_strategy=reconciliation_strategy,
            max_preamble_skip=max_preamble_skip,
            max_normal_skip=max_normal_skip,
        )
