# -*- coding: utf-8 -*-
"""
ctc.py

Generic CTC Segmentation Aligner engine adapter integrating syncope-aware forward
DP trellis segmentation with ModelOutput universal currency and pluggable text preparer strategies.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np

from ctc_segmentation import (  # type: ignore
    CtcSegmentationParameters,
    ctc_segmentation,
    determine_utterance_segments,
)
from transcription.core.alignment.models import (
    AlignedChunk,
    AlignmentMetrics,
    AlignmentOutput,
    CTCAlignerConfig,
    TextChunk,
    TokenEmission,
    WordInterval,
)
from transcription.core.models.output import ModelOutput

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "ctc_emissions"


@runtime_checkable
class TextPreparerProtocol(Protocol):
    """Protocol for transforming words and config into ground-truth index matrix and utterance indices."""

    def __call__(
        self,
        config: Any,
        text: Union[str, Sequence[str]],
        char_list: Optional[Sequence[str]] = None,
        token_masks: Optional[Sequence[Tuple[Sequence[bool], Sequence[bool]]]] = None,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Prepares ground truth matrix and utterance begin indices for ctc_segmentation.

        Args:
            config: CtcSegmentationParameters configuration instance.
            text: Input text sequence or words.
            char_list: Ordered list of vocabulary character tokens.
            token_masks: Optional pre-computed sequence of (syncope_mask, intrusion_mask) tuples.

        Returns:
            Tuple of (ground_truth_mat, utt_begin_indices).
        """
        ...


def default_text_preparer(
    config: Any,
    text: Union[str, Sequence[str]],
    char_list: Optional[Sequence[str]] = None,
    token_masks: Optional[Sequence[Tuple[Sequence[bool], Sequence[bool]]]] = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Default language-agnostic ground truth preparation for ctc_segmentation.
    Maps words and character list into a simple sequential ground-truth transition matrix.
    """
    from ctc_segmentation import prepare_text  # type: ignore

    if isinstance(text, str):
        formatted_text = [text]
    else:
        formatted_text = list(text)
    return prepare_text(config, formatted_text)


class CTCSegmentationAligner:
    """
    Generic CTC Segmentation Aligner engine adapter.
    Leverages syncope-aware CTC trellis segmentation directly on acoustic frame
    log-probabilities (e.g. ModelOutput.lpz).
    Decoupled from Cherokee-specific models or phonotactics via injected TextPreparerProtocol.
    """

    def __init__(
        self,
        config: Optional[CTCAlignerConfig] = None,
        text_preparer: Optional[TextPreparerProtocol] = None,
        char_list: Optional[Sequence[str]] = None,
        pad_id: Optional[int] = None,
    ):
        self.config = config or CTCAlignerConfig()
        self.text_preparer: TextPreparerProtocol = (
            text_preparer or default_text_preparer
        )
        self._char_list = list(char_list) if char_list is not None else None
        self._pad_id = int(pad_id) if pad_id is not None else None

    def _resolve_char_list_and_blank(
        self, model_output: Optional[ModelOutput] = None
    ) -> Tuple[List[str], int]:
        """Resolves char_list and pad_id from explicit attributes or ModelOutput.vocab."""
        if self._char_list is not None and self._pad_id is not None:
            return self._char_list, self._pad_id

        if model_output is not None:
            id2tok = model_output.id_to_token
            max_id = max(id2tok.keys()) if id2tok else -1
            char_list = [id2tok.get(i, "") for i in range(max_id + 1)]
            pad_id = model_output.pad_token_id
            return char_list, pad_id

        raise ValueError(
            "char_list and pad_id must be provided during initialization or inferred from ModelOutput."
        )

    def _extract_word_intervals(
        self,
        words: Sequence[str],
        timings: np.ndarray,
        char_probs: np.ndarray,
        state_list: Sequence[str],
        utt_indices: Sequence[int],
        start_word_idx: int = 0,
        dur_sec: float = float("inf"),
        lead_offset_sec: float = 0.0,
        initial_prev_end: float = 0.0,
    ) -> List[WordInterval]:
        """
        Extract WordInterval objects from CTC segmentation timings and character states.
        """
        word_intervals: List[WordInterval] = []
        prev_end = initial_prev_end
        char_probs_flat = np.asarray(char_probs).ravel()
        timings_arr = np.asarray(timings)

        for local_w_i, raw_w in enumerate(words):
            global_w_i = start_word_idx + local_w_i
            start_idx = utt_indices[global_w_i]
            end_idx = utt_indices[global_w_i + 1]
            char_start_idx = min(start_idx + 1, end_idx)

            sub_timings = timings_arr[char_start_idx:end_idx]
            if len(sub_timings) > 0:
                mask = sub_timings > 0.0
                if (
                    global_w_i == 0
                    and len(state_list) > 0
                    and state_list[0] not in ("", "ε", "[PAD]")
                ):
                    mask[0] = True
                w_timings = sub_timings[mask]
            else:
                w_timings = np.array([], dtype=timings_arr.dtype)

            if len(w_timings) > 0:
                raw_w_start = float(np.min(w_timings))
                raw_w_end = float(np.max(w_timings)) + self.config.index_duration
                if end_idx < len(timings_arr) and timings_arr[end_idx] > 0.0:
                    raw_w_end = max(raw_w_end, float(timings_arr[end_idx]))

                w_start = max(
                    0.0, min(dur_sec, round(raw_w_start - lead_offset_sec, 3))
                )
                w_end = max(
                    w_start, min(dur_sec, round(raw_w_end - lead_offset_sec, 3))
                )
                start_f = int(round(raw_w_start / self.config.index_duration))
                end_f = int(round(raw_w_end / self.config.index_duration))
                f_stop = min(len(state_list), max(start_f + 1, end_f))
                sub_states = (
                    state_list[start_f:f_stop] if start_f < len(state_list) else []
                )
                sub_probs = (
                    char_probs_flat[start_f:f_stop]
                    if start_f < len(char_probs_flat)
                    else np.array([])
                )

                emitted_chars = [s for s in sub_states if s and s not in ("ε", "[PAD]")]
                emitted_w = "".join(emitted_chars) or raw_w

                char_peaks: List[float] = []
                offset = 0
                for s, group in itertools.groupby(sub_states):
                    group_len = sum(1 for _ in group)
                    if s and s not in ("ε", "[PAD]") and offset < len(sub_probs):
                        peak = float(
                            np.max(
                                sub_probs[
                                    offset : min(offset + group_len, len(sub_probs))
                                ]
                            )
                        )
                        char_peaks.append(peak)
                    offset += group_len

                if char_peaks:
                    mean_logprob = float(np.mean(char_peaks))
                    word_conf = float(np.exp(mean_logprob))
                    min_char_prob = float(np.exp(np.min(char_peaks)))
                else:
                    word_conf = 0.0
                    min_char_prob = 0.0
            else:
                w_start = prev_end
                w_end = prev_end
                emitted_chars = []
                emitted_w = ""
                word_conf = 0.0
                min_char_prob = 0.0

            is_low_conf = bool(word_conf < self.config.flag_min_confidence)
            is_low_char_conf = bool(
                self.config.flag_min_char_confidence > 0.0
                and min_char_prob < self.config.flag_min_char_confidence
            )
            is_unaligned = bool(len(w_timings) == 0 or len(emitted_chars) == 0)
            is_flagged = bool(is_low_conf or is_low_char_conf or is_unaligned)

            word_intervals.append(
                WordInterval(
                    word=raw_w,
                    start_sec=w_start,
                    end_sec=w_end,
                    confidence=round(word_conf, 6),
                    flagged=is_flagged,
                    emitted_word=emitted_w,
                    min_char_confidence=round(min_char_prob, 6),
                )
            )
            prev_end = w_end

        return word_intervals

    def align(
        self,
        output: Union[ModelOutput, np.ndarray],
        chunks: Sequence[TextChunk],
        source_id: str = "",
        char_list: Optional[Sequence[str]] = None,
        pad_id: Optional[int] = None,
        source_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> AlignmentOutput:
        """
        Aligns text chunks against acoustic frame log-probabilities.

        Args:
            output: Either a ModelOutput instance or a raw 2D numpy array of log-probabilities (T, V).
            chunks: Sequence of TextChunk reference units to align.
            source_id: Optional identifier for source utterance/audio.
            char_list: Optional ordered vocabulary character tokens (overrides instance default).
            pad_id: Optional blank/pad token ID (overrides instance default).
            source_metadata: Optional metadata (e.g. code-switching masks) passed to text preparer.

        Returns:
            AlignmentOutput with aligned chunks and overall metrics.
        """
        model_out = output if isinstance(output, ModelOutput) else None
        lpz = output.lpz if isinstance(output, ModelOutput) else np.asarray(output)

        if lpz.ndim == 3:
            if lpz.shape[0] == 1:
                lpz = lpz[0]
            else:
                raise ValueError("align expects 2D (T, V) log-probabilities.")

        if char_list is not None and pad_id is not None:
            c_list = list(char_list)
            p_id = int(pad_id)
        elif self._char_list is not None and self._pad_id is not None:
            c_list = self._char_list
            p_id = self._pad_id
        else:
            c_list, p_id = self._resolve_char_list_and_blank(model_out)

        dur_sec = float(lpz.shape[0] * self.config.index_duration)

        if not chunks:
            return AlignmentOutput(
                aligned_chunks=[],
                source_id=source_id,
                raw_tokens=[],
                metrics=AlignmentMetrics(
                    total_chunks=0,
                    matched_chunks=0,
                    match_ratio=0.0,
                    mean_distance_score=0.0,
                    total_ground_truth_chars=0,
                    total_emitted_chars=0,
                    flagged_words_count=0,
                ),
            )

        win_size = max(self.config.min_window_size, min(20000, int(lpz.shape[0])))
        max_win = max(self.config.max_window_size, win_size * 2)

        # Filter syncope and intrusive tokens against vocabulary to avoid KeyError in ctc_segmentation
        valid_syncope: List[Any] = []
        for item in self.config.syncope_tokens:
            if isinstance(item, (list, tuple)):
                filtered_sub = [tok for tok in item if tok in c_list]
                if filtered_sub:
                    valid_syncope.append(
                        tuple(filtered_sub) if isinstance(item, tuple) else filtered_sub
                    )
            elif isinstance(item, str) and item in c_list:
                valid_syncope.append(item)

        valid_intrusive = [tok for tok in self.config.intrusive_tokens if tok in c_list]

        config = CtcSegmentationParameters(
            char_list=c_list,
            blank=p_id,
            syncope_tokens=valid_syncope,
            intrusive_tokens=valid_intrusive,
            intrusive_max_stride=self.config.intrusive_max_stride,
            index_duration=self.config.index_duration,
            min_window_size=win_size,
            max_window_size=max_win,
            score_min_mean_over_L=2,
            replace_spaces_with_blanks=False,
        )

        all_words: List[str] = []
        all_token_masks: List[Tuple[Sequence[bool], Sequence[bool]]] = []
        chunk_word_slices: List[Tuple[int, int]] = []
        has_custom_masks = False

        for c in chunks:
            chunk_words = [w for w in (c.text or "").split() if w]
            w_start = len(all_words)
            all_words.extend(chunk_words)
            chunk_word_slices.append((w_start, len(all_words)))

            cs_meta = (
                source_metadata.get(c.chunk_id, {}).get("code_switched")
                if source_metadata
                else None
            )
            if isinstance(cs_meta, dict):
                has_custom_masks = True
                for t in cs_meta.get("tokens", []):
                    if t.get("canonical_tth"):
                        sync_m = t.get("syncope_mask", [])
                        intrus_m = t.get("intrusion_mask", [])
                        all_token_masks.append((sync_m, intrus_m))
            else:
                for _ in chunk_words:
                    all_token_masks.append(((), ()))

        if not all_words or lpz.shape[0] == 0:
            aligned_chunks = [
                AlignedChunk(
                    chunk_id=c.chunk_id,
                    start_sec=0.0,
                    end_sec=dur_sec,
                    words=[],
                    distance_score=1.0,
                    emitted_text="",
                )
                for c in chunks
            ]
            return AlignmentOutput(
                aligned_chunks=aligned_chunks,
                source_id=source_id,
                raw_tokens=[],
                metrics=AlignmentMetrics(
                    total_chunks=len(chunks),
                    matched_chunks=0,
                    match_ratio=0.0,
                    mean_distance_score=1.0,
                    total_ground_truth_chars=sum(len(c.text) for c in chunks),
                    total_emitted_chars=0,
                    flagged_words_count=0,
                ),
            )

        gt_mat, utt_indices = self.text_preparer(
            config,
            all_words,
            c_list,
            token_masks=all_token_masks if has_custom_masks else None,
        )

        timings, char_probs, state_list = ctc_segmentation(config, lpz, gt_mat)

        chunk_utt_indices = [utt_indices[s] for s, _ in chunk_word_slices] + [
            utt_indices[-1]
        ]
        chunk_texts = [" ".join(all_words[s:e]) for s, e in chunk_word_slices]
        raw_segments = determine_utterance_segments(
            config, chunk_utt_indices, char_probs, timings, chunk_texts
        )

        chunk_word_intervals: List[List[WordInterval]] = []
        prev_end = 0.0
        for c_idx in range(len(chunks)):
            w_start_idx, w_end_idx = chunk_word_slices[c_idx]
            chunk_words = all_words[w_start_idx:w_end_idx]
            w_ints = self._extract_word_intervals(
                words=chunk_words,
                timings=timings,
                char_probs=char_probs,
                state_list=state_list,
                utt_indices=utt_indices,
                start_word_idx=w_start_idx,
                dur_sec=dur_sec,
                lead_offset_sec=0.0,
                initial_prev_end=prev_end,
            )
            if w_ints:
                prev_end = w_ints[-1].end_sec
            chunk_word_intervals.append(w_ints)

        # Compute acoustic core envelope for each chunk
        core_starts: List[float] = []
        core_ends: List[float] = []
        for c_idx in range(len(chunks)):
            raw_s, raw_e, _ = raw_segments[c_idx]
            w_ints = chunk_word_intervals[c_idx]
            aligned_w_starts = [
                w.start_sec for w in w_ints if not w.flagged and w.end_sec > w.start_sec
            ]
            aligned_w_ends = [
                w.end_sec for w in w_ints if not w.flagged and w.end_sec > w.start_sec
            ]

            if aligned_w_starts and aligned_w_ends:
                w_min = min(aligned_w_starts)
                w_max = max(aligned_w_ends)
                c_start = (
                    raw_s
                    if (raw_s > 0.0 and abs(raw_s - w_min) < 1.0 and raw_s <= w_min)
                    else w_min
                )
                c_end = (
                    raw_e
                    if (raw_e > 0.0 and abs(raw_e - w_max) < 1.0 and raw_e >= w_max)
                    else w_max
                )
            else:
                c_start = raw_s if raw_s > 0.0 else 0.0
                c_end = raw_e if raw_e > 0.0 else c_start

            core_starts.append(c_start)
            core_ends.append(c_end)

        # Apply boundary padding and resolve adjacent boundaries at midpoint
        num_chunks = len(chunks)
        c_starts = [0.0] * num_chunks
        c_ends = [0.0] * num_chunks

        if num_chunks > 0:
            c_starts[0] = max(
                0.0, round(core_starts[0] - self.config.boundary_pad_sec, 3)
            )
            for i in range(num_chunks - 1):
                mid = (core_ends[i] + core_starts[i + 1]) / 2.0
                mid = max(c_starts[i], min(dur_sec, mid))
                c_ends[i] = round(mid, 3)
                c_starts[i + 1] = round(mid, 3)
            c_ends[-1] = min(
                dur_sec,
                round(
                    max(c_starts[-1], core_ends[-1] + self.config.boundary_pad_sec), 3
                ),
            )

            # Strictly ensure boundaries cover all valid words and respect [0.0, dur_sec] monotonically
            for i in range(num_chunks):
                w_ints = chunk_word_intervals[i]
                valid_w = [w for w in w_ints if w.end_sec > w.start_sec]
                if valid_w:
                    c_starts[i] = min(c_starts[i], valid_w[0].start_sec)
                    c_ends[i] = max(c_ends[i], valid_w[-1].end_sec)
                c_starts[i] = max(0.0, min(c_starts[i], dur_sec))
                c_ends[i] = max(c_starts[i], min(c_ends[i], dur_sec))
                if i > 0:
                    c_starts[i] = max(c_starts[i], c_ends[i - 1])
                    c_ends[i] = max(c_starts[i], c_ends[i])

        aligned_chunks: List[AlignedChunk] = []
        for i in range(num_chunks):
            c = chunks[i]
            final_start = c_starts[i]
            final_end = c_ends[i]

            w_ints = chunk_word_intervals[i]
            emitted_text = " ".join([w.emitted_word for w in w_ints if w.emitted_word])

            total_chars = len(c.text.replace(" ", ""))
            emitted_chars = len(emitted_text.replace(" ", ""))
            dist_score = (
                abs(total_chars - emitted_chars) / max(1, total_chars)
                if total_chars > 0
                else 0.0
            )
            dist_score = min(1.0, max(0.0, dist_score))

            aligned_chunks.append(
                AlignedChunk(
                    chunk_id=c.chunk_id,
                    start_sec=round(final_start, 3),
                    end_sec=round(final_end, 3),
                    words=w_ints,
                    distance_score=round(dist_score, 4),
                    emitted_text=emitted_text,
                )
            )

        matched_count = sum(1 for ac in aligned_chunks if ac.end_sec > ac.start_sec)
        match_ratio = round(matched_count / max(1, len(aligned_chunks)), 4)
        mean_score = (
            round(
                float(np.mean([ac.distance_score for ac in aligned_chunks])),
                4,
            )
            if aligned_chunks
            else 0.0
        )
        total_gt = sum(len(c.text) for c in chunks)
        total_em = sum(len(ac.emitted_text) for ac in aligned_chunks)
        flagged_w_cnt = sum(len(ac.flagged_words) for ac in aligned_chunks)

        metrics = AlignmentMetrics(
            total_chunks=len(chunks),
            matched_chunks=matched_count,
            match_ratio=match_ratio,
            mean_distance_score=mean_score,
            total_ground_truth_chars=total_gt,
            total_emitted_chars=total_em,
            flagged_words_count=flagged_w_cnt,
        )

        return AlignmentOutput(
            aligned_chunks=aligned_chunks,
            source_id=source_id,
            raw_tokens=[],
            metrics=metrics,
        )


__all__ = [
    "CTCSegmentationAligner",
    "TextPreparerProtocol",
    "default_text_preparer",
]
