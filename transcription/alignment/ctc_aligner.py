# -*- coding: utf-8 -*-
"""
ctc_aligner.py

CTC Segmentation Aligner adapter integrating syncope-aware forward DP trellis
segmentation with workshop-transcription alignment domain models and pipelines.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pydub import AudioSegment
import torch

from ctc_segmentation import (  # type: ignore
    CtcSegmentationParameters,
    ctc_segmentation,
    determine_utterance_segments,
    prepare_text,
)
from transcription.alignment.models import (
    AlignedChunk,
    AlignmentMetrics,
    AlignmentOutput,
    CTCAlignerConfig,
    TextChunk,
    TokenEmission,
    WordInterval,
)
from transcription.alignment.normalizers import (
    normalize_phonetics_for_alignment,
    normalize_syllabary_for_alignment,
)
from transcription.alignment.phonotactics import prepare_cherokee_text
from transcription.models.asr_model import CherokeeASRModel

logger = logging.getLogger(__name__)

DEFAULT_SYNCOPE_TOKENS = ("a", "e", "i", "o", "u", "v")
DEFAULT_CACHE_DIR = (
    Path(__file__).resolve().parent.parent.parent / ".cache" / "ctc_emissions"
)


class CTCSegmentationAligner:
    """
    Alignment engine adapter leveraging syncope-aware CTC trellis segmentation.
    Maps acoustic frame log-probabilities directly to character/word/chunk sequences,
    supporting vowel deletion transitions (syncope) in spoken Cherokee.
    """

    def __init__(
        self,
        model: Optional[CherokeeASRModel] = None,
        config: Optional[CTCAlignerConfig] = None,
        **kwargs: Any,
    ):
        self.model = model
        self.config = config or CTCAlignerConfig()

        self.syncope_tokens = list(self.config.syncope_tokens)
        self.intrusive_tokens = (
            list(self.config.intrusive_tokens) if self.config.intrusive_tokens else []
        )
        self.intrusive_max_stride = int(self.config.intrusive_max_stride)
        self.enforce_phonotactics = bool(self.config.enforce_phonotactics)
        self.flag_min_confidence = float(self.config.flag_min_confidence)
        self.flag_min_char_confidence = float(self.config.flag_min_char_confidence)
        self.index_duration = float(self.config.index_duration)
        self.min_window_size = int(self.config.min_window_size)
        self.max_window_size = int(self.config.max_window_size)
        self.buffer_trail_ms = int(self.config.buffer_trail_ms)
        self.buffer_lead_ms = int(self.config.buffer_lead_ms)
        self.boundary_pad_sec = float(getattr(self.config, "boundary_pad_sec", 0.1))
        self.chunk_seconds = float(self.config.chunk_seconds)
        self.margin_seconds = float(self.config.margin_seconds)
        self.cache = bool(self.config.cache)
        self.cache_dir = (
            Path(self.config.cache_dir)
            if self.config.cache_dir is not None
            else DEFAULT_CACHE_DIR
        )

    def _get_char_list_and_blank(
        self, asr_model: CherokeeASRModel
    ) -> Tuple[List[str], int]:
        vocab = asr_model.processor.tokenizer.get_vocab()
        char_list = [token for token, idx in sorted(vocab.items(), key=lambda x: x[1])]
        pad_id = getattr(asr_model.processor.tokenizer, "pad_token_id", None)
        if pad_id is None or not isinstance(pad_id, (int, np.integer)):
            pad_id = vocab.get("[PAD]", 0)
        if not isinstance(pad_id, (int, np.integer)):
            pad_id = 0
        return char_list, int(pad_id)

    def extract_logits(
        self,
        audio_input: Union[str, Path, AudioSegment, np.ndarray],
        asr_model: Optional[CherokeeASRModel] = None,
        chunk_seconds: Optional[float] = None,
        margin_seconds: Optional[float] = None,
        apply_buffers: bool = True,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Extract log-probability matrix (T, V) from audio input using ASR model.
        Uses overlapping sliding-window inference with margin trimming for long files.
        Returns: (lpz, raw_audio_duration_sec, lead_offset_sec)
        """
        model = asr_model or self.model
        if model is None:
            raise ValueError("ASR model must be provided to extract logits.")

        c_sec = self.chunk_seconds if chunk_seconds is None else float(chunk_seconds)
        m_sec = self.margin_seconds if margin_seconds is None else float(margin_seconds)

        if isinstance(audio_input, (str, Path)):
            audio = (
                AudioSegment.from_file(str(audio_input))
                .set_frame_rate(16000)
                .set_channels(1)
            )
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        elif isinstance(audio_input, AudioSegment):
            audio = audio_input.set_frame_rate(16000).set_channels(1)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        elif isinstance(audio_input, np.ndarray):
            samples = audio_input.astype(np.float32)
            if samples.ndim > 1:
                samples = samples.mean(axis=-1)
        else:
            raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

        total_audio_sec = len(samples) / 16000.0
        lead_offset_sec = 0.0

        if apply_buffers:
            lead_n = int(16000 * (self.buffer_lead_ms / 1000.0))
            trail_n = int(16000 * (self.buffer_trail_ms / 1000.0))
            lead_pad = (
                np.zeros(lead_n, dtype=np.float32)
                if lead_n > 0
                else np.array([], dtype=np.float32)
            )
            trail_pad = (
                np.zeros(trail_n, dtype=np.float32)
                if trail_n > 0
                else np.array([], dtype=np.float32)
            )
            samples = np.concatenate([lead_pad, samples, trail_pad])
            lead_offset_sec = self.buffer_lead_ms / 1000.0

        chunk_len = int(16000 * c_sec)
        if len(samples) <= chunk_len or m_sec <= 0:
            logits = model.get_logits(samples, sample_rate=16000)
            if logits.ndim == 3:
                logits = logits.squeeze(0)
            lpz = torch.nn.functional.log_softmax(logits, dim=-1).cpu().numpy()
        else:
            if hasattr(model, "get_logits_sliding_window") and callable(
                getattr(model, "get_logits_sliding_window")
            ):
                logits = model.get_logits_sliding_window(
                    samples,
                    sample_rate=16000,
                    chunk_seconds=c_sec,
                    margin_seconds=m_sec,
                )
            else:
                margin_samples = int(16000 * m_sec)
                step_samples = chunk_len - 2 * margin_samples
                if step_samples <= 0:
                    step_samples = chunk_len
                margin_frames = int(round(m_sec / self.index_duration))
                cur_start = 0
                total_samples = len(samples)
                logits_list: List[torch.Tensor] = []
                while cur_start < total_samples:
                    cur_end = min(cur_start + chunk_len, total_samples)
                    chunk = samples[cur_start:cur_end]
                    chunk_logits = model.get_logits(chunk, sample_rate=16000)
                    if chunk_logits.ndim == 3:
                        chunk_logits = chunk_logits.squeeze(0)
                    T_chunk = chunk_logits.shape[0]
                    is_first = cur_start == 0
                    is_last = cur_end >= total_samples
                    left_trim = 0 if is_first else margin_frames
                    right_trim = 0 if is_last else margin_frames
                    left_idx = min(left_trim, T_chunk)
                    right_idx = max(left_idx, T_chunk - right_trim)
                    logits_list.append(chunk_logits[left_idx:right_idx])
                    if is_last:
                        break
                    cur_start += step_samples
                logits = torch.cat(logits_list, dim=0)

            if logits.ndim == 3:
                logits = logits.squeeze(0)
            lpz = torch.nn.functional.log_softmax(logits, dim=-1).cpu().numpy()

        return lpz, total_audio_sec, lead_offset_sec

    def _compute_cache_key(
        self,
        audio_input: Union[str, Path, AudioSegment, np.ndarray],
        asr_model: Optional[CherokeeASRModel] = None,
        chunk_seconds: Optional[float] = None,
        margin_seconds: Optional[float] = None,
        apply_buffers: bool = True,
    ) -> str:
        model = asr_model or self.model
        model_name = getattr(model, "model_name", None) or "cherokee_asr"
        c_sec = self.chunk_seconds if chunk_seconds is None else float(chunk_seconds)
        m_sec = self.margin_seconds if margin_seconds is None else float(margin_seconds)

        stem = "audio"
        if isinstance(audio_input, (str, Path)):
            path_obj = Path(audio_input)
            stem = path_obj.stem
            resolved = str(path_obj.resolve()) if path_obj.exists() else str(path_obj)
            if os.path.exists(resolved):
                st = os.stat(resolved)
                input_id = f"file:{resolved}:{st.st_size}:{st.st_mtime}"
            else:
                input_id = f"file:{resolved}"
        elif isinstance(audio_input, AudioSegment):
            raw_bytes = bytes(audio_input.raw_data or b"")
            content_hash = hashlib.sha256(raw_bytes).hexdigest()[:16]
            input_id = f"audioseg:{audio_input.frame_rate}:{audio_input.channels}:{audio_input.sample_width}:{len(audio_input)}:{content_hash}"
        elif isinstance(audio_input, np.ndarray):
            content_hash = hashlib.sha256(audio_input.tobytes()).hexdigest()[:16]
            input_id = f"ndarray:{audio_input.shape}:{audio_input.dtype}:{content_hash}"
        else:
            input_id = f"custom:{repr(audio_input)}"

        lead = self.buffer_lead_ms if apply_buffers else 0
        trail = self.buffer_trail_ms if apply_buffers else 0
        buf_str = f"lead={lead}:trail={trail}:chunk={c_sec}:margin={m_sec}"
        full_key = f"{model_name}|{input_id}|{buf_str}"
        cache_hash = hashlib.sha256(full_key.encode("utf-8")).hexdigest()
        return f"{stem}_{cache_hash[:16]}"

    def get_logits_cached(
        self,
        audio_input: Union[str, Path, AudioSegment, np.ndarray],
        asr_model: Optional[CherokeeASRModel] = None,
        chunk_seconds: Optional[float] = None,
        margin_seconds: Optional[float] = None,
        apply_buffers: bool = True,
        cache: Optional[bool] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Extract log-probability matrix (T, V) from audio input using ASR model,
        or load cached result from disk if caching is enabled.

        Parameters
        ----------
        audio_input : Union[str, Path, AudioSegment, np.ndarray]
            Audio input to extract logits for.
        asr_model : Optional[CherokeeASRModel]
            ASR model to use. If None, uses self.model.
        chunk_seconds : Optional[float]
            Chunk length for batched audio processing. Defaults to self.chunk_seconds.
        margin_seconds : Optional[float]
            Margin length for sliding-window trimming. Defaults to self.margin_seconds.
        apply_buffers : bool
            Whether to pad audio with lead/trail context buffers.
        cache : Optional[bool]
            Whether to use disk caching. If None, defaults to self.cache.
        cache_dir : Optional[Union[str, Path]]
            Directory to store/load cache files. If None, defaults to self.cache_dir.

        Returns
        -------
        Tuple[np.ndarray, float, float]
            (lpz, total_audio_sec, lead_offset_sec)
        """
        use_cache = self.cache if cache is None else bool(cache)
        target_cache_dir = Path(cache_dir) if cache_dir is not None else self.cache_dir
        c_sec = self.chunk_seconds if chunk_seconds is None else float(chunk_seconds)
        m_sec = self.margin_seconds if margin_seconds is None else float(margin_seconds)

        if not use_cache:
            return self.extract_logits(
                audio_input=audio_input,
                asr_model=asr_model,
                chunk_seconds=c_sec,
                margin_seconds=m_sec,
                apply_buffers=apply_buffers,
            )

        target_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = self._compute_cache_key(
            audio_input=audio_input,
            asr_model=asr_model,
            chunk_seconds=c_sec,
            margin_seconds=m_sec,
            apply_buffers=apply_buffers,
        )
        cache_path = target_cache_dir / f"{cache_key}.npz"

        if cache_path.exists():
            try:
                with np.load(cache_path) as data:
                    lpz = data["lpz"]
                    dur_sec = float(data["total_audio_sec"])
                    lead_offset_sec = float(data["lead_offset_sec"])
                return lpz, dur_sec, lead_offset_sec
            except Exception as e:
                logger.warning(
                    "Failed to read cached CTC logits from %s (%s). Recomputing...",
                    cache_path,
                    e,
                )

        # Cache miss: compute forward pass
        lpz, dur_sec, lead_offset_sec = self.extract_logits(
            audio_input=audio_input,
            asr_model=asr_model,
            chunk_seconds=c_sec,
            margin_seconds=m_sec,
            apply_buffers=apply_buffers,
        )

        try:
            np.savez_compressed(
                cache_path,
                lpz=lpz.astype(np.float32),
                total_audio_sec=np.array(dur_sec, dtype=np.float64),
                lead_offset_sec=np.array(lead_offset_sec, dtype=np.float64),
            )
        except Exception as e:
            logger.warning("Failed to write CTC logits cache to %s (%s)", cache_path, e)

        return lpz, dur_sec, lead_offset_sec

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

        Parameters
        ----------
        words : Sequence[str]
            List of word tokens to extract intervals for.
        timings : np.ndarray
            Array of frame timing alignments from ctc_segmentation.
        char_probs : np.ndarray
            Acoustic posterior probability array per frame.
        state_list : Sequence[str]
            Backtracked character/token state emissions per frame.
        utt_indices : Sequence[int]
            Indices in ground-truth matrix denoting utterance/word boundaries.
        start_word_idx : int
            Starting index in utt_indices corresponding to the first word in words.
        dur_sec : float
            Maximum duration of the audio slice / chunk in seconds.
        lead_offset_sec : float
            Lead buffer duration in seconds to subtract from raw timings.
        initial_prev_end : float
            Initial end timestamp of preceding word for unaligned token fallbacks.

        Returns
        -------
        List[WordInterval]
            List of extracted word interval metadata objects.
        """
        word_intervals: List[WordInterval] = []
        prev_end = initial_prev_end

        for local_w_i, raw_w in enumerate(words):
            global_w_i = start_word_idx + local_w_i
            start_idx = utt_indices[global_w_i]
            end_idx = utt_indices[global_w_i + 1]
            char_start_idx = min(start_idx + 1, end_idx)

            # ctc_segmentation initializes unvisited trellis character slots to 0.0.
            # Valid character alignments have t > 0.0, except possibly the very first
            # character of the first word at frame 0 if aligned to non-blank acoustic state.
            w_timings = [
                t
                for idx, t in enumerate(timings[char_start_idx:end_idx])
                if t > 0.0
                or (
                    global_w_i == 0
                    and idx == 0
                    and t == 0.0
                    and len(state_list) > 0
                    and state_list[0] not in ("", "ε", "[PAD]")
                )
            ]

            if w_timings:
                raw_w_start = min(w_timings)
                raw_w_end = max(w_timings) + self.index_duration
                if end_idx < len(timings) and timings[end_idx] > 0.0:
                    raw_w_end = max(raw_w_end, float(timings[end_idx]))

                w_start = max(
                    0.0, min(dur_sec, round(raw_w_start - lead_offset_sec, 3))
                )
                w_end = max(
                    w_start, min(dur_sec, round(raw_w_end - lead_offset_sec, 3))
                )
                start_f = int(round(raw_w_start / self.index_duration))
                end_f = int(round(raw_w_end / self.index_duration))
                emitted_chars = [
                    s
                    for s in state_list[start_f : max(start_f + 1, end_f)]
                    if s and s != "ε" and s != "[PAD]"
                ]
                emitted_w = "".join(emitted_chars) or raw_w

                char_peaks: List[float] = []
                current_char: Optional[str] = None
                current_lps: List[float] = []

                for f in range(start_f, max(start_f + 1, end_f)):
                    s = state_list[f] if f < len(state_list) else ""
                    if s and s not in ("ε", "[PAD]"):
                        if s == current_char:
                            current_lps.append(float(char_probs[f]))
                        else:
                            if current_char is not None and current_lps:
                                char_peaks.append(max(current_lps))
                            current_char = s
                            current_lps = [float(char_probs[f])]
                    else:
                        if current_char is not None and current_lps:
                            char_peaks.append(max(current_lps))
                            current_char = None
                            current_lps = []
                if current_char is not None and current_lps:
                    char_peaks.append(max(current_lps))

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

            is_low_conf = bool(word_conf < self.flag_min_confidence)
            is_low_char_conf = bool(
                self.flag_min_char_confidence > 0.0
                and min_char_prob < self.flag_min_char_confidence
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

    def align_verse_slice(
        self,
        audio_input: Union[str, Path, AudioSegment, np.ndarray],
        chunk_id: str,
        phonetic_text: str,
        syllabary_text: str = "",
        asr_model: Optional[CherokeeASRModel] = None,
        cache: Optional[bool] = None,
    ) -> AlignedChunk:
        """
        Performs intra-verse CTC segmentation on a pre-cut verse audio slice,
        computing precise word intervals and emitted hypothesis text.
        """
        model = asr_model or self.model
        if model is None:
            raise ValueError("ASR model must be provided.")

        lpz, dur_sec, lead_offset_sec = self.get_logits_cached(
            audio_input,
            asr_model=model,
            apply_buffers=True,
            cache=cache,
        )
        char_list, pad_id = self._get_char_list_and_blank(model)

        target_text = (phonetic_text if phonetic_text else syllabary_text) or ""
        words = [w for w in target_text.split() if w]

        if not words or lpz.shape[0] == 0:
            return AlignedChunk(
                chunk_id=chunk_id,
                start_sec=0.0,
                end_sec=dur_sec,
                words=[],
                distance_score=1.0,
                emitted_text="",
            )

        config = CtcSegmentationParameters(
            char_list=char_list,
            blank=pad_id,
            syncope_tokens=self.syncope_tokens,
            intrusive_tokens=self.intrusive_tokens,
            intrusive_max_stride=self.intrusive_max_stride,
            index_duration=self.index_duration,
            score_min_mean_over_L=2,
            replace_spaces_with_blanks=False,
        )

        gt_mat, utt_indices = prepare_cherokee_text(
            config, words, char_list, enforce_phonotactics=self.enforce_phonotactics
        )
        timings, char_probs, state_list = ctc_segmentation(config, lpz, gt_mat)

        word_intervals = self._extract_word_intervals(
            words=words,
            timings=timings,
            char_probs=char_probs,
            state_list=state_list,
            utt_indices=utt_indices,
            start_word_idx=0,
            dur_sec=dur_sec,
            lead_offset_sec=lead_offset_sec,
            initial_prev_end=0.0,
        )

        chunk_start = word_intervals[0].start_sec if word_intervals else 0.0
        chunk_end = (
            max(chunk_start, word_intervals[-1].end_sec) if word_intervals else dur_sec
        )

        # Construct emitted hypothesis directly from backtracked CTC trellis path
        emitted_text = " ".join(
            [w.emitted_word for w in word_intervals if w.emitted_word]
        )

        # Raw ASR decode for confidence/distance metrics
        asr_res = model.decode(lpz)

        return AlignedChunk(
            chunk_id=chunk_id,
            start_sec=chunk_start,
            end_sec=chunk_end,
            words=word_intervals,
            distance_score=round(1.0 - asr_res.confidence, 4),
            emitted_text=emitted_text,
        )

    def align(
        self,
        audio_input: Union[str, Path, AudioSegment, np.ndarray],
        chunks: Sequence[TextChunk],
        source_id: str = "",
        asr_model: Optional[CherokeeASRModel] = None,
        cache: Optional[bool] = None,
    ) -> AlignmentOutput:
        """
        Full chapter/recording alignment:
        Runs syncope-aware CTC segmentation on continuous audio to obtain
        verse-level chunks and intra-verse word intervals.
        """
        model = asr_model or self.model
        if model is None:
            raise ValueError("ASR model must be provided.")

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

        lpz, dur_sec, _ = self.get_logits_cached(
            audio_input, asr_model=model, apply_buffers=False, cache=cache
        )
        char_list, pad_id = self._get_char_list_and_blank(model)

        win_size = max(self.min_window_size, min(20000, int(lpz.shape[0])))
        max_win = max(self.max_window_size, win_size * 2)

        config = CtcSegmentationParameters(
            char_list=char_list,
            blank=pad_id,
            syncope_tokens=self.syncope_tokens,
            intrusive_tokens=self.intrusive_tokens,
            intrusive_max_stride=self.intrusive_max_stride,
            index_duration=self.index_duration,
            min_window_size=win_size,
            max_window_size=max_win,
            score_min_mean_over_L=2,
            replace_spaces_with_blanks=False,
        )

        all_words: List[str] = []
        chunk_word_slices: List[Tuple[int, int]] = []
        for c in chunks:
            chunk_words = [w for w in (c.text or "").split() if w]
            w_start = len(all_words)
            all_words.extend(chunk_words)
            chunk_word_slices.append((w_start, len(all_words)))

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

        gt_mat, utt_indices = prepare_cherokee_text(
            config, all_words, char_list, enforce_phonotactics=self.enforce_phonotactics
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
            valid_w = [w for w in w_ints if w.end_sec > w.start_sec]
            if valid_w:
                cs = valid_w[0].start_sec
                ce = valid_w[-1].end_sec
            else:
                cs = max(0.0, float(raw_s) if float(raw_s) >= 0.0 else 0.0)
                ce = max(cs, float(raw_e) if float(raw_e) >= 0.0 else cs)
            core_starts.append(cs)
            core_ends.append(ce)

        # Apply boundary padding and resolve adjacent boundaries at midpoint
        num_chunks = len(chunks)
        c_starts = [0.0] * num_chunks
        c_ends = [0.0] * num_chunks

        if num_chunks > 0:
            c_starts[0] = max(0.0, round(core_starts[0] - self.boundary_pad_sec, 3))
            for i in range(num_chunks - 1):
                mid = (core_ends[i] + core_starts[i + 1]) / 2.0
                mid = max(c_starts[i], min(dur_sec, mid))
                c_ends[i] = round(mid, 3)
                c_starts[i + 1] = round(mid, 3)
            c_ends[-1] = min(
                dur_sec,
                round(max(c_starts[-1], core_ends[-1] + self.boundary_pad_sec), 3),
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
        for c_idx, chunk in enumerate(chunks):
            word_intervals = chunk_word_intervals[c_idx]
            c_start = c_starts[c_idx]
            c_end = c_ends[c_idx]
            _, _, score = raw_segments[c_idx]

            if word_intervals:
                emitted_text = " ".join(
                    [w.emitted_word for w in word_intervals if w.emitted_word]
                )
            else:
                start_frame = int(round(c_start / self.index_duration))
                end_frame = int(round(c_end / self.index_duration))
                emitted_chars = [
                    s
                    for s in state_list[start_frame:end_frame]
                    if s and s != "ε" and s != "[PAD]"
                ]
                emitted_text = " ".join(
                    "".join(emitted_chars).replace("|", " ").split()
                )

            aligned_chunks.append(
                AlignedChunk(
                    chunk_id=chunk.chunk_id,
                    start_sec=c_start,
                    end_sec=c_end,
                    words=word_intervals,
                    distance_score=round(float(score), 4),
                    emitted_text=emitted_text,
                )
            )

        # Compute summary metrics
        matched_chunks = [c for c in aligned_chunks if c.end_sec > c.start_sec]
        match_ratio = round(len(matched_chunks) / max(1, len(chunks)), 4)
        mean_score = round(
            float(
                np.mean([c.distance_score for c in matched_chunks])
                if matched_chunks
                else 0.0
            ),
            4,
        )

        metrics = AlignmentMetrics(
            total_chunks=len(chunks),
            matched_chunks=len(matched_chunks),
            match_ratio=match_ratio,
            mean_distance_score=mean_score,
            total_ground_truth_chars=sum(len(c.text) for c in chunks),
            total_emitted_chars=sum(len(c.emitted_text) for c in aligned_chunks),
            flagged_words_count=sum(len(c.flagged_words) for c in aligned_chunks),
        )

        return AlignmentOutput(
            aligned_chunks=aligned_chunks,
            source_id=source_id,
            raw_tokens=[],
            metrics=metrics,
        )


def get_logits_cached(
    audio_input: Union[str, Path, AudioSegment, np.ndarray],
    asr_model: Optional[CherokeeASRModel] = None,
    chunk_seconds: float = 30.0,
    margin_seconds: float = 1.0,
    apply_buffers: bool = True,
    cache: bool = True,
    cache_dir: Optional[Union[str, Path]] = None,
    buffer_lead_ms: int = 100,
    buffer_trail_ms: int = 300,
) -> Tuple[np.ndarray, float, float]:
    """
    Convenience function to extract or load cached CTC acoustic log-probabilities (lpz).

    Parameters
    ----------
    audio_input : Union[str, Path, AudioSegment, np.ndarray]
        Audio input (file path, AudioSegment, or ndarray).
    asr_model : Optional[CherokeeASRModel]
        ASR model instance. Required if cache is False or on cache miss.
    chunk_seconds : float
        Audio chunk size in seconds for batched model forward passes.
    margin_seconds : float
        Margin length in seconds for sliding-window trimming.
    apply_buffers : bool
        Whether to pad lead/trail acoustic context buffers.
    cache : bool
        Whether to check and write to disk cache (default True).
    cache_dir : Optional[Union[str, Path]]
        Custom directory to store cache files (default: .cache/ctc_emissions).
    buffer_lead_ms : int
        Lead buffer in milliseconds (default 100).
    buffer_trail_ms : int
        Trail buffer in milliseconds (default 300).

    Returns
    -------
    Tuple[np.ndarray, float, float]
        (lpz, total_audio_sec, lead_offset_sec)
    """
    aligner = CTCSegmentationAligner(
        model=asr_model,
        buffer_lead_ms=buffer_lead_ms,
        buffer_trail_ms=buffer_trail_ms,
        chunk_seconds=chunk_seconds,
        margin_seconds=margin_seconds,
        cache=cache,
        cache_dir=cache_dir,
    )
    return aligner.get_logits_cached(
        audio_input=audio_input,
        asr_model=asr_model,
        chunk_seconds=chunk_seconds,
        margin_seconds=margin_seconds,
        apply_buffers=apply_buffers,
        cache=cache,
        cache_dir=cache_dir,
    )
