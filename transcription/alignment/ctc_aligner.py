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
    TextChunk,
    TokenEmission,
    WordInterval,
)
from transcription.alignment.normalizers import (
    normalize_phonetics_for_alignment,
    normalize_syllabary_for_alignment,
)
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
        syncope_tokens: Sequence[str] = DEFAULT_SYNCOPE_TOKENS,
        syncope_penalty: float = 2.0,
        index_duration: float = 0.02,
        chunk_normalizer: Optional[Callable[[str], str]] = None,
        min_window_size: int = 8000,
        max_window_size: int = 100000,
        buffer_trail_ms: int = 300,
        buffer_lead_ms: int = 100,
        intrusive_tokens: Sequence[str] = ("h", "'"),
        intrusive_penalty: float = 0.1,
        flag_min_confidence: float = 0.01,
        flag_min_char_duration_sec: float = 0.03,
        cache: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        self.model = model
        self.syncope_tokens = list(syncope_tokens)
        self.syncope_penalty = float(syncope_penalty)
        self.intrusive_tokens = list(intrusive_tokens) if intrusive_tokens else []
        self.intrusive_penalty = float(intrusive_penalty)
        self.flag_min_confidence = float(flag_min_confidence)
        self.flag_min_char_dur = float(flag_min_char_duration_sec)
        self.index_duration = float(index_duration)
        self.chunk_norm = chunk_normalizer or normalize_phonetics_for_alignment
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.buffer_trail_ms = buffer_trail_ms
        self.buffer_lead_ms = buffer_lead_ms
        self.cache = cache
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR

    def _get_char_list_and_blank(
        self, asr_model: CherokeeASRModel
    ) -> Tuple[List[str], int]:
        vocab = asr_model.processor.tokenizer.get_vocab()
        char_list = [token for token, idx in sorted(vocab.items(), key=lambda x: x[1])]
        pad_id = getattr(asr_model.processor.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = vocab.get("[PAD]", 18)
        return char_list, pad_id

    def extract_logits(
        self,
        audio_input: Union[str, Path, AudioSegment, np.ndarray],
        asr_model: Optional[CherokeeASRModel] = None,
        chunk_seconds: float = 30.0,
        apply_buffers: bool = True,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Extract log-probability matrix (T, V) from audio input using ASR model.
        Uses batched chunk processing for memory efficiency and speed on long files.
        Returns: (lpz, raw_audio_duration_sec, lead_offset_sec)
        """
        model = asr_model or self.model
        if model is None:
            raise ValueError("ASR model must be provided to extract logits.")

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

        chunk_len = int(16000 * chunk_seconds)
        if len(samples) <= chunk_len:
            logits = model.get_logits(samples, sample_rate=16000)
            if logits.ndim == 3:
                logits = logits.squeeze(0)
            lpz = torch.nn.functional.log_softmax(logits, dim=-1).cpu().numpy()
        else:
            audio_chunks = [
                samples[i : i + chunk_len] for i in range(0, len(samples), chunk_len)
            ]
            chunk_logits = [model.get_logits(c) for c in audio_chunks]
            all_logits = torch.cat(chunk_logits, dim=0)
            lpz = torch.nn.functional.log_softmax(all_logits, dim=-1).cpu().numpy()

        return lpz, total_audio_sec, lead_offset_sec

    def _compute_cache_key(
        self,
        audio_input: Union[str, Path, AudioSegment, np.ndarray],
        asr_model: Optional[CherokeeASRModel] = None,
        chunk_seconds: float = 30.0,
        apply_buffers: bool = True,
    ) -> str:
        model = asr_model or self.model
        model_name = getattr(model, "model_name", None) or "cherokee_asr"

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
        buf_str = f"lead={lead}:trail={trail}:chunk={chunk_seconds}"
        full_key = f"{model_name}|{input_id}|{buf_str}"
        cache_hash = hashlib.sha256(full_key.encode("utf-8")).hexdigest()
        return f"{stem}_{cache_hash[:16]}"

    def get_logits_cached(
        self,
        audio_input: Union[str, Path, AudioSegment, np.ndarray],
        asr_model: Optional[CherokeeASRModel] = None,
        chunk_seconds: float = 30.0,
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
        chunk_seconds : float
            Chunk length for batched audio processing.
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

        if not use_cache:
            return self.extract_logits(
                audio_input=audio_input,
                asr_model=asr_model,
                chunk_seconds=chunk_seconds,
                apply_buffers=apply_buffers,
            )

        target_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = self._compute_cache_key(
            audio_input=audio_input,
            asr_model=asr_model,
            chunk_seconds=chunk_seconds,
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
            chunk_seconds=chunk_seconds,
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

        norm_text = self.chunk_norm(phonetic_text)
        words = norm_text.split()

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
            syncope_penalty=self.syncope_penalty,
            intrusive_tokens=self.intrusive_tokens,
            intrusive_penalty=self.intrusive_penalty,
            index_duration=self.index_duration,
            score_min_mean_over_L=2,
            replace_spaces_with_blanks=False,
        )

        gt_mat, utt_indices = prepare_text(config, words, char_list)
        timings, char_probs, state_list = ctc_segmentation(config, lpz, gt_mat)

        word_intervals: List[WordInterval] = []
        prev_end = 0.0

        for w_idx, raw_w in enumerate(words):
            start_idx = utt_indices[w_idx]
            end_idx = utt_indices[w_idx + 1]
            w_timings = [t for t in timings[start_idx:end_idx] if t > 0.0]

            if w_timings:
                raw_w_start = min(w_timings)
                raw_w_end = max(w_timings) + self.index_duration
                w_start = max(0.0, round(raw_w_start - lead_offset_sec, 3))
                w_end = min(dur_sec, round(raw_w_end - lead_offset_sec, 3))
            else:
                w_start = prev_end
                w_end = prev_end

            # Extract emitted word tokens from state_list across raw slice
            start_f = (
                int(round(min(w_timings) / self.index_duration)) if w_timings else 0
            )
            end_f = (
                int(round((max(w_timings) + self.index_duration) / self.index_duration))
                if w_timings
                else 0
            )
            emitted_chars = [
                s
                for s in state_list[start_f : max(start_f + 1, end_f)]
                if s and s != "ε" and s != "[PAD]"
            ]
            emitted_w = "".join(emitted_chars) or raw_w

            w_dur = max(0.0, w_end - w_start)
            # Word confidence from acoustic character state log-probabilities
            char_state_lps = [
                char_probs[f]
                for f in range(start_f, max(start_f + 1, end_f))
                if state_list[f] and state_list[f] not in ("ε", "[PAD]")
            ]
            if w_timings and char_state_lps:
                mean_logprob = float(np.mean(char_state_lps))
                word_conf = float(np.exp(mean_logprob))
            else:
                word_conf = 0.0

            is_low_conf = bool(word_conf < self.flag_min_confidence)
            is_unaligned = bool(len(w_timings) == 0 or len(emitted_chars) == 0)
            is_flagged = bool(is_low_conf or is_unaligned)

            word_intervals.append(
                WordInterval(
                    word=raw_w,
                    start_sec=w_start,
                    end_sec=w_end,
                    confidence=round(word_conf, 6),
                    flagged=is_flagged,
                    emitted_word=emitted_w,
                )
            )
            prev_end = w_end

        chunk_start = word_intervals[0].start_sec if word_intervals else 0.0
        chunk_end = word_intervals[-1].end_sec if word_intervals else dur_sec

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

        lpz, dur_sec, _ = self.get_logits_cached(
            audio_input, asr_model=model, apply_buffers=False, cache=cache
        )
        char_list, pad_id = self._get_char_list_and_blank(model)

        config = CtcSegmentationParameters(
            char_list=char_list,
            blank=pad_id,
            syncope_tokens=self.syncope_tokens,
            syncope_penalty=self.syncope_penalty,
            intrusive_tokens=self.intrusive_tokens,
            intrusive_penalty=self.intrusive_penalty,
            index_duration=self.index_duration,
            min_window_size=max(self.min_window_size, lpz.shape[0]),
            max_window_size=max(self.max_window_size, lpz.shape[0] * 2),
            score_min_mean_over_L=2,
            replace_spaces_with_blanks=False,
        )

        texts = [self.chunk_norm(c.text).replace(" ", "|") for c in chunks]
        gt_mat, utt_indices = prepare_text(config, texts, char_list)

        timings, char_probs, state_list = ctc_segmentation(config, lpz, gt_mat)
        raw_segments = determine_utterance_segments(
            config, utt_indices, char_probs, timings, texts
        )

        aligned_chunks: List[AlignedChunk] = []

        for c_idx, chunk in enumerate(chunks):
            start_utt_idx = utt_indices[c_idx]
            end_utt_idx = utt_indices[c_idx + 1]
            chunk_timings = [t for t in timings[start_utt_idx:end_utt_idx] if t > 0.0]

            if chunk_timings:
                c_start = round(min(chunk_timings), 3)
                c_end = round(max(chunk_timings) + self.index_duration, 3)
            else:
                raw_s, raw_e, _ = raw_segments[c_idx]
                c_start = (
                    round(float(raw_s), 3)
                    if float(raw_s) > 0.0
                    else (aligned_chunks[-1].end_sec if aligned_chunks else 0.0)
                )
                c_end = round(float(raw_e), 3) if float(raw_e) > c_start else c_start

            # Extract word intervals within chunk
            words = self.chunk_norm(chunk.text).split()
            word_intervals: List[WordInterval] = []
            if words:
                w_start_f = int(round(c_start / self.index_duration))
                w_end_f = int(round(c_end / self.index_duration))
                frames_per_word = max(1, (w_end_f - w_start_f) // len(words))

                for w_i, raw_w in enumerate(words):
                    ws = round(c_start + w_i * frames_per_word * self.index_duration, 3)
                    we = round(
                        min(
                            c_end,
                            c_start + (w_i + 1) * frames_per_word * self.index_duration,
                        ),
                        3,
                    )
                    word_intervals.append(
                        WordInterval(
                            word=raw_w,
                            start_sec=ws,
                            end_sec=we,
                            confidence=1.0,
                            flagged=False,
                            emitted_word=raw_w,
                        )
                    )

            # Emitted text in frame range from backtracked state_list
            start_frame = int(round(c_start / self.index_duration))
            end_frame = int(round(c_end / self.index_duration))
            emitted_chars = [
                s
                for s in state_list[start_frame:end_frame]
                if s and s != "ε" and s != "[PAD]"
            ]
            emitted_text = " ".join("".join(emitted_chars).replace("|", " ").split())

            aligned_chunks.append(
                AlignedChunk(
                    chunk_id=chunk.chunk_id,
                    start_sec=c_start,
                    end_sec=c_end,
                    words=word_intervals,
                    distance_score=round(float(raw_segments[c_idx][2]), 4),
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
        cache=cache,
        cache_dir=cache_dir,
    )
    return aligner.get_logits_cached(
        audio_input=audio_input,
        asr_model=asr_model,
        chunk_seconds=chunk_seconds,
        apply_buffers=apply_buffers,
        cache=cache,
        cache_dir=cache_dir,
    )
