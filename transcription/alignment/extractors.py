# -*- coding: utf-8 -*-
"""
extractors.py

Concrete ASREmissionsExtractor implementations and audio chunk preparation helper.
"""

import hashlib
import json
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)
import numpy as np
from pydub import AudioSegment

from transcription.alignment.models import TokenEmission
from transcription.models.asr_model import CherokeeASRModel, WordConfidence
from transcription.audio.segment import segment_long_audio, AudioChunk


def prepare_audio_chunks(audio_input: Any, skip_vad: bool = False) -> List[AudioChunk]:
    """
    Prepares audio chunks from various audio input types (path str, AudioSegment, ndarray),
    either bypassing VAD (skip_vad=True) or using segment_long_audio VAD segmentation (skip_vad=False).
    """
    if skip_vad:
        if isinstance(audio_input, str):
            audio_seg = AudioSegment.from_file(audio_input)
        elif isinstance(audio_input, AudioSegment):
            audio_seg = audio_input
        elif isinstance(audio_input, np.ndarray):
            samples = audio_input
            if samples.dtype in (np.float32, np.float64):
                samples = (samples * 32767).astype(np.int16)
            audio_seg = AudioSegment(
                samples.tobytes(),
                frame_rate=16000,
                sample_width=2,
                channels=1,
            )
        else:
            audio_seg = audio_input

        if isinstance(audio_seg, AudioSegment):
            return [
                AudioChunk(
                    chunk_index=0,
                    audio=audio_seg,
                    start_sec=0.0,
                    end_sec=round(len(audio_seg) / 1000.0, 3),
                )
            ]
        return []
    else:
        return segment_long_audio(audio_input)


@runtime_checkable
class ASREmissionsExtractor(Protocol):
    """Protocol / base interface for extracting token emissions from audio input."""

    def extract(self, audio_input: Any = None) -> List[TokenEmission]:
        """Extracts token emissions with timestamps from audio input."""
        ...


class CherokeeASRExtractor(ASREmissionsExtractor):
    """
    ASREmissionsExtractor adapter wrapping CherokeeASRModel.
    Handles audio file path, AudioSegment, or ndarray; uses VAD segmentation when skip_vad=False;
    extracts logits and word confidences via CherokeeASRModel; produces List[TokenEmission].
    """

    def __init__(self, model: CherokeeASRModel, skip_vad: bool = False):
        self.model = model
        self.skip_vad = skip_vad

    def extract(self, audio_input: Any = None) -> List[TokenEmission]:
        """Extracts token emissions with timestamps from audio input."""
        if audio_input is None:
            return []
        chunks = prepare_audio_chunks(audio_input, skip_vad=self.skip_vad)
        emissions: List[TokenEmission] = []

        for c in chunks:
            samples = np.array(c.audio.get_array_of_samples(), dtype=np.float32)
            if c.audio.channels > 1:
                samples = samples.reshape((-1, c.audio.channels)).mean(axis=1)
            max_val = float(1 << (8 * c.audio.sample_width - 1))
            samples = samples / max_val

            logits = self.model.get_logits(samples, sample_rate=c.audio.frame_rate)
            word_confidences = self.model.get_word_confidences(logits)

            for w in word_confidences:
                if isinstance(w, WordConfidence):
                    word_str = w.word
                    start_t = w.start_time
                    end_t = w.end_time
                    conf = w.confidence
                elif isinstance(w, dict):
                    word_str = w.get("word", "")
                    start_t = w.get("start_time", w.get("start_sec", 0.0))
                    end_t = w.get("end_time", w.get("end_sec", 0.0))
                    conf = w.get("confidence", 1.0)
                else:
                    word_str = getattr(w, "word", "")
                    start_t = getattr(w, "start_time", 0.0)
                    end_t = getattr(w, "end_time", 0.0)
                    conf = getattr(w, "confidence", 1.0)

                emissions.append(
                    TokenEmission(
                        word=word_str,
                        start_sec=round(c.start_sec + start_t, 3),
                        end_sec=round(c.start_sec + end_t, 3),
                        confidence=conf,
                    )
                )

        return emissions


class CallbackEmissionsExtractor(ASREmissionsExtractor):
    """
    ASREmissionsExtractor adapter wrapping a callable/callback function.
    """

    def __init__(self, callback: Callable[..., Any], skip_vad: bool = False):
        self.callback = callback
        self.skip_vad = skip_vad

    def extract(self, audio_input: Any = None) -> List[TokenEmission]:
        """Extracts token emissions with timestamps from audio input using callback."""
        if audio_input is None:
            return []
        chunks = prepare_audio_chunks(audio_input, skip_vad=self.skip_vad)
        emissions: List[TokenEmission] = []

        for c in chunks:
            samples = np.array(c.audio.get_array_of_samples(), dtype=np.float32)
            if c.audio.channels > 1:
                samples = samples.reshape((-1, c.audio.channels)).mean(axis=1)
            max_val = float(1 << (8 * c.audio.sample_width - 1))
            samples = samples / max_val

            chunk_words = self.callback(samples, c.audio.frame_rate)

            for w in chunk_words:
                if isinstance(w, TokenEmission):
                    emissions.append(
                        TokenEmission(
                            word=w.word,
                            start_sec=round(c.start_sec + w.start_sec, 3),
                            end_sec=round(c.start_sec + w.end_sec, 3),
                            confidence=w.confidence,
                        )
                    )
                elif isinstance(w, dict):
                    emissions.append(
                        TokenEmission(
                            word=w.get("word", ""),
                            start_sec=round(
                                c.start_sec
                                + w.get("start_time", w.get("start_sec", 0.0)),
                                3,
                            ),
                            end_sec=round(
                                c.start_sec + w.get("end_time", w.get("end_sec", 0.0)),
                                3,
                            ),
                            confidence=w.get("confidence", 1.0),
                        )
                    )
                else:
                    emissions.append(
                        TokenEmission(
                            word=getattr(w, "word", ""),
                            start_sec=round(
                                c.start_sec
                                + getattr(
                                    w, "start_time", getattr(w, "start_sec", 0.0)
                                ),
                                3,
                            ),
                            end_sec=round(
                                c.start_sec
                                + getattr(w, "end_time", getattr(w, "end_sec", 0.0)),
                                3,
                            ),
                            confidence=getattr(w, "confidence", 1.0),
                        )
                    )

        return emissions


class PrecomputedEmissionsExtractor(ASREmissionsExtractor):
    """
    ASREmissionsExtractor adapter for pre-computed token emissions.
    """

    def __init__(
        self,
        token_emissions: Optional[
            Sequence[Union[TokenEmission, Dict[str, Any]]]
        ] = None,
    ):
        parsed: List[TokenEmission] = []
        for t in token_emissions or []:
            if isinstance(t, TokenEmission):
                parsed.append(t)
            elif isinstance(t, dict):
                parsed.append(
                    TokenEmission(
                        word=t.get("word", ""),
                        start_sec=t.get("start_time", t.get("start_sec", 0.0)),
                        end_sec=t.get("end_time", t.get("end_sec", 0.0)),
                        confidence=t.get("confidence", 1.0),
                    )
                )
            else:
                parsed.append(
                    TokenEmission(
                        word=getattr(t, "word", ""),
                        start_sec=getattr(
                            t, "start_time", getattr(t, "start_sec", 0.0)
                        ),
                        end_sec=getattr(t, "end_time", getattr(t, "end_sec", 0.0)),
                        confidence=getattr(t, "confidence", 1.0),
                    )
                )
        self.token_emissions = parsed

    def extract(self, audio_input: Any = None) -> List[TokenEmission]:
        """Returns precomputed token emissions."""
        return list(self.token_emissions)


class CachedASREmissionsExtractor(ASREmissionsExtractor):
    """
    Caching ASREmissionsExtractor wrapper that persists TokenEmission lists to disk.
    Bypasses wrapped ASR model inference on cache hits using deterministic cache keys.
    """

    def __init__(
        self,
        extractor: ASREmissionsExtractor,
        cache_dir: Optional[Union[str, Path]] = None,
        cache_key_prefix: Optional[str] = None,
    ):
        self.extractor = extractor
        self.cache_dir = (
            Path(cache_dir) if cache_dir is not None else Path(".cache/emissions")
        )
        self.cache_key_prefix = cache_key_prefix

    def _resolve_prefix(self) -> str:
        if self.cache_key_prefix is not None:
            return str(self.cache_key_prefix)
        extractor = self.extractor
        if hasattr(extractor, "model"):
            model = getattr(extractor, "model")
            if hasattr(model, "model_name") and model.model_name:
                return str(model.model_name)
            name_attr = getattr(model, "name", None)
            if name_attr:
                return str(name_attr)
        ext_name = getattr(extractor, "name", None)
        if ext_name:
            return str(ext_name)
        return extractor.__class__.__name__

    def _get_cache_key(self, audio_input: Any) -> Optional[str]:
        if audio_input is None:
            return None

        prefix = self._resolve_prefix()

        if isinstance(audio_input, (str, Path)):
            p = Path(audio_input)
            if p.exists() and p.is_file():
                try:
                    stat = p.stat()
                    raw_id = f"{prefix}|{p.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"
                except OSError:
                    raw_id = f"{prefix}|{p.resolve()}"
            else:
                raw_id = f"{prefix}|{str(audio_input)}"
            return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()

        if isinstance(audio_input, AudioSegment):
            hasher = hashlib.sha256()
            hasher.update(prefix.encode("utf-8"))
            hasher.update(
                f"|channels:{audio_input.channels}|frame_rate:{audio_input.frame_rate}|sample_width:{audio_input.sample_width}|".encode(
                    "utf-8"
                )
            )
            raw = getattr(audio_input, "raw_data", None)
            if isinstance(raw, (bytes, bytearray, memoryview)):
                hasher.update(raw)
            return hasher.hexdigest()

        if isinstance(audio_input, np.ndarray):
            hasher = hashlib.sha256()
            hasher.update(prefix.encode("utf-8"))
            hasher.update(
                f"|dtype:{audio_input.dtype}|shape:{audio_input.shape}|".encode("utf-8")
            )
            hasher.update(audio_input.tobytes())
            return hasher.hexdigest()

        if isinstance(audio_input, (bytes, bytearray)):
            hasher = hashlib.sha256()
            hasher.update(prefix.encode("utf-8"))
            hasher.update(b"|bytes|")
            hasher.update(bytes(audio_input))
            return hasher.hexdigest()

        raw_id = f"{prefix}|{repr(audio_input)}"
        return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def _load_from_cache(self, cache_path: Path) -> Optional[List[TokenEmission]]:
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                return None
            return [
                TokenEmission(
                    word=str(item["word"]),
                    start_sec=float(item["start_sec"]),
                    end_sec=float(item["end_sec"]),
                    confidence=float(item.get("confidence", 1.0)),
                )
                for item in data
                if isinstance(item, dict)
                and "word" in item
                and "start_sec" in item
                and "end_sec" in item
            ]
        except Exception:
            return None

    def _save_to_cache(self, cache_path: Path, emissions: List[TokenEmission]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "word": t.word,
                "start_sec": t.start_sec,
                "end_sec": t.end_sec,
                "confidence": t.confidence,
            }
            for t in emissions
        ]
        temp_path = cache_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        temp_path.replace(cache_path)

    def extract(self, audio_input: Any = None) -> List[TokenEmission]:
        """Extracts token emissions with timestamps from audio input, utilizing disk caching."""
        if audio_input is None:
            return []

        cache_key = self._get_cache_key(audio_input)
        if cache_key is not None:
            cache_path = self._get_cache_path(cache_key)
            cached_emissions = self._load_from_cache(cache_path)
            if cached_emissions is not None:
                return cached_emissions

        emissions = self.extractor.extract(audio_input)
        if cache_key is not None:
            cache_path = self._get_cache_path(cache_key)
            self._save_to_cache(cache_path, emissions)

        return emissions
