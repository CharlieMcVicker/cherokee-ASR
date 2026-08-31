# -*- coding: utf-8 -*-
"""
extractors.py

Concrete ASREmissionsExtractor implementations.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
from pydub import AudioSegment

from transcription.alignment.domain.models import TokenEmission
from transcription.alignment.ports.protocols import ASREmissionsExtractor
from transcription.models.asr_model import CherokeeASRModel, WordConfidence
from transcription.audio.segment import segment_long_audio, AudioChunk


class CherokeeASRExtractor(ASREmissionsExtractor):
    """
    ASREmissionsExtractor adapter wrapping CherokeeASRModel.
    Handles audio file path, AudioSegment, or ndarray; uses VAD segmentation when skip_vad=False;
    extracts logits and word confidences via CherokeeASRModel; produces List[TokenEmission].
    """

    def __init__(self, model: CherokeeASRModel, skip_vad: bool = False):
        self.model = model
        self.skip_vad = skip_vad

    def _prepare_chunks(self, audio_input: Any) -> List[AudioChunk]:
        if self.skip_vad:
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

    def extract(self, audio_input: Any) -> List[TokenEmission]:
        """Extracts token emissions with timestamps from audio input."""
        chunks = self._prepare_chunks(audio_input)
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

    def _prepare_chunks(self, audio_input: Any) -> List[AudioChunk]:
        if self.skip_vad:
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

    def extract(self, audio_input: Any) -> List[TokenEmission]:
        """Extracts token emissions with timestamps from audio input using callback."""
        chunks = self._prepare_chunks(audio_input)
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
