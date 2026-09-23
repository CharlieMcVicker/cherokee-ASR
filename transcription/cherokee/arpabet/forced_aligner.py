# -*- coding: utf-8 -*-
"""
transcription.cherokee.arpabet.forced_aligner

Word-level forced alignment for English audio using torchaudio's MMS_FA pipeline.
Aligns speech audio waveforms against sentence transcripts to produce precise
start and end timestamps for each word.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Any, List, Optional, Protocol, Sequence, Tuple, Union
import torch
import torchaudio

logger = logging.getLogger(__name__)

# Regex pattern for cleaning words for MMS_FA dictionary
_CLEAN_WORD_RE = re.compile(r"[^a-zA-Z'\-]")


@dataclass(frozen=True)
class AlignedWordSpan:
    """
    Immutable representation of an aligned word boundary in an audio recording.
    """

    word: str
    start_sec: float
    end_sec: float
    duration: float
    score: float
    token_count: int = 0

    def __post_init__(self) -> None:
        if self.end_sec < self.start_sec:
            object.__setattr__(self, "end_sec", self.start_sec)
        if self.duration < 0.0:
            object.__setattr__(
                self, "duration", max(0.0, self.end_sec - self.start_sec)
            )


class ForcedAlignerProtocol(Protocol):
    """Protocol for word-level forced alignment."""

    def align(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        words: Union[str, Sequence[str]],
    ) -> List[AlignedWordSpan]: ...


class MMSForcedAligner:
    """
    Torchaudio MMS_FA forced aligner for extracting word boundaries from audio.
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        bundle: Any = None,
    ) -> None:
        if bundle is None:
            bundle = torchaudio.pipelines.MMS_FA
        self.bundle = bundle
        self.target_sample_rate: int = bundle.sample_rate

        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self._model = self.bundle.get_model().to(self.device)
        self._model.eval()
        self._cpu_model: Optional[Any] = None
        self._tokenizer = self.bundle.get_tokenizer()
        self._aligner = self.bundle.get_aligner()
        self._dictionary = self.bundle.get_dict()

    def _get_cpu_model(self) -> Any:
        """Lazily initialize or return CPU model for fallback and long waveforms."""
        if self._cpu_model is None:
            if self.device.type == "cpu":
                model = self._model
            else:
                model = self.bundle.get_model().cpu()
                model.eval()
            self._cpu_model = model
        return self._cpu_model

    def clean_word(self, word: str) -> str:
        """
        Normalize and clean a word to characters supported by MMS_FA dictionary.
        """
        return _CLEAN_WORD_RE.sub("", word).strip().lower()

    def prepare_transcript(
        self, words: Union[str, Sequence[str]]
    ) -> Tuple[List[str], List[str]]:
        """
        Prepare transcript words, returning (original_words, cleaned_words).
        Only words with non-empty cleaned representations are kept.
        """
        if isinstance(words, str):
            raw_words = words.strip().split()
        else:
            raw_words = list(words)

        orig_kept: List[str] = []
        clean_kept: List[str] = []

        for w in raw_words:
            w_str = str(w).strip()
            clean = self.clean_word(w_str)
            if clean:
                orig_kept.append(w_str)
                clean_kept.append(clean)

        return orig_kept, clean_kept

    def align(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        words: Union[str, Sequence[str]],
    ) -> List[AlignedWordSpan]:
        """
        Extract word boundaries from waveform given transcript words.

        Args:
            waveform: Audio tensor of shape [channels, samples] or [samples].
            sample_rate: Sample rate of the input waveform in Hz.
            words: Space-delimited string or sequence of word strings.

        Returns:
            List of AlignedWordSpan instances.
        """
        orig_words, clean_words = self.prepare_transcript(words)
        if not clean_words:
            return []

        # Ensure 2D tensor [channels, samples]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim > 2:
            waveform = waveform.view(waveform.shape[0], -1)

        # Convert to mono if multi-channel
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.target_sample_rate
            )
            waveform = resampler(waveform)

        num_samples = waveform.shape[-1]
        if num_samples == 0:
            return []

        total_duration_sec = num_samples / self.target_sample_rate

        # Tokenize cleaned words
        try:
            tokens = self._tokenizer(clean_words)
        except Exception as e:
            logger.warning("Tokenizer error on words %s: %s", clean_words, e)
            return []

        total_tokens = sum(len(t) for t in tokens)
        if total_tokens == 0:
            return []

        # CTC forced aligner requires sufficient frames for tokens (approx 20ms per frame)
        # If the audio is too short, forced aligner will raise a RuntimeError
        estimated_frames = num_samples // 320
        if estimated_frames < total_tokens:
            logger.warning(
                "Audio too short (%d frames) for %d CTC tokens, skipping",
                estimated_frames,
                total_tokens,
            )
            return []

        # Forward pass on target device (MPS has convolution limits for waveforms > 15s)
        try:
            if self.device.type == "mps" and num_samples > 250000:
                cpu_model = self._get_cpu_model()
                with torch.inference_mode():
                    emissions, _ = cpu_model(waveform.cpu())
            else:
                with torch.inference_mode():
                    emissions, _ = self._model(waveform.to(self.device))
        except Exception as e:
            # Fallback to CPU if device hits backend execution limit
            try:
                cpu_model = self._get_cpu_model()
                with torch.inference_mode():
                    emissions, _ = cpu_model(waveform.cpu())
            except Exception as cpu_e:
                logger.warning("Model forward pass error: %s", cpu_e)
                return []

        # MMS aligner runs on CPU
        emissions_cpu = emissions[0].cpu()
        num_frames = emissions_cpu.shape[0]
        if num_frames == 0:
            return []

        try:
            token_spans = self._aligner(emissions_cpu, tokens)
        except RuntimeError as e:
            # Handles CTC alignment failures gracefully (e.g. target too long or acoustic mismatch)
            logger.debug("Forced alignment CTC error: %s", e)
            return []
        except Exception as e:
            logger.warning("Unexpected alignment error: %s", e)
            return []

        seconds_per_frame = total_duration_sec / num_frames
        aligned_spans: List[AlignedWordSpan] = []

        for i, spans in enumerate(token_spans):
            if not spans:
                continue

            orig_w = orig_words[i]
            start_frame = spans[0].start
            end_frame = spans[-1].end

            start_sec = round(
                max(0.0, min(total_duration_sec, start_frame * seconds_per_frame)), 4
            )
            end_sec = round(
                max(start_sec, min(total_duration_sec, end_frame * seconds_per_frame)),
                4,
            )
            duration = round(end_sec - start_sec, 4)

            scores = [float(s.score) for s in spans]
            mean_score = sum(scores) / len(scores) if scores else 0.0

            aligned_spans.append(
                AlignedWordSpan(
                    word=orig_w,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    duration=duration,
                    score=round(mean_score, 4),
                    token_count=len(spans),
                )
            )

        return aligned_spans


_DEFAULT_ALIGNER: Optional[MMSForcedAligner] = None


def get_default_forced_aligner() -> MMSForcedAligner:
    """Return or lazily initialize the default global MMSForcedAligner instance."""
    global _DEFAULT_ALIGNER
    if _DEFAULT_ALIGNER is None:
        _DEFAULT_ALIGNER = MMSForcedAligner()
    return _DEFAULT_ALIGNER
