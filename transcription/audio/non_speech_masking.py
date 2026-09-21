# -*- coding: utf-8 -*-
"""
non_speech_masking.py

Silero VAD-based continuous speech probability extraction and soft-masking
of acoustic log-probabilities (lpz) for CTC segmentation alignment.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Union
import urllib.request

import numpy as np
from numpy.typing import NDArray
from pydub import AudioSegment
import torch

logger = logging.getLogger(__name__)

SILERO_VAD_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.jit"
DEFAULT_VAD_CACHE_PATH = Path.home() / ".cache" / "silero_vad.jit"


class SileroVADDetector:
    """
    Singleton / encapsulated runner for Silero VAD TorchScript JIT model.
    """

    _instance: Optional["SileroVADDetector"] = None

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.model_path = Path(model_path) if model_path else DEFAULT_VAD_CACHE_PATH
        self.model = self._load_model()

    def _load_model(self) -> torch.jit.ScriptModule:
        if not self.model_path.exists():
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading Silero VAD JIT model to {self.model_path}...")
            urllib.request.urlretrieve(SILERO_VAD_URL, str(self.model_path))
            logger.info("Silero VAD downloaded successfully.")

        model = torch.jit.load(str(self.model_path), map_location="cpu")
        model.eval()
        return model

    @classmethod
    def get_default_detector(
        cls, model_path: Optional[Union[str, Path]] = None
    ) -> "SileroVADDetector":
        if cls._instance is None:
            cls._instance = cls(model_path=model_path)
        return cls._instance

    def predict_speech_probabilities(
        self,
        audio_tensor_16k: torch.Tensor,
        window_size_samples: int = 512,
    ) -> NDArray[np.float32]:
        """
        Runs Silero VAD sequentially across 16kHz mono audio tensor.
        Returns 1D numpy array of speech probabilities per 512-sample (32ms) window.
        """
        if audio_tensor_16k.dim() > 1:
            audio_tensor_16k = audio_tensor_16k.squeeze()
        if audio_tensor_16k.dim() != 1:
            raise ValueError(
                f"Audio tensor must be 1D mono, got shape {audio_tensor_16k.shape}"
            )

        num_samples = audio_tensor_16k.shape[0]
        if num_samples == 0:
            return np.array([], dtype=np.float32)

        probs: list[float] = []
        # Reset internal RNN state for clean sequence evaluation
        if hasattr(self.model, "reset_states"):
            self.model.reset_states()

        with torch.no_grad():
            for i in range(0, num_samples, window_size_samples):
                chunk = audio_tensor_16k[i : i + window_size_samples]
                if len(chunk) < window_size_samples:
                    chunk = torch.nn.functional.pad(
                        chunk, (0, window_size_samples - len(chunk))
                    )
                prob = float(self.model(chunk, 16000))
                probs.append(prob)

        return np.asarray(probs, dtype=np.float32)


def _load_audio_as_16k_tensor(
    audio: Union[str, Path, AudioSegment, np.ndarray, torch.Tensor],
    sample_rate: int = 16000,
) -> torch.Tensor:
    """
    Standardizes input audio into a 1D float32 PyTorch tensor sampled at 16kHz.
    """
    if isinstance(audio, (str, Path)):
        seg = AudioSegment.from_file(str(audio))
        seg = seg.set_frame_rate(sample_rate).set_channels(1)
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0
        return torch.from_numpy(samples)
    elif isinstance(audio, AudioSegment):
        seg = audio.set_frame_rate(sample_rate).set_channels(1)
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0
        return torch.from_numpy(samples)
    elif isinstance(audio, np.ndarray):
        arr = audio.astype(np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=-1)
        if np.max(np.abs(arr)) > 1.0:
            arr = arr / 32768.0
        return torch.from_numpy(arr)
    elif isinstance(audio, torch.Tensor):
        t = audio.float()
        if t.dim() > 1:
            t = t.mean(dim=-1)
        if torch.max(torch.abs(t)) > 1.0:
            t = t / 32768.0
        return t
    else:
        raise TypeError(f"Unsupported audio type: {type(audio)}")


def mask_non_speech_logits(
    lpz: NDArray[np.float32],
    audio: Union[str, Path, AudioSegment, np.ndarray, torch.Tensor],
    sample_rate: int = 16000,
    index_duration: float = 0.02,
    blank_id: int = 0,
    p_low: float = 0.15,
    p_high: float = 0.60,
    pad_ms: int = 60,
    detector: Optional[SileroVADDetector] = None,
) -> NDArray[np.float32]:
    """
    Applies continuous soft-masking to acoustic log-probabilities (lpz) based on
    Silero VAD speech probabilities.

    Mathematical Formulation:
      gamma(p_t) = clip((p_t - p_low) / (p_high - p_low), 0.0, 1.0)
      P_tilde[t] = gamma(p_t) * P[t] + (1 - gamma(p_t)) * e_blank
      lpz_masked[t] = log(P_tilde[t] + eps)

    Args:
        lpz: (T, V) array of acoustic log-probabilities.
        audio: Audio input (file path, AudioSegment, ndarray, or tensor).
        sample_rate: Audio sample rate (default 16000).
        index_duration: Frame stride duration in seconds (default 0.02s / 20ms).
        blank_id: Index of the CTC blank token in the vocabulary (default 0).
        p_low: Speech probability lower bound below which frames become 100% blank.
        p_high: Speech probability upper bound above which frames remain 100% untouched.
        pad_ms: Temporal padding in milliseconds to dilate speech regions.
        detector: Optional SileroVADDetector instance.

    Returns:
        Conditioned (T, V) array of log-probabilities of identical shape and dtype.
    """
    T, V = lpz.shape
    if T == 0:
        return lpz.copy()

    vad_engine = detector or SileroVADDetector.get_default_detector()
    audio_tensor = _load_audio_as_16k_tensor(audio, sample_rate=sample_rate)

    # 512 samples at 16kHz = 32ms window
    window_samples = 512
    window_sec = window_samples / float(sample_rate)
    window_probs = vad_engine.predict_speech_probabilities(
        audio_tensor, window_size_samples=window_samples
    )

    if len(window_probs) == 0:
        return lpz.copy()

    # Time centers for VAD windows
    vad_times = (np.arange(len(window_probs)) + 0.5) * window_sec

    # Time centers for ASR frames (T)
    asr_times = (np.arange(T) + 0.5) * index_duration

    # Interpolate VAD probabilities onto exact ASR frame grid
    p_speech = np.interp(
        asr_times, vad_times, window_probs, left=0.0, right=0.0
    ).astype(np.float32)

    # Apply speech dilation / padding if pad_ms > 0
    if pad_ms > 0:
        pad_frames = int(round((pad_ms / 1000.0) / index_duration))
        if pad_frames > 0:
            # 1D max filter / dilation over moving window
            dilated = np.copy(p_speech)
            for shift in range(1, pad_frames + 1):
                dilated = np.maximum(
                    dilated, np.pad(p_speech[shift:], (0, shift), mode="edge")
                )
                dilated = np.maximum(
                    dilated, np.pad(p_speech[:-shift], (shift, 0), mode="edge")
                )
            p_speech = dilated

    # Continuous soft gate gamma in [0.0, 1.0]
    denom = max(1e-5, p_high - p_low)
    gamma = np.clip((p_speech - p_low) / denom, 0.0, 1.0)[:, None]

    # Convex posterior blending in probability space
    # P[t] = exp(lpz[t])
    # P_tilde[t] = gamma * P[t] + (1 - gamma) * e_blank
    P = np.exp(lpz)
    e_blank = np.zeros((1, V), dtype=np.float32)
    e_blank[0, blank_id] = 1.0

    P_tilde = gamma * P + (1.0 - gamma) * e_blank
    lpz_masked = np.log(np.maximum(P_tilde, 1e-12)).astype(lpz.dtype)

    return lpz_masked


def extract_vad_intervals(
    audio: Union[str, Path, AudioSegment, np.ndarray, torch.Tensor],
    sample_rate: int = 16000,
    p_low: float = 0.15,
    p_high: float = 0.60,
    pad_ms: int = 60,
    detector: Optional[SileroVADDetector] = None,
) -> list[tuple[float, float, str]]:
    """
    Extracts contiguous VAD state intervals ('speech', 'mid', 'non-speech')
    spanning the full audio timeline.

    Returns:
        List of (start_sec, end_sec, label) tuples.
    """
    vad_engine = detector or SileroVADDetector.get_default_detector()
    audio_tensor = _load_audio_as_16k_tensor(audio, sample_rate=sample_rate)

    total_samples = audio_tensor.shape[0]
    if total_samples == 0:
        return []

    total_sec = round(total_samples / float(sample_rate), 3)

    window_samples = 512
    window_sec = window_samples / float(sample_rate)
    window_probs = vad_engine.predict_speech_probabilities(
        audio_tensor, window_size_samples=window_samples
    )

    if len(window_probs) == 0:
        return [(0.0, total_sec, "non-speech")]

    # Optional speech dilation
    p_speech = np.copy(window_probs)
    if pad_ms > 0:
        pad_steps = int(round((pad_ms / 1000.0) / window_sec))
        if pad_steps > 0:
            dilated = np.copy(p_speech)
            for shift in range(1, pad_steps + 1):
                dilated = np.maximum(
                    dilated, np.pad(p_speech[shift:], (0, shift), mode="edge")
                )
                dilated = np.maximum(
                    dilated, np.pad(p_speech[:-shift], (shift, 0), mode="edge")
                )
            p_speech = dilated

    # Classify windows
    labels: list[str] = []
    for p in p_speech:
        if p >= p_high:
            labels.append("speech")
        elif p <= p_low:
            labels.append("non-speech")
        else:
            labels.append("mid")

    # Merge contiguous identical labels
    intervals: list[tuple[float, float, str]] = []
    curr_label = labels[0]
    curr_start = 0.0

    for i in range(1, len(labels)):
        if labels[i] != curr_label:
            end_t = min(total_sec, round(i * window_sec, 3))
            intervals.append((curr_start, end_t, curr_label))
            curr_label = labels[i]
            curr_start = end_t

    intervals.append((curr_start, total_sec, curr_label))
    return intervals
