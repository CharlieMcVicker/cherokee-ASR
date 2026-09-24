# -*- coding: utf-8 -*-
"""
Audio perturbation transforms for ASR evaluation and robustness testing.

Provides composable audio transformations including:
- AudioTransform: Abstract base class for all audio transformations.
- AdditiveNoise: Additive noise perturbation supporting white, pink, and ambient profiles.
- TimeWarp: Time-stretching/warping with pitch preservation via phase vocoding.
- AcousticFilter: Butterworth bandpass acoustic filtering (e.g., telephone/telecom bandwidth).
- ComposeTransforms: Sequential pipeline of multiple audio transformations.
"""

from abc import ABC, abstractmethod
import math
import os
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Union

import numpy as np
import scipy.signal
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T

__all__ = [
    "AudioTransform",
    "AdditiveNoise",
    "TimeWarp",
    "AcousticFilter",
    "ComposeTransforms",
]


class AudioTransform(ABC):
    """
    Abstract base class for audio waveform transformations.

    All subclasses must implement __call__(waveform, sample_rate),
    accepting a 1D [samples] or 2D [channels, samples] torch.Tensor
    and returning a transformed tensor preserving channel dimensions.
    """

    @abstractmethod
    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Apply the transform to an input audio waveform.

        Args:
            waveform: Audio tensor of shape [channels, samples] or [samples].
            sample_rate: Sample rate in Hz.

        Returns:
            Transformed audio tensor with preserved channel dimensions.
        """
        pass


class AdditiveNoise(AudioTransform):
    """
    Applies additive noise at a specified Signal-to-Noise Ratio (SNR in dB).

    Supports:
    - "white": Uniform spectral density Gaussian noise.
    - "pink": 1/f power spectral density synthesis via 1/sqrt(f) FFT magnitude scaling.
    - "ambient": Real-world ambient environmental audio loaded from disk (or synthetic fallback).
    """

    def __init__(
        self,
        snr_db: float = 10.0,
        noise_type: Literal["white", "pink", "ambient"] = "white",
        ambient_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Initialize AdditiveNoise transform.

        Args:
            snr_db: Target Signal-to-Noise Ratio in decibels.
            noise_type: Type of noise to inject ('white', 'pink', 'ambient').
            ambient_path: Optional path to ambient audio file when noise_type is 'ambient'.
        """
        super().__init__()
        self.snr_db = float(snr_db)
        if noise_type not in ("white", "pink", "ambient"):
            raise ValueError(
                f"Unsupported noise_type '{noise_type}'. Must be one of 'white', 'pink', 'ambient'."
            )
        self.noise_type = noise_type
        self.ambient_path = Path(ambient_path) if ambient_path is not None else None

    def _generate_pink_noise(
        self,
        num_channels: int,
        num_samples: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Generate pink noise via spectral synthesis (1 / sqrt(f) FFT magnitude scaling).
        """
        # Generate white Gaussian noise in time domain
        w = torch.randn(num_channels, num_samples, device=device, dtype=dtype)
        # Compute 1D Real FFT along time dimension
        w_fft = torch.fft.rfft(w, n=num_samples, dim=-1)
        freq_bins = w_fft.shape[-1]
        k = torch.arange(freq_bins, device=device, dtype=dtype)
        # Magnitude scaling: 1 / sqrt(k + 1)
        scale = 1.0 / torch.sqrt(k + 1.0)
        pink_fft = w_fft * scale
        # Inverse FFT back to time domain
        pink = torch.fft.irfft(pink_fft, n=num_samples, dim=-1)
        # Normalize RMS to unit energy
        rms = torch.sqrt(torch.mean(pink**2, dim=-1, keepdim=True) + 1e-12)
        return pink / rms

    def _generate_low_frequency_noise(
        self,
        num_channels: int,
        num_samples: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Generate low-frequency synthetic ambient rumble fallback.
        """
        w = torch.randn(num_channels, num_samples, device=device, dtype=dtype)
        w_fft = torch.fft.rfft(w, n=num_samples, dim=-1)
        k = torch.arange(w_fft.shape[-1], device=device, dtype=dtype)
        scale = 1.0 / (k + 1.0)
        low_fft = w_fft * scale
        low_noise = torch.fft.irfft(low_fft, n=num_samples, dim=-1)
        rms = torch.sqrt(torch.mean(low_noise**2, dim=-1, keepdim=True) + 1e-12)
        return low_noise / rms

    def _load_ambient_audio(
        self,
        target_channels: int,
        target_samples: int,
        sample_rate: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Load ambient audio file, resample to target_rate, and tile/crop to match shape.
        """
        if self.ambient_path is not None and os.path.isfile(self.ambient_path):
            try:
                # Attempt loading with soundfile or torchaudio
                loaded_audio: Optional[torch.Tensor] = None
                orig_sr: int = sample_rate
                try:
                    data, sr = sf.read(str(self.ambient_path), dtype="float32")
                    tensor = torch.from_numpy(data)
                    if tensor.dim() == 1:
                        loaded_audio = tensor.unsqueeze(0)
                    else:
                        loaded_audio = tensor.t()
                    orig_sr = sr
                except Exception:
                    loaded_audio, orig_sr = torchaudio.load(str(self.ambient_path))

                if loaded_audio is not None:
                    if orig_sr != sample_rate:
                        loaded_audio = torchaudio.functional.resample(
                            loaded_audio, orig_sr, sample_rate
                        )
                    loaded_audio = loaded_audio.to(device=device, dtype=dtype)

                    # Adjust channels
                    amb_ch = loaded_audio.shape[0]
                    if amb_ch < target_channels:
                        reps = (target_channels + amb_ch - 1) // amb_ch
                        loaded_audio = loaded_audio.repeat(reps, 1)[:target_channels]
                    elif amb_ch > target_channels:
                        loaded_audio = loaded_audio[:target_channels]

                    # Adjust sample length (tile/loop or crop)
                    amb_len = loaded_audio.shape[-1]
                    if amb_len < target_samples:
                        reps = (target_samples + amb_len - 1) // amb_len
                        loaded_audio = loaded_audio.repeat(1, reps)[
                            ..., :target_samples
                        ]
                    else:
                        loaded_audio = loaded_audio[..., :target_samples]

                    rms = torch.sqrt(
                        torch.mean(loaded_audio**2, dim=-1, keepdim=True) + 1e-12
                    )
                    return loaded_audio / rms
            except Exception:
                pass

        # Fallback to low-frequency noise
        return self._generate_low_frequency_noise(
            target_channels, target_samples, device, dtype
        )

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Inject additive noise into the waveform at the configured SNR.
        """
        orig_dim = waveform.dim()
        if orig_dim == 1:
            x = waveform.unsqueeze(0)
        else:
            x = waveform

        px = torch.mean(x**2)
        if px == 0:
            return waveform.clone()

        num_channels, num_samples = x.shape[0], x.shape[-1]

        if self.noise_type == "white":
            eta = torch.randn_like(x)
        elif self.noise_type == "pink":
            eta = self._generate_pink_noise(
                num_channels, num_samples, x.device, x.dtype
            )
        elif self.noise_type == "ambient":
            eta = self._load_ambient_audio(
                num_channels, num_samples, sample_rate, x.device, x.dtype
            )
        else:
            eta = torch.randn_like(x)

        p_eta = torch.mean(eta**2) + 1e-12
        alpha = torch.sqrt(px / (p_eta * (10.0 ** (self.snr_db / 10.0))))
        noisy = x + alpha * eta
        noisy = torch.clamp(noisy, -1.0, 1.0)

        if orig_dim == 1:
            return noisy.squeeze(0)
        return noisy


class TimeWarp(AudioTransform):
    """
    Time-stretches/warps audio rate while strictly preserving pitch.

    Uses Phase Vocoder / STFT time-stretching (torchaudio.transforms.TimeStretch)
    so that speed/tempo changes (e.g. 0.85x to 1.15x) do not shift fundamental frequencies.
    """

    def __init__(
        self,
        rate: float = 1.0,
        n_fft: int = 512,
        hop_length: Optional[int] = None,
    ) -> None:
        """
        Initialize TimeWarp transform.

        Args:
            rate: Speed modification factor (e.g., 0.85 = slower/longer, 1.15 = faster/shorter).
            n_fft: STFT analysis window size.
            hop_length: STFT hop length (defaults to n_fft // 4).
        """
        super().__init__()
        if rate <= 0:
            raise ValueError(f"TimeWarp rate must be strictly positive, got {rate}")
        self.rate = float(rate)
        self.n_fft = int(n_fft)
        self.hop_length = (
            int(hop_length) if hop_length is not None else (self.n_fft // 4)
        )

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Apply pitch-preserving time warp to waveform.
        """
        if abs(self.rate - 1.0) < 1e-5:
            return waveform.clone()

        orig_shape = waveform.shape
        orig_dim = waveform.dim()
        orig_dtype = waveform.dtype
        if orig_dim == 1:
            x = waveform.unsqueeze(0)
        elif orig_dim == 2:
            x = waveform
        else:
            x = waveform.reshape(-1, orig_shape[-1])

        # torchaudio TimeStretch requires float32 or float64
        compute_dtype = (
            torch.float32
            if orig_dtype not in (torch.float32, torch.float64)
            else orig_dtype
        )
        x_comp = x.to(dtype=compute_dtype)

        # Dynamic FFT configuration for very short signals
        num_samples = x_comp.shape[-1]
        n_fft = self.n_fft
        hop_length = self.hop_length

        if num_samples < n_fft:
            pad_amount = n_fft - num_samples
            x_comp = torch.nn.functional.pad(x_comp, (0, pad_amount))

        win_length = n_fft
        window = torch.hann_window(win_length, device=x_comp.device, dtype=x_comp.dtype)

        stft = torch.stft(
            x_comp,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        )

        ts = T.TimeStretch(hop_length=hop_length, n_freq=n_fft // 2 + 1)
        stretched_stft = ts(stft, self.rate)

        stretched = torch.istft(
            stretched_stft,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
        )

        stretched = stretched.to(dtype=orig_dtype)

        if orig_dim == 1:
            return stretched.squeeze(0)
        elif orig_dim == 2:
            return stretched
        else:
            return stretched.reshape(*orig_shape[:-1], stretched.shape[-1])


class AcousticFilter(AudioTransform):
    """
    Butterworth bandpass acoustic filter (e.g., standard telephone 300 - 3400 Hz passband).

    Applies an order-N Butterworth bandpass filter using Second-Order Sections (SOS)
    for numerical stability across arbitrary sample rates.
    """

    def __init__(
        self,
        lowcut: float = 300.0,
        highcut: float = 3400.0,
        order: int = 4,
    ) -> None:
        """
        Initialize AcousticFilter transform.

        Args:
            lowcut: Lower cutoff frequency in Hz.
            highcut: Upper cutoff frequency in Hz.
            order: Butterworth filter order.
        """
        super().__init__()
        if lowcut <= 0:
            raise ValueError(f"lowcut must be positive, got {lowcut}")
        if highcut <= lowcut:
            raise ValueError(
                f"highcut ({highcut}) must be greater than lowcut ({lowcut})"
            )
        self.lowcut = float(lowcut)
        self.highcut = float(highcut)
        self.order = int(order)

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Filter waveform using Butterworth bandpass SOS filter.
        """
        fs = float(sample_rate)
        nyquist = 0.5 * fs

        # Clamp cutoffs within valid Nyquist limits
        low = max(1.0, min(self.lowcut, nyquist - 10.0))
        high = min(self.highcut, nyquist - 1.0)

        if low >= high:
            low = max(1.0, nyquist * 0.05)
            high = max(low + 10.0, nyquist * 0.95)

        sos = scipy.signal.butter(
            self.order,
            [low, high],
            btype="bandpass",
            fs=fs,
            output="sos",
        )

        orig_device = waveform.device
        orig_dtype = waveform.dtype

        x_np = waveform.detach().cpu().numpy()
        filtered_np = scipy.signal.sosfilt(sos, x_np, axis=-1)

        filtered = torch.from_numpy(np.ascontiguousarray(filtered_np)).to(
            device=orig_device, dtype=orig_dtype
        )
        return filtered


class ComposeTransforms(AudioTransform):
    """
    Sequentially applies a series of AudioTransform operations.
    """

    def __init__(self, transforms: Sequence[AudioTransform]) -> None:
        """
        Initialize ComposeTransforms pipeline.

        Args:
            transforms: Sequence of AudioTransform instances.
        """
        super().__init__()
        self.transforms: List[AudioTransform] = list(transforms)

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Sequentially execute all child transforms on waveform.
        """
        out = waveform
        for transform in self.transforms:
            out = transform(out, sample_rate)
        return out

    def __repr__(self) -> str:
        transform_names = ", ".join(t.__class__.__name__ for t in self.transforms)
        return f"ComposeTransforms([{transform_names}])"
